"""Fast zero-Docker local/Colab runner for pullgym episodes.

Materializes only ``data/`` (read-only) and ``artifacts/`` inside an ephemeral
temporary workspace per episode while keeping ``truth`` and ``ref_sigma``
strictly in parent-process memory. Writes Harbor-compatible trial directories
so ``pullgym report`` works identically on both Harbor and run-fast outputs.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from pullgym import __version__, scoring
from pullgym.export import VARIANTS, build_truth_doc, heldout_seeds, instruction, task_id
from pullgym.families import Family

FAST_DATA_DIR = "data"
FAST_ANSWER_PATH = "artifacts/answer.json"
MAX_TOOL_OUTPUT_CHARS = 16_000
_SENSITIVE_ENV_FRAGMENTS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
)
_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class AgentError(RuntimeError):
    """Unrecoverable agent or API error during an episode."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# Client errors that a model's own generation can cause mid-episode (malformed
# tool-call JSON rejected by the provider, context-window overflow). After turn 1
# they end the episode and the answer on disk is scored, so re-running with
# `--resume` cannot keep only the lucky draws. On turn 1 they almost always mean
# a configuration problem, so they still raise. 413 (provider request-size or
# rate limits) and 429 are not the model's fault and are never scored.
MODEL_ATTRIBUTABLE_HTTP = frozenset({400, 422})


@dataclass
class EpisodeWorkspace:
    """Ephemeral filesystem view exposed to an agent for a single episode."""

    root: Path
    data_dir: Path
    answer_path: Path
    tool_timeout_sec: float = 30.0
    _step_counter: int = 0

    def clean_env(self) -> dict[str, str]:
        """Minimal environment for agent subprocesses without API keys."""
        env: dict[str, str] = {}
        for k, v in os.environ.items():
            upper = k.upper()
            if any(frag in upper for frag in _SENSITIVE_ENV_FRAGMENTS):
                continue
            if k in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "VIRTUAL_ENV"}:
                env[k] = v
        py_bin = str(Path(sys.executable).parent)
        cur_path = env.get("PATH", "")
        if py_bin and (not cur_path or cur_path.split(os.pathsep)[0] != py_bin):
            env["PATH"] = f"{py_bin}{os.pathsep}{cur_path}" if cur_path else py_bin
        env["PULLGYM_DATA_DIR"] = str(self.data_dir)
        env["PULLGYM_ANSWER_PATH"] = str(self.answer_path)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        return env

    def _run_subprocess(self, cmd: list[str], timeout: float | None = None) -> dict[str, Any]:
        limit = self.tool_timeout_sec if timeout is None else timeout
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.root),
            env=self.clean_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=limit)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            stdout, stderr = proc.communicate()
        elapsed = round(time.monotonic() - t0, 3)
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_sec": elapsed,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr),
        }

    def run_python(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Execute a Python snippet inside the episode workspace."""
        self._step_counter += 1
        script = self.root / f"_step_{self._step_counter:03d}.py"
        script.write_text(code, encoding="utf-8")
        return self._run_subprocess([sys.executable, str(script)], timeout=timeout)

    def run_bash(self, command: str, timeout: float | None = None) -> dict[str, Any]:
        """Execute a shell command inside the episode workspace."""
        return self._run_subprocess(["/bin/bash", "-lc", command], timeout=timeout)


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n... [truncated {omitted} chars] ...\n{text[-half:]}"


def answer_status(answer_path: Path, quantities: list[str]) -> str:
    """Public-schema status of the answer file: "missing", "valid" or "invalid: ...".

    Uses only scoring.load_answer (JSON syntax, required keys, finite numbers,
    strictly increasing quantiles). It never sees the truth, the reference or
    the prior, so it is safe to show to the agent mid-episode.
    """
    if not answer_path.exists() and not answer_path.is_symlink():
        return "missing"
    try:
        scoring.load_answer(str(answer_path), quantities)
    except scoring.AnswerError as e:
        return f"invalid: {e}"
    return "valid"


def _set_readonly(data_dir: Path) -> None:
    for p in data_dir.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(data_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _set_writable(root: Path) -> None:
    if not root.exists():
        return
    try:
        os.chmod(root, stat.S_IRWXU)
    except OSError:
        pass
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        try:
            os.chmod(dp, stat.S_IRWXU)
        except OSError:
            pass
        for name in filenames:
            try:
                os.chmod(dp / name, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass


@dataclass
class AgentRunResult:
    """Outcome metadata returned by an agent after attempting an episode."""

    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    api_error: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)


class EpisodeAgent(Protocol):
    agent_name: str
    model_name: str | None

    def solve(
        self,
        prompt: str,
        workspace: EpisodeWorkspace,
        family: Family,
    ) -> AgentRunResult: ...


class OracleAgent:
    """Runs the family's reference analysis on the visible data files only."""

    agent_name = "oracle"
    model_name: str | None = None

    def solve(
        self,
        prompt: str,
        workspace: EpisodeWorkspace,
        family: Family,
    ) -> AgentRunResult:
        files = {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(workspace.data_dir.iterdir())
            if p.is_file() and p.name != "apparatus_note.md"
        }
        ref = family.reference_analysis(files)
        answer = {q: dict(zip(scoring.KEYS, ref[q])) for q in family.quantities}
        workspace.answer_path.parent.mkdir(parents=True, exist_ok=True)
        workspace.answer_path.write_text(json.dumps(answer, indent=2), encoding="utf-8")
        return AgentRunResult(
            turns=1,
            transcript=[{"role": "oracle", "answer": answer}],
            log_lines=["oracle: wrote reference_analysis quantiles"],
        )


class PriorAgent:
    """Baseline agent that ignores the data and reports the family prior."""

    agent_name = "prior"
    model_name: str | None = None

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, list[float]]] = {}

    def solve(
        self,
        prompt: str,
        workspace: EpisodeWorkspace,
        family: Family,
    ) -> AgentRunResult:
        if family.name not in self._cache:
            self._cache[family.name] = family.prior_quantiles()
        prior = self._cache[family.name]
        answer = {q: dict(zip(scoring.KEYS, prior[q])) for q in family.quantities}
        workspace.answer_path.parent.mkdir(parents=True, exist_ok=True)
        workspace.answer_path.write_text(json.dumps(answer, indent=2), encoding="utf-8")
        return AgentRunResult(
            turns=1,
            transcript=[{"role": "prior", "answer": answer}],
            log_lines=["prior: wrote prior_quantiles"],
        )


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python 3 code (numpy, scipy, pandas, matplotlib available) "
                "in the workspace directory. Read data files from `data/` and write "
                "your final JSON answer to `artifacts/answer.json`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python 3 source code to run."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Execute a bash command in the workspace directory (e.g., list files "
                "or inspect `data/apparatus_note.md`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT_TOOLS = (
    "You are an experimental physicist and statistician. "
    "Use `run_bash` or `run_python` to read `data/apparatus_note.md` and inspect all "
    "data files in `data/`, perform a rigorous statistical analysis that properly "
    "accounts for statistical and systematic uncertainties, and write the final "
    "five-quantile JSON answer to `artifacts/answer.json`. "
    "Verify that `artifacts/answer.json` exists, is valid JSON, and has strictly "
    "increasing finite quantiles before finishing."
)

SYSTEM_PROMPT_CODE = (
    "You are an experimental physicist and statistician. "
    "You have access to a Python 3 environment (numpy, scipy, pandas, matplotlib) "
    "whose working directory contains `data/` (including `data/apparatus_note.md`) "
    "and `artifacts/`. On each turn, output a single ```python ... ``` code block to "
    "inspect files, run calculations, and write the final JSON answer to "
    "`artifacts/answer.json`. Once `artifacts/answer.json` is written and verified, "
    "reply briefly without a code block to finish."
)


ChatFn = Callable[[dict[str, Any]], dict[str, Any]]


def openai_http_chat(
    base_url: str,
    api_key: str | None = None,
    timeout_sec: float = 180.0,
    max_retries: int = 5,
) -> ChatFn:
    """Create a stdlib-only HTTP client for OpenAI-compatible /chat/completions."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def _call(payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"pullgym/{__version__}",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                retry_after = e.headers.get("Retry-After") if e.headers else None
                last_err = AgentError(f"HTTP {e.code} from {endpoint}: {detail}", status=e.code)
                if e.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    wait = min(2.0**attempt, 30.0)
                    if retry_after:
                        try:
                            wait = max(wait, min(float(retry_after) + 1.0, 60.0))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    continue
                raise last_err from None
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as e:
                last_err = AgentError(f"Request error calling {endpoint}: {e}")
                if attempt < max_retries:
                    time.sleep(min(2.0**attempt, 30.0))
                    continue
                raise last_err from None
        raise last_err or AgentError(f"Failed calling {endpoint}")

    return _call


def _with_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` whose last message carries an ephemeral cache_control block.

    One breakpoint per request on the newest message: each turn reads the prefix
    cached by the previous turn and writes the extended one. The transcript is not
    modified.
    """
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last.get("content")
    if isinstance(content, str) and content:
        last["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        blocks = [dict(b) for b in content]
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        last["content"] = blocks
    else:
        return messages
    return [*messages[:-1], last]


class OpenAICompatibleAgent:
    """Multi-turn tool-using agent backed by any OpenAI-compatible endpoint."""

    agent_name = "pullgym-fast"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        chat_fn: ChatFn | None = None,
        max_turns: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        tool_mode: str = "tools",
        http_timeout_sec: float = 180.0,
        extra_body: dict[str, Any] | None = None,
        cache_breakpoint: bool = False,
    ) -> None:
        if tool_mode not in {"tools", "code"}:
            raise ValueError(f"unknown tool_mode {tool_mode!r}; expected 'tools' or 'code'")
        self.model_name = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.extra_body = dict(extra_body or {})
        # Mark the newest message as a prompt-cache breakpoint on every request
        # (Anthropic models, directly or via OpenRouter, cache only marked prefixes).
        self.cache_breakpoint = cache_breakpoint
        if chat_fn is not None:
            self._chat = chat_fn
        else:
            url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            self._chat = openai_http_chat(url, key, timeout_sec=http_timeout_sec)

    def _execute_tool(
        self, name: str, raw_args: str | dict[str, Any], workspace: EpisodeWorkspace
    ) -> dict[str, Any]:
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as e:
                return {"error": f"invalid tool arguments JSON: {e}"}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return {"error": "tool arguments must be a JSON object"}

        if name == "run_python":
            code = args.get("code")
            if not isinstance(code, str):
                return {"error": "run_python requires a string 'code' argument"}
            return workspace.run_python(code)
        if name == "run_bash":
            command = args.get("command")
            if not isinstance(command, str):
                return {"error": "run_bash requires a string 'command' argument"}
            return workspace.run_bash(command)
        return {"error": f"unknown tool {name!r}"}

    def solve(
        self,
        prompt: str,
        workspace: EpisodeWorkspace,
        family: Family,
    ) -> AgentRunResult:
        sys_prompt = SYSTEM_PROMPT_TOOLS if self.tool_mode == "tools" else SYSTEM_PROMPT_CODE
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        result = AgentRunResult(transcript=list(messages))
        names = list(family.quantities)

        def status() -> str:
            return answer_status(workspace.answer_path, names)

        for turn in range(1, self.max_turns + 1):
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": _with_cache_breakpoint(messages) if self.cache_breakpoint else messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.extra_body,
            }
            if self.tool_mode == "tools":
                payload["tools"] = OPENAI_TOOLS
                payload["tool_choice"] = "auto"

            try:
                resp = self._chat(payload)
            except AgentError as e:
                if turn == 1 or e.status not in MODEL_ATTRIBUTABLE_HTTP:
                    raise
                result.api_error = str(e)[:500]
                result.log_lines.append(
                    f"turn {turn}: api_error status={e.status}; episode ended "
                    f"(answer_status={status()})"
                )
                break
            result.turns = turn
            usage = resp.get("usage") or {}
            result.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            result.completion_tokens += int(usage.get("completion_tokens") or 0)
            # Prompt tokens served from the provider's prefix cache (OpenAI-style
            # prompt_tokens_details, or DeepSeek's prompt_cache_hit_tokens).
            # Billed cost when the provider reports it (OpenRouter: usage.cost, USD).
            if isinstance(usage.get("cost"), (int, float)):
                result.cost += float(usage["cost"])
            details = usage.get("prompt_tokens_details") or {}
            result.cached_tokens += int(
                details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0
            )

            choices = resp.get("choices") or []
            if not choices:
                raise AgentError("model response contained no choices")
            msg = choices[0].get("message") or {}
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
                if msg.get("reasoning_content") is not None:
                    assistant_msg["reasoning_content"] = msg["reasoning_content"]
            messages.append(assistant_msg)
            transcript_msg = dict(assistant_msg)
            if msg.get("reasoning_content") is not None:
                transcript_msg["reasoning_content"] = msg["reasoning_content"]
            result.transcript.append(transcript_msg)

            if self.tool_mode == "tools" and tool_calls:
                for idx_tc, tc in enumerate(tool_calls):
                    fn = tc.get("function") or {}
                    tname = fn.get("name", "")
                    out = self._execute_tool(tname, fn.get("arguments", "{}"), workspace)
                    if idx_tc == len(tool_calls) - 1:
                        st = status()
                        out["turn"] = f"{turn}/{self.max_turns}"
                        out["answer_status"] = st
                        note = self._budget_note(turn, st)
                        if note:
                            out["note"] = note
                    result.log_lines.append(
                        f"turn {turn}: tool={tname} exit={out.get('exit_code', 'err')}"
                    )
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{turn}"),
                        "content": json.dumps(out),
                    }
                    messages.append(tool_msg)
                    result.transcript.append(tool_msg)
                continue

            # Code-block fallback (always active in 'code' mode; also helps
            # models in 'tools' mode that emit ```python without tool_calls).
            content = assistant_msg["content"]
            blocks = _PYTHON_FENCE_RE.findall(content)
            if blocks and (self.tool_mode == "code" or status() != "valid"):
                out = workspace.run_python(blocks[-1])
                st = status()
                result.log_lines.append(
                    f"turn {turn}: code_block exit={out.get('exit_code')} answer_status={st}"
                )
                # A tools-mode model that falls back to a code block is done once
                # its answer validates; an invalid answer gets a chance to be fixed.
                if st == "valid" and self.tool_mode == "tools":
                    break
                feedback = (
                    f"Execution result (exit_code={out['exit_code']}, "
                    f"timed_out={out['timed_out']}):\n"
                    f"STDOUT:\n{out['stdout']}\n"
                    f"STDERR:\n{out['stderr']}\n"
                    f"turn: {turn}/{self.max_turns}\n"
                    f"answer_status (`{FAST_ANSWER_PATH}`): {st}"
                )
                note = self._budget_note(turn, st)
                if note:
                    feedback += f"\nNote: {note}"
                fb_msg = {"role": "user", "content": feedback}
                messages.append(fb_msg)
                result.transcript.append(fb_msg)
                continue

            # No tool calls and no python block to execute -> done.
            result.log_lines.append(f"turn {turn}: finished (answer_status={status()})")
            break

        return result

    def _budget_note(self, turn: int, st: str) -> str | None:
        """Turn-budget warning, shown only while the answer is not yet valid."""
        if st == "valid":
            return None
        left = self.max_turns - turn
        if left == 1:
            return (
                f"Final turn next ({self.max_turns}/{self.max_turns}): "
                f"write and verify `{FAST_ANSWER_PATH}` now."
            )
        if left == 3 and self.max_turns >= 6:
            return (
                f"{left} turns left: write a valid `{FAST_ANSWER_PATH}` soon; "
                "you can refine it afterwards."
            )
        return None


@dataclass
class EpisodeOutcome:
    task_name: str
    family: str
    seed: int
    split: str
    variant: str
    trial_dir: Path
    elapsed_sec: float
    rewards: dict[str, float] | None = None
    error: str | None = None
    skipped: bool = False


def run_episode(
    family: Family,
    seed: int,
    split: str,
    variant: str,
    agent: EpisodeAgent,
    job_dir: Path,
    *,
    prior: dict[str, list[float]] | None = None,
    tool_timeout_sec: float = 30.0,
    resume: bool = False,
) -> EpisodeOutcome:
    """Run and score a single episode in an isolated temporary workspace."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(VARIANTS)}")

    name = task_id(family, split, seed)
    trial_dir = job_dir / name
    details_path = trial_dir / "verifier" / "details.json"
    reward_path = trial_dir / "verifier" / "reward.json"

    if resume and details_path.is_file() and reward_path.is_file():
        rewards = json.loads(reward_path.read_text(encoding="utf-8"))
        return EpisodeOutcome(
            task_name=name,
            family=family.name,
            seed=seed,
            split=split,
            variant=variant,
            trial_dir=trial_dir,
            elapsed_sec=0.0,
            rewards=rewards,
            skipped=True,
        )

    ep = family.episode(seed)
    ref = family.reference_analysis(ep.files)
    prior_q = prior if prior is not None else family.prior_quantiles()
    truth_doc = build_truth_doc(family, ep, ref, prior_q, name, split, variant, seed)
    prompt = instruction(family, variant, data_dir=FAST_DATA_DIR, answer_path=FAST_ANSWER_PATH)

    trial_dir.mkdir(parents=True, exist_ok=True)
    agent_block = {"name": agent.agent_name, "model_name": agent.model_name}
    cfg: dict[str, Any] = {
        "task": {"name": f"pullgym/{name}", "family": family.name, "tier": family.tier},
        "trial_name": name,
        "job_id": job_dir.name,
        "split": split,
        "variant": variant,
        "agent": agent_block,
    }
    if split == "dev":
        cfg["seed"] = seed
    (trial_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"pullgym_{family.name}_") as tmp:
        root = Path(tmp)
        data_dir = root / FAST_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        for fname, text in ep.files.items():
            (data_dir / fname).write_text(text, encoding="utf-8")
        (data_dir / "apparatus_note.md").write_text(ep.note, encoding="utf-8")
        _set_readonly(data_dir)

        answer_path = root / FAST_ANSWER_PATH
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        ws = EpisodeWorkspace(
            root=root,
            data_dir=data_dir,
            answer_path=answer_path,
            tool_timeout_sec=tool_timeout_sec,
        )

        try:
            run_res = agent.solve(prompt, ws, family)
            elapsed = time.monotonic() - t0
            scored = scoring.score_episode(str(answer_path), truth_doc)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            err_text = f"{type(exc).__name__}: {exc}"
            (trial_dir / "trial.log").write_text(err_text + "\n", encoding="utf-8")
            (trial_dir / "exception.txt").write_text(err_text + "\n", encoding="utf-8")
            return EpisodeOutcome(
                task_name=name,
                family=family.name,
                seed=seed,
                split=split,
                variant=variant,
                trial_dir=trial_dir,
                elapsed_sec=elapsed,
                error=err_text,
            )
        finally:
            _set_writable(root)

    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    stale_exc = trial_dir / "exception.txt"
    if stale_exc.exists():
        stale_exc.unlink()
    (verifier_dir / "reward.json").write_text(json.dumps(scored["rewards"]), encoding="utf-8")
    (verifier_dir / "details.json").write_text(
        json.dumps(scored["details"], indent=2), encoding="utf-8"
    )
    result_doc = {
        "harness_version": __version__,
        "config": {"agent": agent_block},
        "agent_info": {
            "name": agent.agent_name,
            "model_info": {"name": agent.model_name},
        },
        "timing": {"elapsed_sec": round(elapsed, 3)},
        "usage": {
            "turns": run_res.turns,
            "prompt_tokens": run_res.prompt_tokens,
            "completion_tokens": run_res.completion_tokens,
            "cached_tokens": run_res.cached_tokens,
            **({"cost": round(run_res.cost, 6)} if run_res.cost else {}),
        },
        **({"api_error": run_res.api_error} if run_res.api_error else {}),
        "rewards": scored["rewards"],
    }
    (trial_dir / "result.json").write_text(json.dumps(result_doc, indent=2), encoding="utf-8")
    (trial_dir / "transcript.json").write_text(
        json.dumps(run_res.transcript, indent=2), encoding="utf-8"
    )
    (trial_dir / "trial.log").write_text("\n".join(run_res.log_lines) + "\n", encoding="utf-8")

    return EpisodeOutcome(
        task_name=name,
        family=family.name,
        seed=seed,
        split=split,
        variant=variant,
        trial_dir=trial_dir,
        elapsed_sec=elapsed,
        rewards=scored["rewards"],
    )


def default_job_id(agent: EpisodeAgent, variant: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = agent.model_name or agent.agent_name
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-")[:48] or "run"
    return f"{ts}_{slug}_{variant}"


def run_suite(
    families: list[Family],
    *,
    split: str = "dev",
    n: int = 10,
    start: int = 0,
    variant: str = "calibrated",
    secret: str | None = None,
    agent: EpisodeAgent,
    out_dir: Path,
    job_id: str | None = None,
    concurrency: int = 4,
    tool_timeout_sec: float = 30.0,
    resume: bool = False,
    on_complete: Callable[[int, int, EpisodeOutcome], None] | None = None,
) -> tuple[Path, list[EpisodeOutcome]]:
    """Run a batch of episodes across families with bounded concurrency."""
    if split == "test" and not secret:
        raise ValueError("secret is required for the test split")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    jid = job_id or default_job_id(agent, variant)
    job_dir = out_dir / jid
    job_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute prior quantiles once per family (20k draws each).
    priors = {fam.name: fam.prior_quantiles() for fam in families}

    specs: list[tuple[Family, int]] = []
    for fam in families:
        if split == "dev":
            seeds = list(range(start, start + n))
        else:
            assert secret is not None
            seeds = heldout_seeds(secret, fam.name, start + n)[start:]
        for seed in seeds:
            specs.append((fam, seed))

    total = len(specs)
    outcomes: list[EpisodeOutcome] = []

    def _run_one(spec: tuple[Family, int]) -> EpisodeOutcome:
        fam, seed = spec
        return run_episode(
            fam,
            seed,
            split,
            variant,
            agent,
            job_dir,
            prior=priors[fam.name],
            tool_timeout_sec=tool_timeout_sec,
            resume=resume,
        )

    if concurrency == 1 or total <= 1:
        for idx, spec in enumerate(specs, 1):
            oc = _run_one(spec)
            outcomes.append(oc)
            if on_complete:
                on_complete(idx, total, oc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run_one, spec) for spec in specs]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                oc = fut.result()
                outcomes.append(oc)
                if on_complete:
                    on_complete(idx, total, oc)

    return job_dir, outcomes
