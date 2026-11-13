# pullgym

**Do AI agents report honest error bars?**

pullgym is an evaluation and RL environment. An agent gets data from a simulated physics experiment, analyses it with a shell and Python, and reports a measured value with its uncertainty. The environment knows the true value. It scores the agent on whether its error bars are honest: not too small (overconfident) and not too large (uselessly safe).

The question it answers:

> When an agent reports a 68% interval on a measured quantity, how often does the true value fall inside it? A calibrated analyst is right 68% of the time.

> **Status: pre-release (v0.1 in progress).** Three experiment families, the scorer, the Harbor exporter and the reporting tool work, and each family's reference analysis is verified to be calibrated. There are no model results yet. Interfaces may change before v0.1.

## Why

Agents are increasingly asked to analyse data, not only write code. A wrong central value is easy to catch. A correct central value with an error bar three times too small is not, and it is the failure that matters most in experimental science.

Existing work measures nearby things:

- **Confidence in task success.** Agent uncertainty-quantification work asks whether the agent knows that it solved the task. It does not ask whether its error bar on a measured quantity is right.
- **Distributional estimates without data.** [OpenEstimate](https://arxiv.org/abs/2510.15096) asks models for priors on real-world statistics and finds them often overconfident. The model gets no data and no tools.
- **Statistics question answering.** [StatABench](https://arxiv.org/abs/2606.22977) scores answers to statistics questions and modelling tasks.
- **Physics with simulators.** [PhysGym](https://arxiv.org/abs/2507.15550), PhysAgentGym and WorldSolver test law discovery or simulation code. [Collider-Bench](https://arxiv.org/abs/2605.13950) tests reproduction of LHC analyses.

pullgym scores the uncertainty itself, against a known truth, over many independent episodes. All scoring is computed from numbers, with no LLM judge.

## Quickstart

```bash
git clone https://github.com/kekeandzeyu/pullgym && cd pullgym
uv sync

uv run pullgym list                      # the experiment families
uv run pullgym calibrate --n 2000        # check every reference analysis is calibrated
uv run pullgym probe --n 500             # degenerate answering strategies vs the reference
uv run pullgym export --n 100 --out tasks    # Harbor task folders, dev split
```

Then run an agent on the tasks with [Harbor](https://www.harborframework.com/) and tabulate the results:

```bash
harbor run -p tasks -a <agent> -m <model> -o jobs
uv run pullgym report jobs               # coverage, pull and sharpness per model and family
```

## How an episode works

1. A **generator** draws hidden parameters θ from a prior, using a seed.
2. It simulates the experiment and writes the data files and an **apparatus note** to `/app/data`. The note gives the instrument resolution, calibration information and known quirks, as a lab logbook would.
3. The **agent** analyses the data however it likes, with Python, numpy, scipy, pandas and matplotlib available.
4. The agent writes its result to `/logs/artifacts/answer.json`.
5. The **verifier** runs in a separate, fresh sandbox, receives only that file, compares it with θ and writes a reward.

The physics is undergraduate lab level. The difficulty is the statistics: nuisance parameters, systematics, correlated errors, low counts and the cases where the honest answer is "not measured".

## Answer format

For each quantity, the agent reports five quantiles of its belief, at 2.5%, 16%, 50%, 84% and 97.5%:

```json
{
  "lifetime": {"q025": 21.3, "q16": 22.4, "q50": 23.0, "q84": 23.6, "q975": 24.7}
}
```

A Gaussian result `x ± σ` is close enough to `x − 2σ, x − σ, x, x + σ, x + 2σ`. Asymmetric low-count intervals and upper limits fit the same format. Quantiles must be finite numbers in strictly increasing order.

An answer that is missing or malformed is scored as if the agent had reported the prior (the quantiles an analyst who never looked at the data could honestly give), minus a penalty of 1. Giving up is therefore never better than an honest, uninformative answer.

## Scoring

| Level | Metric | What it tells you |
|---|---|---|
| Per episode (the reward) | `1 − L / L₀`, where `L` is the mean quantile (pinball) loss divided by the reference analysis's σ for that episode, and `L₀ ≈ 0.200` the expected loss of an exactly calibrated Gaussian as sharp as the reference | A proper scoring rule: the expected reward is highest when the reported quantiles match the agent's actual belief. The reference analysis averages about 0, a perfect point estimate would get 1, and worse answers go negative. |
| Per family | Coverage of the 68% (`q16`–`q84`) and 95% (`q025`–`q975`) intervals | The headline numbers: 68% and 95% when calibrated. |
| Per family | Pull = (`q50` − truth) / σ, with σ taken from the half-interval on the truth's side. Reported: mean, RMS and a robust width (half the 16–84 percentile range of the pulls) | The numbers physicists read: mean 0 and width 1 when calibrated. The robust width is reported alongside the RMS because a single wildly overconfident episode dominates the RMS. |
| Per family | Sharpness: the agent's σ divided by the reference analysis's σ | Separates "honest but sloppy" from "honest and efficient". |

`reward.json` holds the per-episode numbers that Harbor aggregates. `details.json` holds the truth, the answer and any validation error, and is what `pullgym report` reads.

## Task families

| Tier | What makes it hard | Families |
|---|---|---|
| 1. Clean statistics | Nothing; a sanity check | **`line_fit`**: a cart's velocity from positions with known Gaussian errors. Planned: weighted mean; Poisson counting rate |
| 2. Nuisance parameters | Something must be fitted or subtracted | **`decay_background`**: a lifetime from a binned decay curve over a flat background of unknown rate. Planned: peak over a sloped background; low counts where the Gaussian approximation fails |
| 3. Systematics | The statistical error alone is wrong | **`voltmeter_gain`**: a voltage read many times on a meter whose gain has a certified uncertainty. Planned: correlated errors; errors in both x and y; stated errors that are too small; outliers; detector dead time |
| 4. Traps | The honest answer is a limit, or "this is not measured" | Planned: no signal present; many places to look for a peak; inconsistent units in the apparatus note |

v0.1 targets 12 families.

### Reference analyses are calibrated

Every family ships a reference analysis written by a human. Before any model is run, it must reach nominal coverage on the family's own generator. Numbers below come from `pullgym calibrate --n 2000`; the standard error is 1.0 point on 68% coverage and 0.5 points on 95%.

| Family | Coverage 68% | Coverage 95% | Pull mean | Pull RMS | Pull robust width |
|---|---|---|---|---|---|
| `line_fit` | 66.9% | 94.9% | +0.01 | 1.01 | 1.01 |
| `decay_background` | 67.5% | 94.2% | +0.02 | 1.03 | 1.01 |
| `voltmeter_gain` | 69.3% | 94.8% | −0.02 | 1.00 | 0.97 |

The test suite repeats this check with fewer seeds on every run.

### Adding a family

A family is one Python class in `src/pullgym/families/`:

```python
class MyExperiment(Family):
    name = "my_experiment"
    tier = 2
    title = "..."
    quantities = {"g": Quantity("m/s^2", "local gravitational acceleration")}
    task = "One paragraph telling the agent what to determine."

    def sample_theta(self, rng) -> dict: ...                # hidden parameters, from a prior
    def truth(self, theta) -> dict: ...                     # requested quantities from theta
    def simulate(self, theta, rng) -> dict[str, str]: ...   # data files the agent sees
    def apparatus_note(self, theta) -> str: ...             # the lab logbook; must not leak theta
    def reference_analysis(self, files) -> dict: ...        # five quantiles per quantity
```

Register it in `src/pullgym/families/__init__.py`. The test suite then checks it automatically: that it is deterministic, that the reference analysis is calibrated, that degenerate strategies score below the reference, and that the truth does not appear in visible files.

## Splits and hack resistance

The generator is the source of truth. Task folders are built from it for a chosen list of seeds.

- **Dev split:** seeds 0, 1, 2, … are public. Use it to develop agents and to reproduce published numbers.
- **Test split:** seeds derived from a private secret (`pullgym export --split test --secret-file ...`), with opaque task names, regenerated for each release. Published results use this split. Because the generator is procedural, a public answer key for the dev split does not reveal the test split.
- Verification runs in Harbor's separate mode: a fresh sandbox that does not inherit the agent's filesystem. The only thing passed across is `/logs/artifacts/answer.json`. The true values and the reference analysis exist only in the verifier image.
- The verifier is standard-library Python. It rejects symlinks, files over 1 MB, NaN and infinity, non-numeric values, values with magnitude above 10³⁰, and unordered quantiles, and it always writes finite rewards.
- `pullgym probe` checks that the reference beats degenerate strategies: intervals too narrow or too wide, a shifted centre, and ignoring the data. The full probe results are published with each release.

## Instruction variants

Task instructions come in two variants, chosen at export (`--variant`):

- `calibrated`: says that the answer is scored on calibration and explains what that means.
- `neutral`: asks for the result and its uncertainty as quantiles, with no mention of scoring.

How the question is asked changes the answers, so results are always reported per variant.

## Results

No model results yet. The first table will report coverage and pull per family for several frontier agents, alongside the reference analysis.

## Development

```bash
uv sync
uv run pytest
```

## Contributing

New experiment families are the most useful contribution, especially systematics that real labs run into. Open an issue that describes the experiment, what makes its uncertainty hard and how you would analyse it, or send a pull request with a family class that passes the test suite.

## License

[MIT](LICENSE)
