# pullgym

**Do AI agents report honest error bars?**

pullgym is an evaluation and RL environment. An agent gets data from a simulated physics experiment, analyses it with a shell and Python, and reports a measured value with its uncertainty. The environment knows the true value. It scores the agent on whether its error bars are honest: not too small (overconfident) and not too large (uselessly safe).

The question it answers:

> When an agent reports a 1σ interval on a measured quantity, how often does the true value fall inside it? A calibrated analyst is right 68% of the time.

> **Status: pre-release.** This README describes the design. There is no code, no task set and no result yet. Interfaces may change before v0.1.

## Why

Agents are increasingly asked to analyse data, not only write code. A wrong central value is easy to catch. A correct central value with an error bar three times too small is not, and it is the failure that matters most in experimental science.

Existing work measures nearby things:

- **Confidence in task success.** Agent uncertainty-quantification work asks whether the agent knows that it solved the task. It does not ask whether its error bar on a measured quantity is right.
- **Distributional estimates without data.** [OpenEstimate](https://arxiv.org/abs/2510.15096) asks models for priors on real-world statistics and finds them often overconfident. The model gets no data and no tools.
- **Statistics question answering.** [StatABench](https://arxiv.org/abs/2606.22977) scores answers to statistics questions and modelling tasks.
- **Physics with simulators.** [PhysGym](https://arxiv.org/abs/2507.15550), PhysAgentGym and WorldSolver test law discovery or simulation code. [Collider-Bench](https://arxiv.org/abs/2605.13950) tests reproduction of LHC analyses.

pullgym scores the uncertainty itself, against a known truth, over many independent episodes. All scoring is computed from numbers, with no LLM judge.

## How an episode works

1. A **generator** draws hidden parameters θ from a prior, using a seed.
2. It simulates the experiment and writes the data files and an **apparatus note** into the agent's working directory. The note gives the instrument resolution, calibration information and known quirks, as a lab logbook would.
3. The **agent** analyses the data however it likes.
4. The agent writes `answer.json` with its result for each requested quantity.
5. The **verifier** compares the answer with θ and writes a reward.

The physics is undergraduate lab level. The difficulty is the statistics: nuisance parameters, systematics, correlated errors, low counts and the cases where the honest answer is "not measured".

## Answer format

For each quantity, the agent reports five quantiles of its belief:

```json
{
  "decay_constant": {
    "unit": "1/s",
    "q025": 0.0412, "q16": 0.0431, "q50": 0.0440, "q84": 0.0449, "q975": 0.0468
  }
}
```

A symmetric Gaussian result `x ± σ` is written as `x − 2σ, x − σ, x, x + σ, x + 2σ`, rounded from ±1.96σ. Asymmetric low-count intervals and upper limits fit the same format: for an upper limit, `q025` and `q16` sit at the physical boundary. The verifier rejects quantiles that are out of order or missing.

## Scoring

| Level | Metric | What it tells you |
|---|---|---|
| Per episode (the reward) | Quantile (pinball) loss over the five quantiles, divided by the reference analysis's 1σ width for that episode | A proper scoring rule: the expected score is best when the reported quantiles match the agent's actual belief. It is bounded and usable for RL. |
| Per family | Coverage of the 68% (q16–q84) and 95% (q025–q975) intervals | The headline numbers: 68% and 95% when calibrated. |
| Per family | Pull distribution, pull = (q50 − truth) / (half-width on the side of the truth): mean, RMS and a robust width (half the 16–84 percentile range) | The numbers physicists read: mean 0 and width 1 when calibrated. The robust width is reported alongside the RMS because a single wildly overconfident episode dominates the RMS. |
| Per family | Sharpness: reported 68% half-width divided by the reference analysis's 1σ | Separates "honest but sloppy" from "honest and efficient". |

Coverage is also reported in bins of the true parameter, so an analysis that is calibrated on average but fails near a physical boundary still shows up.

Every family ships a **reference analysis** written by a human. Before any model is run, the test suite checks that the reference analysis itself reaches nominal coverage on thousands of seeds. That check validates the generator and the verifier as well as the reference.

## Task families

| Tier | What makes it hard | Examples |
|---|---|---|
| 1. Clean statistics | Nothing; a sanity check | Straight-line fit with known Gaussian errors; weighted mean of repeated measurements; Poisson counting rate |
| 2. Nuisance parameters | Something must be fitted or subtracted | Exponential decay over a flat background; peak over a sloped background; low counts where the Gaussian approximation fails |
| 3. Systematics | The statistical error alone is wrong | Calibration offset with its own stated uncertainty; correlated errors; errors in both x and y; stated errors that are too small; outliers; detector dead time |
| 4. Traps | The honest answer is a limit, or "this is not measured" | No signal present; many places to look for a peak; inconsistent units in the apparatus note |

v0.1 targets 12 families. Each family is one Python class:

```python
class Family:
    def sample_theta(self, seed) -> dict: ...         # hidden truth
    def simulate(self, theta, seed) -> Files: ...     # data the agent sees
    def apparatus_note(self, theta) -> str: ...       # the lab logbook
    def reference_analysis(self, files) -> Answer: ... # the human baseline
```

## Splits and hack resistance

The generator is the source of truth. Task folders are built from it for a chosen list of seeds.

- **Dev split:** public seeds. Use it to develop agents and to reproduce published numbers.
- **Test split:** built from seeds that are not published, and regenerated for each release. Published results use this split. Because the generator is procedural, a public answer key for the dev split does not reveal the test split.
- Verification runs in Harbor's separate mode: a fresh sandbox that does not inherit the agent's filesystem. The only thing passed across is the declared `answer.json`. The true values and the reference analysis live only on the verifier side.
- Tasks run without network access.
- Before each release, the verifier is attacked through the one input it accepts, `answer.json`: empty, malformed or oversized files, wrong types, NaN and infinity, unordered quantiles, and constant answers or degenerate widths (enormous or vanishing). The probe results are published with the release.

## Running it (planned)

Tasks will be exported in [Harbor](https://www.harborframework.com/) format, so existing agent harnesses run them unchanged:

```bash
pullgym export --family decay_background --split dev --out tasks/
harbor run -p tasks/decay_background -a <agent> -m <model>
pullgym report runs/     # coverage, pull and sharpness tables per family
```

An adapter for Prime Intellect's [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) is planned after v0.1.

The task instruction states that the answer is scored on calibration, and gives the answer format. Results are always reported with the exact instruction text used, because how the question is asked changes the answers.

## Results

None yet. The first table will report coverage and pull per family for several frontier agents, alongside the reference analysis.

## Contributing

New experiment families are the most useful contribution, especially systematics that real labs run into. A contributor guide will come with v0.1. Until then, opening an issue that describes the experiment, what makes its uncertainty hard and how you would analyse it is welcome.

## License

To be decided before the first release.
