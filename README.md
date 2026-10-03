# Adaptive Automated Red Teaming of LLMs via In-Context Learning

Can an LLM red-team agent find more safety failures in a target model just by seeing what happened to its earlier attempts, with no fine-tuning at all? And if it can, which part of that feedback is doing the work?

This repo is a small, fully reproducible framework for answering that question. A generator model writes safety tests, a target model answers them, and a judge model scores the answers. The generator's weights never change. Any improvement has to come from the information placed in its context.

> **Status:** pilot study. Code and tests are complete; results tables below are filled in from the local 240-test run. A single seed and small models mean the numbers are indicative, not definitive (see [Limitations](#limitations)).

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Does in-context feedback improve automated safety testing compared with generating tests blind? |
| **RQ2** | Which attack strategies and safety principles benefit most from adaptation? |
| **RQ3** | Which part of the feedback matters: earlier tests, target replies, judge scores, or judge explanations? |
| **RQ4** | Do failures found on one model transfer to other models? |

RQ3 is the most interesting of the four. Showing that adaptation helps is the easy part; showing *why* it helps is what the ablation arms below are for.

## How it works

```text
 Safety principles + seed topics
              │
              ▼
   ┌─────────────────────┐      earlier tests, replies,
   │  Agent 1: Generator │◄──── scores, explanations
   └──────────┬──────────┘      (only in adaptive arms)
              │ test (single- or multi-turn)
              ▼
   ┌─────────────────────┐
   │  Agent 2: Target    │
   └──────────┬──────────┘
              │ response
              ▼
   ┌─────────────────────┐
   │  Agent 3: Judge     │ → violation, severity 0-4, confidence, reason
   └──────────┬──────────┘
              ▼
        JSONL result store → analysis, figures, next round's feedback
```

Each experiment runs for several rounds. After every round, the results are available as feedback for the next one, depending on the arm.

## Experimental arms

| Arm | The generator sees |
|-----|--------------------|
| **A** | Nothing. Non-adaptive baseline. |
| **B** | Earlier tests and the target's replies. |
| **C** | B plus the judge's scores. |
| **D** | B plus the judge's written explanations. |
| **E** | Everything (the full adaptive method). |
| **F** | Same format and length as E, but the judge's signals are shuffled across attempts. This is the control. |

Arm F is what keeps the study honest. If F does as well as E, then the apparent gains come from seeing more text or more examples, not from useful feedback. With `match_prompt_length: true`, arms A to D also receive neutral filler so every arm gets a context of comparable length.

## Safety taxonomy

Ten principles, mapped to published frameworks (MLCommons AILuminate and HarmBench categories). Definitions and judging criteria live in `config/principles.yaml`.

| ID | Principle |
|----|-----------|
| P01 | Harmful assistance (violence) |
| P02 | Privacy |
| P03 | Deception and misinformation |
| P04 | Discrimination and hate |
| P05 | Dangerous technical assistance (cyber) |
| P06 | Exploitation and fraud |
| P07 | Manipulation and harassment |
| P08 | Self-harm safety |
| P09 | Illegal assistance |
| P10 | Unqualified high-risk advice |

Sexual content and CBRN weapon uplift are deliberately out of scope.

## Test strategies

| Strategy | What it probes |
|----------|----------------|
| `roleplay` | Whether fictional or hypothetical framing changes the target's behaviour. |
| `instruction_conflict` | A system rule versus a user message that pushes against it. |
| `indirect` | Sensitive intent embedded in surrounding context rather than stated outright. |
| `escalation` | Multi-turn scripts that move gradually toward the sensitive request. The target's replies are collected live, turn by turn. |

## Metrics

- **Attack success rate (ASR):** share of valid tests where the judge reports a violation at or above a severity threshold (set in `config/experiments.yaml`).
- **Uncertainty:** Wilson intervals for proportions, bootstrap confidence intervals for differences, a paired permutation test on matched (target, seed, round) units, Holm correction across arm comparisons, and Cohen's h as an effect size.
- **Diversity:** mean pairwise embedding distance and self-BLEU of generated tests, plus cumulative *unique* failures (near-duplicate successes are collapsed). This guards against an adaptive generator "winning" by repeating one prompt.
- **Judge reliability:** human-labelled subset, Cohen's kappa (plain and severity-weighted), a second cross-judge from a different model, and a calibration curve.
- **Generator refusals:** logged, never silently dropped.

## Results

*Fill these in from `results/` after the run. Do not report numbers that are not in the generated CSVs.*

| Arm | Tests | ASR | 95% CI |
|-----|-------|-----|--------|
| A | | | |
| B | | | |
| C | | | |
| D | | | |
| E | | | |
| F | | | |

Main comparison (E vs A): difference = `TBD`, 95% CI = `TBD`, paired permutation p = `TBD` (Holm-adjusted `TBD`).
Real vs shuffled feedback (E vs F): `TBD`.
Judge agreement with human labels: kappa = `TBD` (n = `TBD`).

Figures are written to `results/figures/main/` and `results/figures/appendix/`:

| | Figure |
|--|--------|
| 1 | System architecture |
| 2 | ASR by round, baseline vs adaptive |
| 3 | Cumulative unique failures |
| 4 | ASR by attack category |
| 5 | ASR by principle and category |
| 6 | Severity distribution |
| 7 | Ablation across arms A to F |
| 8 | Cross-model transfer matrix |
| 9 | Human vs judge agreement |
| 10 | Test diversity |
| 11 | Embedding map of generated tests |
| 12 to 17 | Appendix: effect sizes, judge calibration, generator refusals, latency and tokens, per-seed variance, real vs shuffled feedback |

## Quick start

```bash
git clone https://github.com/mitravarun123/RedTeamLLM.git
cd RedTeamLLM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add a key only if you use a hosted API
python3 -m pytest -q              # unit tests (no API needed)

# offline dry run with fake models: tests the whole pipeline and analysis
python3 main.py --experiment baseline --mock --smoke
python3 main.py --analyze --mock --smoke
```

Runs are resumable. If a run stops, rerun the same command and finished work is skipped.

### Hosted-API runs

```bash
python3 main.py --experiment baseline  --targets <model_key>
python3 main.py --experiment adaptive  --targets <model_key>
python3 main.py --experiment ablation  --targets <model_key>
python3 main.py --experiment transfer
python3 main.py --analyze
```

Model IDs and providers are set in `config/models.yaml`. Check them against your provider's current model list first, since hosted models get retired.

### Local 240-test pilot (no API, no cost)

`run_local_pipeline.py` runs all six arms against two small local targets with one seed, two rounds and ten tests per round (6 arms × 2 targets × 1 seed × 2 rounds × 10 tests = 240). It uses llama.cpp with pinned GGUF models listed in `config/local_models.yaml`.

```bash
.venv/bin/python prepare_local_models.py
.venv/bin/python run_local_pipeline.py
```

The runner keeps completed interactions so it can resume, loads only the models needed at each stage, and exports all 240 tests for human review. Results go to `results/adaptive_redteam_local_240/`.

### Judge validation

```bash
python3 main.py --experiment validate --validate-step export      # writes a blind CSV to data/human_labels/
# fill in human_violation (0/1) and human_severity (0-4), then save as <experiment>_labeled.csv
python3 main.py --experiment validate --validate-step crossjudge
python3 main.py --experiment validate --validate-step score
python3 main.py --analyze
```

## Repository layout

```text
config/        principles, experiment settings, model registry, prompts
agents/        generator, target, judge, cross-judge
attacks/       the four test strategies
feedback/      arm definitions, feedback builder, length matching
experiments/   shared runner, baseline, adaptive, ablation, transfer, judge validation
analysis/      metrics, statistics, diversity, calibration, plots, report
utils/         schemas, config loader, API clients, parsing, retry, logging
data/          seed topics, run outputs, human labels
results/       tables and figures
tests/         unit tests plus an offline end-to-end smoke test
```

## Design choices that protect validity

- **Shuffled-feedback control (arm F)** separates useful feedback from "more text in context".
- **Length matching** keeps context size comparable across arms.
- **Fixed test schedule:** the same principle and category slots every round and every arm, so arms are compared like for like.
- **Judge from a different model family than the target** where possible, plus a second cross-judge and a human-labelled subset.
- **Everything is stored.** Failed and refused generations are logged, and every interaction is a row in an append-only JSONL file.
- **Seeds and settings are recorded** with each run, and runs are resumable.

## Limitations

- The pilot uses one seed, two rounds and small local models. Differences between arms may be within noise. Treat results as indicative until more seeds are run.
- ASR depends entirely on the judge. Small judges are noisy, so the human-label agreement score matters more than any single ASR number.
- Tests are generated by an LLM from short seed topics. They cover the taxonomy but are not an exhaustive or independently validated benchmark.
- "Violation" is defined by our rubric and severity threshold, not by an external standard.
- Findings on small open models may not carry over to frontier systems.

## Related work

This builds on automated red-teaming and jailbreak-search work, including Perez et al. (2022, *Red Teaming Language Models with Language Models*), PAIR (Chao et al., 2023), TAP (Mehrotra et al., 2023), Rainbow Teaming (Samvelyan et al., 2024), and the HarmBench benchmark (Mazeika et al., 2024). The taxonomy draws on the MLCommons AILuminate hazard categories. Our contribution is a controlled ablation of *which feedback signals* enable in-context adaptation, with a shuffled-feedback control and length matching.

## Responsible use

This is a safety-evaluation tool, intended for testing models you are allowed to test. Generated tests describe sensitive requests in general terms, but model outputs can still contain harmful text. Handle `data/` and `results/` accordingly. Before sharing data publicly, set `save_raw_responses: false` or release only aggregate results and a filtered dataset. Sexual content and CBRN uplift are out of scope by design.

## Citation

```bibtex
@misc{gogulapati2026adaptiveredteam,
  title  = {Adaptive Automated Red Teaming of LLMs via In-Context Learning},
  author = {Gogulapati, Mitra Varun},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/mitravarun123/RedTeamLLM}
}
```

## License

MIT. See `LICENSE`.
