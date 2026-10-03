# Adaptive Automated Red Teaming of LLMs via In-Context Learning

Does an LLM red-team generator find more safety failures when it sees earlier attempts and
judge feedback in its context (no fine-tuning) than when it generates blind?

Three agents: generator -> target -> judge. Tests span four strategies (roleplay,
instruction_conflict, indirect, escalation) over ten safety principles mapped to MLCommons
AILuminate and HarmBench categories.

## Arms

| Arm | Generator sees |
|-----|----------------|
| A | nothing (non-adaptive baseline) |
| B | earlier tests + target replies |
| C | B + judge scores |
| D | B + judge explanations |
| E | everything (proposed method) |
| F | same as E but judge signals are shuffled across attempts (control) |

With `match_prompt_length: true`, arms A to D get neutral filler so context length matches E.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env              # add your HF_TOKEN
pytest -q                         # unit tests
python main.py --experiment baseline --mock --smoke && python main.py --analyze --mock --smoke
python main.py --experiment baseline --smoke      # tiny real run
python main.py --experiment baseline
python main.py --experiment adaptive
python main.py --experiment ablation
python main.py --experiment transfer
python main.py --experiment validate --validate-step export      # label data/human_labels/*_to_label.csv,
                                                                 # save as <experiment>_labeled.csv
python main.py --experiment validate --validate-step crossjudge
python main.py --experiment validate --validate-step score
python main.py --analyze          # tables in results/, figures in results/figures/
```

Runs are resumable: rerun the same command after a crash or rate-limit stop.
Check model IDs in `config/models.yaml` against the Hugging Face Inference Providers before starting.

## Local 240-test pilot

`run_local_pipeline.py` runs all six arms against two small local targets, with one seed, two rounds, and ten tests per round. It uses the project-local llama.cpp runtime and pinned GGUF files listed in `config/local_models.yaml`; no HF inference token or credits are needed. After building the runtime, run:

```bash
.venv/bin/python prepare_local_models.py
.venv/bin/python run_local_pipeline.py
```

The runner preserves completed interactions for resume, loads only the models needed at each stage, exports all 240 tests for human review, and deletes only its verified downloaded weight files after automated checks succeed. Results are stored in `results/adaptive_redteam_local_240/`; its README lists all dataset paths and human-validation instructions. A completed run can be scored against human labels without downloading weights again.

## Responsible use

Raw model outputs can contain harmful text. They are git-ignored. Before sharing data, set
`save_raw_responses: false` or release only aggregate results and a filtered dataset.
Sexual content and CBRN uplift are deliberately out of scope.
# RedTeamLLM
