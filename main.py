#!/usr/bin/env python
"""Entry point.

  python main.py --experiment baseline      # arm A
  python main.py --experiment adaptive      # arm E
  python main.py --experiment ablation      # arms A-F (finished arms are skipped)
  python main.py --experiment transfer
  python main.py --experiment validate --validate-step export|crossjudge|score
  python main.py --analyze
Add --smoke for a tiny run and --mock to run offline with no API key.
"""

from __future__ import annotations

import argparse
import json

from utils.config import load_config
from utils.logger import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive automated red teaming pipeline")
    p.add_argument("--experiment", choices=["baseline", "adaptive", "ablation", "transfer", "validate"])
    p.add_argument("--analyze", action="store_true", help="build tables and figures from stored results")
    p.add_argument("--validate-step", choices=["export", "crossjudge", "score"], default="export")
    p.add_argument("--targets", nargs="+", help="target model keys from models.yaml")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--tests-per-round", type=int)
    p.add_argument("--config-dir")
    p.add_argument("--smoke", action="store_true", help="2 rounds x 4 tests, 1 seed, separate output folder")
    p.add_argument("--mock", action="store_true", help="offline fake models, writes to a separate *_mock folder")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.experiment and not args.analyze:
        parser.error("choose --experiment and/or --analyze")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = load_config(args.config_dir)
    over: dict = {}
    if args.rounds:
        over["rounds"] = args.rounds
    if args.tests_per_round:
        over["tests_per_round"] = args.tests_per_round
    if args.seeds:
        over["seeds"] = args.seeds
    if args.smoke:
        over.update(rounds=2, tests_per_round=4, seeds=[0], max_workers=2)
    name = cfg.exp.name + ("_smoke" if args.smoke else "") + ("_mock" if args.mock else "")
    cfg = cfg.with_exp(name=name, **over)
    setup_logging(log_file=cfg.results_dir() / "logs" / f"{cfg.exp.name}.log")
    log = get_logger("main")
    targets = args.targets or cfg.targets
    log.info("Experiment '%s' | targets=%s | seeds=%s | rounds=%d x %d tests",
             cfg.exp.name, targets, cfg.exp.seeds, cfg.exp.rounds, cfg.exp.tests_per_round)

    if args.experiment:
        from utils.hf_client import MultiClient, MockClient

        client = MockClient() if args.mock else MultiClient()
        if args.experiment == "baseline":
            from experiments.baseline import run_baseline
            run_baseline(cfg, client, targets)
        elif args.experiment == "adaptive":
            from experiments.adaptive import run_adaptive
            run_adaptive(cfg, client, targets)
        elif args.experiment == "ablation":
            from experiments.ablation import run_ablation
            run_ablation(cfg, client, targets)
        elif args.experiment == "transfer":
            from experiments.transfer import run_transfer
            for src in targets:
                run_transfer(cfg, client, src, targets)
        elif args.experiment == "validate":
            from experiments.judge_validation import export_for_labeling, run_cross_judge, score_agreement
            if args.validate_step == "export":
                print("Wrote:", export_for_labeling(cfg))
            elif args.validate_step == "crossjudge":
                print("Wrote:", run_cross_judge(cfg, client))
            else:
                print(json.dumps(score_agreement(cfg), indent=2, default=str))

    if args.analyze:
        from analysis.report import analyze
        analyze(cfg)


if __name__ == "__main__":
    main()