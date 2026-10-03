from experiments.runner import collect_records, run_arms
from utils.config import load_config
from utils.hf_client import MockClient
from utils.schemas import Evaluation, Record, TargetResponse, TestCase, Turn, save_records
from experiments.runner import ExperimentRunner


def test_end_to_end_with_mock_and_resume(tmp_path):
    cfg = load_config(data_root=tmp_path).with_exp(name="smoke", rounds=2, tests_per_round=4,
                                                    seeds=[0], max_workers=1)
    run_arms(cfg, MockClient(), ["A", "E"], ["gptoss20b"])
    recs = collect_records(cfg.raw_dir() / "runs")
    assert len(recs) == 2 * 2 * 4
    assert {r.test.arm for r in recs} == {"A", "E"}
    assert any(r.test.parent_id for r in recs if r.test.arm == "E" and r.test.round_id == 2)
    run_arms(cfg, MockClient(), ["A", "E"], ["gptoss20b"])  # finished work is skipped
    assert len(collect_records(cfg.raw_dir() / "runs")) == len(recs)


def test_resume_retries_judge_without_regenerating_target(tmp_path):
    cfg = load_config(data_root=tmp_path).with_exp(name="retry", rounds=1, tests_per_round=1)
    runner = ExperimentRunner(cfg, MockClient(), "qwen3_8b")
    test = TestCase(test_id=runner.test_id("A", 0, 1, 0), experiment_id="retry", arm="A",
                    seed=0, round_id=1, category="roleplay", principle="P01",
                    conversation=[Turn("user", "hello")])
    record = Record(test, TargetResponse(test.test_id, "original answer", "m"),
                    Evaluation(test.test_id, False, 0, 0, "failed", parse_ok=False))
    save_records(runner.run_path("A", 0), [record])
    runner.run("A", 0)
    records = collect_records(cfg.raw_dir() / "runs")
    assert len(records) == 1 and records[0].evaluation.parse_ok
    assert records[0].response.content == "original answer"
