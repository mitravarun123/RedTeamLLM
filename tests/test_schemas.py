import pytest

from utils.schemas import (Evaluation, Record, TargetResponse, TestCase, Turn,
                           append_jsonl, load_records, read_jsonl, save_records)


def make_test(**kw):
    base = dict(test_id="t1", experiment_id="e", arm="E", seed=0, round_id=1, category="escalation",
                principle="P01", conversation=[Turn("user", "a"), Turn("user", "b")])
    base.update(kw)
    return TestCase(**base)


def test_testcase_roundtrip_and_props():
    t = make_test()
    assert t.is_multi_turn and t.prompt == "b" and t.mode == "adaptive"
    assert TestCase.from_dict(t.to_dict()) == t


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        make_test(arm="Z")
    with pytest.raises(ValueError):
        make_test(category="nope")
    with pytest.raises(ValueError):
        Turn("robot", "x")


def test_evaluation_clamps_and_success():
    e = Evaluation(test_id="t", violation=True, severity=9, confidence=3.0, reason="r")
    assert e.severity == 4 and e.confidence == 1.0 and e.is_success(1)
    assert not Evaluation(test_id="t", violation=False, severity=0, confidence=0.5, reason="").is_success()
    assert not Evaluation(test_id="t", violation=True, severity=2, confidence=0.5, reason="",
                          parse_ok=False).is_success()


def test_record_jsonl_roundtrip(tmp_path):
    rec = Record(make_test(), TargetResponse("t1", "hi", "m"), Evaluation("t1", True, 2, 0.8, "why"))
    path = tmp_path / "x.jsonl"
    save_records(path, [rec, rec])
    assert len(list(read_jsonl(path))) == 2
    assert load_records(path)[0].evaluation.severity == 2
    assert rec.to_row()["round"] == 1