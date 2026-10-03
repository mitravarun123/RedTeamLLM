import hashlib
import csv

import pytest

from run_local_pipeline import cleanup_weights, local_config
from utils.local_client import output_schema
from utils.local_client import LocalClient
from unittest.mock import Mock


def test_local_config_has_exactly_240_slots(tmp_path):
    cfg, weights = local_config(tmp_path)
    assert len(cfg.targets) * len(cfg.exp.arms) * len(cfg.exp.seeds) * cfg.exp.rounds * cfg.exp.tests_per_round == 240
    assert cfg.exp.max_workers == 1 and cfg.exp.human_label_sample_size == 240
    assert len(weights) == 3 and all(s.provider == "local" for s in cfg.models.values())


def test_schema_applies_only_to_authors_and_judges():
    assert output_schema([{"role": "user", "content": "hello"}]) is None
    messages = [{"role": "system", "content": "test author on an AI safety evaluation team"},
                {"role": "user", "content": "TEST STRATEGY: escalation"}]
    assert output_schema(messages)["properties"]["turns"]["minItems"] == 3
    schema = output_schema([{"role": "system", "content": "careful safety evaluator"}])
    assert schema["properties"]["violation"]["type"] == "boolean"


def test_cleanup_removes_only_verified_weights(tmp_path):
    payload = b"my weights"
    (tmp_path / "owned.gguf").write_bytes(payload)
    (tmp_path / "other.gguf").write_bytes(b"keep me")
    weights = {"model": {"filename": "owned.gguf", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    assert cleanup_weights(tmp_path, weights) == len(payload)
    assert not (tmp_path / "owned.gguf").exists()
    assert (tmp_path / "other.gguf").read_bytes() == b"keep me"


def test_cleanup_preserves_modified_weights(tmp_path):
    path = tmp_path / "owned.gguf"
    path.write_bytes(b"changed")
    weights = {"model": {"filename": "owned.gguf", "size": 7, "sha256": "incorrect"}}
    with pytest.raises(RuntimeError, match="changed"):
        cleanup_weights(tmp_path, weights)
    assert path.exists()


def test_human_export_backfills_when_all_judgments_positive(tmp_path, monkeypatch):
    import experiments.judge_validation as validation
    from tests.test_feedback_arms import item
    cfg, _ = local_config(tmp_path)
    monkeypatch.setattr(validation, "collect_records", lambda _: [item(i, True, 2) for i in range(10)])
    path = validation.export_for_labeling(cfg, n=8)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert all(row["human_violation"] == "" for row in rows)
    assert "violation" not in rows[0]


def test_stage_switch_unloads_unused_model(tmp_path):
    cfg, weights = local_config(tmp_path)
    client = LocalClient(cfg, weights, tmp_path, tmp_path)
    primary, unused = Mock(), Mock()
    unused.poll.return_value = None
    client.servers = {"qwen15": primary, "qwen05": unused}
    client.urls = {"qwen15": "primary", "qwen05": "unused"}
    client.set_active(["qwen15", "smol036"])
    unused.terminate.assert_called_once()
    unused.wait.assert_called_once()
    assert "qwen05" not in client.servers and "qwen15" in client.servers
    assert client.active_models == {"qwen15", "smol036"}
    with pytest.raises(ValueError, match="at most two"):
        client.set_active(["qwen15", "qwen05", "smol036"])
