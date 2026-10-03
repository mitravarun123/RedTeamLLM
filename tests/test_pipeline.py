from unittest.mock import Mock

import pytest

from run_pipeline import PipelineClient
from utils.hf_client import MultiClient


def test_payment_error_stops_following_requests(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    client = PipelineClient()
    error = RuntimeError("payment required")
    error.status_code = 402
    chat = Mock(side_effect=error)
    monkeypatch.setattr(MultiClient, "chat", chat)
    spec = Mock(id="org/model")
    with pytest.raises(RuntimeError, match="payment required"):
        client.chat(spec, [])
    with pytest.raises(RuntimeError, match="HTTP 402"):
        client.chat(spec, [])
    assert chat.call_count == 1
