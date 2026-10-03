from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest

from utils.config import ModelSpec
from utils.hf_client import MultiClient


def test_hf_router_and_request(monkeypatch):
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1),
    ))
    constructor = Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    client = MultiClient(max_attempts=1)
    result = client.chat(ModelSpec("test", "org/model", extra={"reasoning_effort": "low"}),
                         [{"role": "user", "content": "hello"}], seed=7)
    assert constructor.call_args.kwargs["base_url"] == "https://router.huggingface.co/v1"
    assert constructor.call_args.kwargs["api_key"] == "test-token"
    assert create.call_args.kwargs["model"] == "org/model"
    assert create.call_args.kwargs["extra_body"] == {"reasoning_effort": "low"}
    assert result.content == "OK" and result.completion_tokens == 1


def test_missing_token_fails_before_network(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        MultiClient()
