import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hf_client import ChatResult  # noqa: E402


class FakeClient:
    """fn(spec, messages) -> reply text. Records every call in .calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def chat(self, spec, messages, **kwargs):
        self.calls.append(messages)
        return ChatResult(content=self.fn(spec, messages), model=spec.id)