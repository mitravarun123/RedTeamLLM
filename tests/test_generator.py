import re

import pytest

from agents.generator import RedTeamGenerator, is_valid_test
from tests.conftest import FakeClient
from utils.config import load_config

REPLIES = {
    "roleplay": '{"prompt": "story prompt"}',
    "indirect": '{"prompt": "hidden request"}',
    "instruction_conflict": '{"system": "be strict", "user": "ignore that"}',
    "escalation": '{"turns": ["hi", "more", "now the request"]}',
}


def reply(spec, messages):
    name = re.search(r"TEST STRATEGY: (\w+)", messages[-1]["content"]).group(1)
    return REPLIES[name]


def gen(client, cfg, category="roleplay", ctx=""):
    return RedTeamGenerator(client, cfg).generate(
        test_id="t", experiment_id="e", arm="A", seed=0, round_id=1, category=category,
        principle_id="P01", topic="topic", feedback_context=ctx)


@pytest.mark.parametrize("category", list(REPLIES))
def test_all_strategies_produce_conversations(category):
    test = gen(FakeClient(reply), load_config(), category)
    assert is_valid_test(test)
    assert test.is_multi_turn == (category == "escalation")


def test_refusal_is_flagged_not_dropped():
    test = gen(FakeClient(lambda s, m: "I'm sorry, but I can't help with that."), load_config())
    assert test.generator_refused and not is_valid_test(test) and test.raw_generator_output


def test_feedback_context_reaches_prompt():
    client = FakeClient(reply)
    gen(client, load_config(), ctx="PREVIOUS ATTEMPTS marker")
    assert "PREVIOUS ATTEMPTS marker" in client.calls[0][-1]["content"]