from agents.evaluator import SafetyEvaluator
from tests.conftest import FakeClient
from utils.config import load_config
from utils.schemas import TargetResponse, TestCase, Turn


def make_test():
    return TestCase(test_id="t", experiment_id="e", arm="A", seed=0, round_id=1, category="roleplay",
                    principle="P03", conversation=[Turn("user", "hello")])


def test_parses_verdict():
    cfg = load_config()
    client = FakeClient(lambda s, m: '```json\n{"violation": true, "severity": 3, "confidence": 0.8, "reason": "x"}\n```')
    ev = SafetyEvaluator(client, cfg).evaluate(make_test(), TargetResponse("t", "some reply", "m"))
    assert ev.violation and ev.severity == 3 and ev.parse_ok and ev.principle == "P03"


def test_retries_once_then_flags_parse_failure():
    cfg = load_config()
    client = FakeClient(lambda s, m: "not json")
    ev = SafetyEvaluator(client, cfg).evaluate(make_test(), TargetResponse("t", "reply", "m"))
    assert not ev.parse_ok and len(client.calls) == 2


def test_target_error_skips_judge():
    cfg = load_config()
    client = FakeClient(lambda s, m: "{}")
    ev = SafetyEvaluator(client, cfg).evaluate(make_test(), TargetResponse("t", "", "m", error="boom"))
    assert not ev.parse_ok and not client.calls