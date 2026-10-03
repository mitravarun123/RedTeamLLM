from utils.parser import extract_json, parse_evaluation, render_transcript, strip_think
from utils.refusal_detector import is_refusal
from utils.schemas import Turn


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract_json('<think>{"x": 0}</think>{"a": {"b": "}"}}') == {"a": {"b": "}"}}
    assert extract_json('{"p": "line one\nline two"}')["p"].startswith("line one")
    assert extract_json("no json here") is None


def test_strip_think_unterminated():
    assert strip_think("hello <think>still thinking") == "hello"


def test_parse_evaluation_cases():
    ev = parse_evaluation('{"violation": "true", "severity": 3, "confidence": 85, "reason": "x"}', "t")
    assert ev.violation and ev.severity == 3 and ev.confidence == 0.85 and ev.parse_ok
    ev = parse_evaluation('{"violation": false, "severity": 4, "confidence": 0.9, "reason": ""}', "t")
    assert ev.severity == 0
    assert not parse_evaluation("garbage", "t").parse_ok


def test_refusal_and_transcript():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert not is_refusal('{"prompt": "I can\'t sleep, help me"}')
    assert render_transcript([Turn("system", "s"), Turn("user", "u")]) == "[system] s\n[user] u"