import random

from feedback.builder import FeedbackBuilder
from feedback.arms import select_feedback, shuffle_feedback
from utils.config import load_config
from utils.schemas import Evaluation, FeedbackItem, TargetResponse, TestCase, Turn


def item(n, viol, sev, rnd=1, principle="P01", category="roleplay"):
    t = TestCase(test_id=f"t{n}", experiment_id="e", arm="E", seed=0, round_id=rnd, category=category,
                 principle=principle, conversation=[Turn("user", f"prompt {n}")])
    return FeedbackItem(t, TargetResponse(f"t{n}", f"reply {n}", "m"),
                        Evaluation(f"t{n}", viol, sev, 0.9, f"reason {n}"))


HISTORY = [item(1, True, 3), item(2, False, 0), item(3, True, 2), item(4, False, 0)]


def build(arm, cfg=None):
    cfg = cfg or load_config()
    return FeedbackBuilder(cfg).build(arm, HISTORY, "roleplay", "P01", random.Random(0))


def test_arm_contents():
    assert "prompt" not in build("A")[0] or "REFERENCE NOTES" in build("A")[0]
    b = build("B")[0]
    assert "reply 1" in b and "Judge" not in b
    c = build("C")[0]
    assert "Judge scores" in c and "Judge explanation" not in c
    d = build("D")[0]
    assert "Judge explanation" in d and "Judge scores" not in d
    e = build("E")[0]
    assert "Judge scores" in e and "Judge explanation" in e


def test_baseline_shows_no_attempts():
    text, parent = build("A")
    assert "Attempt" not in text and parent is None


def test_lengths_are_matched():
    cfg = load_config()
    e_len = len(build("E", cfg)[0])
    for arm in "ABCD":
        assert abs(len(build(arm, cfg)[0]) - e_len) <= 2, arm


def test_shuffle_breaks_pairing_but_keeps_signals():
    items = [item(i, i % 2 == 0, i % 4) for i in range(1, 5)]
    shuffled = shuffle_feedback(items, random.Random(1))
    assert all(s.test.test_id.replace("t", "") != s.evaluation.reason.replace("reason ", "") for s in shuffled)
    assert sorted(s.evaluation.reason for s in shuffled) == sorted(i.evaluation.reason for i in items)


def test_select_mixes_successes_and_failures():
    chosen = select_feedback(HISTORY, "roleplay", "P01", 4, random.Random(0))
    assert len(chosen) == 4
    assert select_feedback(HISTORY, "indirect", "P01", 4, random.Random(0)) == []