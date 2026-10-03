from experiments.runner import run_arms


def run_adaptive(cfg, client, target_keys):
    """Proposed method: arm E, full in-context feedback (tests, replies, scores, explanations)."""
    run_arms(cfg, client, ["E"], target_keys)