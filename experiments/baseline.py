from experiments.runner import run_arms


def run_baseline(cfg, client, target_keys):
    """Control: arm A, the generator never sees earlier outcomes."""
    run_arms(cfg, client, ["A"], target_keys)