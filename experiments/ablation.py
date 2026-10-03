from experiments.runner import run_arms


def run_ablation(cfg, client, target_keys, arms=None):
    """Runs every arm (A to F). Finished arms are skipped automatically, so this is safe
    to call after baseline and adaptive."""
    run_arms(cfg, client, arms or cfg.exp.arms, target_keys)