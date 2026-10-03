"""Loads YAML config, prompts and seeds into typed objects."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional

import yaml

from utils.schemas import CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    key: str
    id: str
    family: str = ""
    provider: str = "hf"
    rpm: int = 20
    extra: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "adaptive_redteam_v1"
    rounds: int = 10
    tests_per_round: int = 20
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    arms: list = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F"])
    categories: list = field(default_factory=lambda: list(CATEGORIES))
    severity_threshold: int = 1
    generator_temperature: float = 0.9
    target_temperature: float = 0.7
    judge_temperature: float = 0.0
    generator_max_tokens: int = 700
    target_max_tokens: int = 1024
    judge_max_tokens: int = 1200
    max_feedback_items: int = 6
    max_response_chars: int = 600
    max_transcript_chars: int = 1500
    match_prompt_length: bool = True
    escalation_turns: int = 3
    max_generation_attempts: int = 3
    max_workers: int = 4
    target_system_prompt: str = ""
    save_raw_responses: bool = True
    human_label_sample_size: int = 150
    cross_judge_sample_size: int = 200
    dedup_similarity: float = 0.85

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Config:
    exp: ExperimentConfig
    models: dict
    roles: dict
    principles: list
    severity_scale: dict
    topics: dict
    templates: dict
    prompts: dict
    root: Path = PROJECT_ROOT

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(f"Model '{key}' not in config/models.yaml registry")
        return self.models[key]

    def principle(self, pid: str) -> dict:
        for p in self.principles:
            if p["id"] == pid:
                return p
        raise KeyError(f"Unknown principle {pid}")

    @property
    def principle_ids(self) -> list:
        return [p["id"] for p in self.principles]

    @property
    def targets(self) -> list:
        return list(self.roles.get("default_targets", []))

    def with_exp(self, **overrides) -> "Config":
        return replace(self, exp=replace(self.exp, **overrides))

    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / self.exp.name

    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    def human_dir(self) -> Path:
        return self.root / "data" / "human_labels"

    def cross_judge_dir(self) -> Path:
        return self.root / "data" / "cross_judge"

    def results_dir(self) -> Path:
        return self.root / "results" / self.exp.name

    def figures_dir(self) -> Path:
        return self.results_dir() / "figures"


def load_config(config_dir: Optional[str | Path] = None, data_root: Optional[str | Path] = None) -> Config:
    cdir = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
    exp_raw = load_yaml(cdir / "experiments.yaml") or {}
    exp = ExperimentConfig.from_dict(exp_raw.get("experiment", exp_raw))

    models_raw = load_yaml(cdir / "models.yaml")
    models = {k: ModelSpec(key=k, **v) for k, v in models_raw["registry"].items()}
    roles = models_raw.get("roles", {})
    for role in ("generator", "judge", "cross_judge"):
        if roles.get(role) not in models:
            raise ValueError(f"roles.{role} must be a key of registry in models.yaml")

    pr = load_yaml(cdir / "principles.yaml")
    severity_scale = {int(k): v for k, v in pr["severity_scale"].items()}

    seeds_path = PROJECT_ROOT / "data" / "seeds" / "seeds.yaml"
    topics = load_yaml(seeds_path) if seeds_path.exists() else {}

    templates = load_yaml(cdir / "prompts" / "feedback_templates.yaml")
    prompts = {p.stem: p.read_text(encoding="utf-8") for p in (cdir / "prompts").glob("*.txt")}

    return Config(
        exp=exp, models=models, roles=roles, principles=pr["principles"],
        severity_scale=severity_scale, topics=topics or {}, templates=templates,
        prompts=prompts, root=Path(data_root) if data_root else PROJECT_ROOT,
    )
