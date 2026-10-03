"""Agent 2: the model under evaluation. Multi-turn tests are played out live."""

from __future__ import annotations

from utils.config import Config, ModelSpec
from utils.parser import strip_think
from utils.schemas import TargetResponse, TestCase, Turn
from utils.seeding import derive_seed


def _needs_rollout(conv: list) -> bool:
    users = sum(1 for t in conv if t.role == "user")
    return users > 1 and not any(t.role == "assistant" for t in conv)


class TargetAgent:
    def __init__(self, client, spec: ModelSpec, cfg: Config):
        self.client, self.spec, self.cfg = client, spec, cfg

    def _call(self, turns: list, seed: int):
        msgs = [t.to_dict() for t in turns]
        sys_prompt = self.cfg.exp.target_system_prompt
        if sys_prompt and not any(t.role == "system" for t in turns):
            msgs.insert(0, {"role": "system", "content": sys_prompt})
        res = self.client.chat(
            self.spec, msgs, temperature=self.cfg.exp.target_temperature,
            max_tokens=self.cfg.exp.target_max_tokens, seed=seed,
        )
        return res, strip_think(res.content)

    def respond(self, test: TestCase) -> TargetResponse:
        conv = test.conversation
        if not conv:
            return TargetResponse(test.test_id, "", self.spec.id, error="empty_conversation")
        base = derive_seed(test.test_id, self.spec.key)
        p_tok = c_tok = 0
        lat = 0.0
        text = ""
        try:
            if _needs_rollout(conv):
                realized = [t for t in conv if t.role == "system"]
                users = [t for t in conv if t.role == "user"]
                for i, user_turn in enumerate(users):
                    realized.append(user_turn)
                    res, text = self._call(realized, base + i)
                    p_tok, c_tok, lat = p_tok + res.prompt_tokens, c_tok + res.completion_tokens, lat + res.latency_s
                    if i < len(users) - 1:
                        realized.append(Turn("assistant", text))
                test.conversation = realized  # store the transcript that was really sent
            else:
                res, text = self._call(conv, base)
                p_tok, c_tok, lat = res.prompt_tokens, res.completion_tokens, res.latency_s
        except Exception as exc:  # noqa: BLE001
            return TargetResponse(test.test_id, "", self.spec.id, p_tok, c_tok, lat,
                                  error=f"{type(exc).__name__}: {exc}")
        if not text.strip():
            return TargetResponse(test.test_id, "", self.spec.id, p_tok, c_tok, lat, error="empty_response")
        return TargetResponse(test.test_id, text, self.spec.id, p_tok, c_tok, lat)