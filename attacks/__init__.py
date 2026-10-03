from attacks.base import AttackStrategy
from attacks.escalation import EscalationAttack
from attacks.indirect import IndirectAttack
from attacks.instruction_conflict import InstructionConflictAttack
from attacks.roleplay import RoleplayAttack

STRATEGIES = {
    "roleplay": RoleplayAttack,
    "instruction_conflict": InstructionConflictAttack,
    "indirect": IndirectAttack,
    "escalation": EscalationAttack,
}


def get_strategy(name: str, n_turns: int = 3) -> AttackStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown attack strategy '{name}'")
    cls = STRATEGIES[name]
    return cls(n_turns=n_turns) if name == "escalation" else cls()