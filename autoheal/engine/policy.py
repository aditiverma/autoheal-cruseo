"""Deterministic policy engine.

The non-negotiable gate. No matter how confident the Nemotron planner is, or
what a safety-review model says, an action only executes if it passes these
deterministic checks. This is the part that is NOT an LLM — it is the
infrastructure control that enforces allowed action types, parameter ranges,
and blast-radius limits. (In the KubeCon architecture this is where Kubernetes
RBAC / quotas / admission control live.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import PolicyDecision, RemediationAction


@dataclass
class Policy:
    # action name -> the single config parameter it is allowed to change
    allowed_actions: dict[str, str] = field(default_factory=lambda: {
        "set_learning_rate": "lr",
        "set_batch_size": "batch_size",
        "set_epochs": "epochs",
        "set_grad_clip": "grad_clip",
        "set_grad_accum": "grad_accum",
    })
    # parameter -> (inclusive min, inclusive max)
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "lr": (1e-5, 1.0),
        "batch_size": (1, 256),
        "epochs": (1, 20),          # blast-radius ceiling on compute
        "grad_clip": (0.1, 10.0),
        "grad_accum": (1, 16),
    })
    max_attempts: int = 6
    max_rejections: int = 4

    def check(self, action: RemediationAction) -> PolicyDecision:
        violations: list[str] = []

        if action.action not in self.allowed_actions:
            return PolicyDecision(
                allowed=False,
                violations=[f"action '{action.action}' not in allowlist "
                            f"{sorted(self.allowed_actions)}"],
                reason="unsupported action type",
            )

        expected_param = self.allowed_actions[action.action]
        if action.parameter != expected_param:
            violations.append(
                f"action '{action.action}' may only modify '{expected_param}', "
                f"not '{action.parameter}'"
            )

        val = action.new_value
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            violations.append(f"new_value for '{action.parameter}' must be numeric, "
                              f"got {val!r}")
        else:
            lo, hi = self.bounds.get(action.parameter, (None, None))
            if lo is not None and not (lo <= val <= hi):
                violations.append(
                    f"{action.parameter}={val} outside allowed range [{lo}, {hi}]"
                )
            if action.parameter in ("batch_size", "epochs", "grad_accum") and \
                    isinstance(val, float) and not val.is_integer():
                violations.append(f"{action.parameter} must be an integer, got {val}")

        if violations:
            return PolicyDecision(allowed=False, violations=violations,
                                  reason="policy violation")
        return PolicyDecision(allowed=True, reason="within policy")
