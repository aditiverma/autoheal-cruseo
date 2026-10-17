"""Deterministic policy engine — the agent's authorization layer.

The non-negotiable gate, and the local analogue of Kubernetes authorization.
No matter how confident the Nemotron planner is, or what a safety-review model
says, an action only executes if it passes these deterministic checks.

Crucially, this models **least privilege for the agent**: the agent holds no
cluster credentials and cannot call any infrastructure API — it only *proposes*
actions, and this gate (the control plane's job, not the model's) decides what
is admissible. The grant is **default-deny**: only the allowlisted actions
within quota are possible; everything else (deleting resources, reading secrets,
editing this policy, acting outside the namespace) is structurally impossible,
not merely discouraged. In a Kubernetes deployment this is where a controller's
ServiceAccount/Role, ResourceQuota, and admission control live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import PolicyDecision, RemediationAction


@dataclass
class Policy:
    # identity + scope the agent's proposals are confined to
    principal: str = "autoheal-agent"
    namespace: str = "team-ml"
    # action name -> the single config parameter it is allowed to change
    # (the agent's "verbs"; default-deny — anything not listed is impossible)
    allowed_actions: dict[str, str] = field(default_factory=lambda: {
        "set_learning_rate": "lr",
        "set_batch_size": "batch_size",
        "set_epochs": "epochs",
        "set_grad_clip": "grad_clip",
        "set_grad_accum": "grad_accum",
    })
    # parameter -> (inclusive min, inclusive max) — the resource quotas
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "lr": (1e-5, 1.0),
        "batch_size": (1, 256),
        "epochs": (1, 20),          # blast-radius ceiling on compute
        "grad_clip": (0.1, 10.0),
        "grad_accum": (1, 16),
    })
    # capabilities that are structurally denied (for display; default-deny makes
    # these impossible regardless — there is no action that could request them)
    denied_capabilities: list[str] = field(default_factory=lambda: [
        "delete resources", "read secrets", "modify its own policy/guardrails",
        "act outside its namespace", "escalate its own permissions",
    ])
    max_attempts: int = 6
    max_rejections: int = 4

    def grant(self) -> dict:
        """The agent's capability grant, for the trace (an RBAC-style summary)."""
        return {
            "principal": self.principal,
            "namespace": self.namespace,
            "allowed_actions": sorted(self.allowed_actions),
            "quotas": {p: list(b) for p, b in self.bounds.items()},
            "default_deny": True,
            "denied": self.denied_capabilities,
        }

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
