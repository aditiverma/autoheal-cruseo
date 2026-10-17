"""Remediation planner.

Given a diagnosis, the current config, the policy constraints, and the history
of what has already been tried (including rejected proposals), produces exactly
ONE structured RemediationAction. It is told the policy's allowed actions and
bounds so it can propose something admissible — but the deterministic policy
engine, not this agent, has the final say.
"""
from __future__ import annotations

from ..engine.policy import Policy
from ..schemas import Diagnosis, RemediationAction, TrainConfig

SYSTEM = (
    "You are the remediation planner in an autonomous ML pipeline healer. "
    "Propose exactly ONE concrete configuration change that most directly "
    "addresses the diagnosed root cause. The platform enforces resource limits "
    "and a compute budget that you may not know exactly; if a prior proposal was "
    "REJECTED, the rejection in the history states the enforced limit — comply "
    "with it. Do not repeat a change that was already tried and failed, and do "
    "not re-propose a value that policy already rejected. Return a single "
    "structured RemediationAction."
)


def _policy_text(policy: Policy) -> str:
    actions = "\n".join(f"  - {a} (modifies '{p}')"
                        for a, p in policy.allowed_actions.items())
    return (f"You may only use these action types:\n{actions}\n"
            "Choose values appropriate to the diagnosis. The deterministic "
            "policy engine will enforce the exact permitted ranges and reject "
            "anything outside them.")


class RemediationPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, diagnosis: Diagnosis, config: TrainConfig, policy: Policy,
             history_text: str) -> RemediationAction:
        user = (
            "=== DIAGNOSIS ===\n"
            f"failure_class={diagnosis.failure_class}\n"
            f"root_cause={diagnosis.root_cause}\n"
            f"reasoning={diagnosis.reasoning}\n\n"
            "=== CURRENT CONFIG ===\n"
            f"{config.model_dump_json(indent=2)}\n\n"
            "=== ALLOWED ACTIONS ===\n"
            f"{_policy_text(policy)}\n\n"
            "=== HISTORY (already tried / rejected, with enforced limits) ===\n"
            f"{history_text or 'none'}\n\n"
            "Propose one RemediationAction. Set old_value to the current value of "
            "the parameter you change."
        )
        return self.llm.complete_json(
            system=SYSTEM, user=user, schema=RemediationAction, role="reason"
        )
