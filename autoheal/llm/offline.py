"""Offline LLM stand-in for development and CI.

NOT for the demo or submission. This produces deterministic, hard-coded
structured responses so the recursive loop, policy engine, executor, and
verifier can be exercised end-to-end without network access or a Crusoe key.
The judged demo must run against the real Crusoe/Nemotron client.
"""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from ..schemas import (
    Diagnosis,
    DoneCheck,
    DoneDefinition,
    RemediationAction,
    SafetyReview,
)

T = TypeVar("T", bound=BaseModel)


class OfflineLLM:
    name = "offline"
    model = "offline-stub"
    safety_model = "offline-stub"

    def __init__(self) -> None:
        self._plan_calls = 0

    def list_models(self) -> list[str]:
        return ["offline-stub"]

    def complete_json(
        self, *, system: str, user: str, schema: Type[T], role: str = "reason"
    ) -> T:
        if schema is DoneDefinition:
            return DoneDefinition(  # type: ignore[return-value]
                goal=user[:120],
                checks=[
                    DoneCheck(name="training process exits 0", kind="process_success"),
                    DoneCheck(name="checkpoint loadable", kind="checkpoint_loadable"),
                    DoneCheck(
                        name="val accuracy >= 0.90",
                        kind="metric_threshold",
                        metric="val_accuracy",
                        op=">=",
                        threshold=0.90,
                    ),
                ],
                rationale="offline canned definition of done",
            )
        if schema is Diagnosis:
            u = user.lower()
            if "nan" in u or "diverg" in u or "inf" in u:
                return Diagnosis(  # type: ignore[return-value]
                    failure_class="LR_DIVERGENCE",
                    root_cause="Learning rate too high; loss diverged to NaN.",
                    reasoning="offline canned diagnosis",
                    confidence=0.8,
                    evidence_cited=["loss became NaN"],
                )
            return Diagnosis(  # type: ignore[return-value]
                failure_class="UNDERFIT",
                root_cause="Converged but validation accuracy below threshold.",
                reasoning="offline canned diagnosis",
                confidence=0.6,
                evidence_cited=["val_accuracy below threshold"],
            )
        if schema is RemediationAction:
            self._plan_calls += 1
            if "LR_DIVERGENCE" in user or "NaN" in user:
                return RemediationAction(  # type: ignore[return-value]
                    action="set_learning_rate",
                    parameter="lr",
                    new_value=0.05,
                    reason="offline: lower lr to stop divergence",
                )
            # underfit path: first propose an out-of-policy jump, then a safe one
            if self._plan_calls <= 2:
                return RemediationAction(  # type: ignore[return-value]
                    action="set_epochs",
                    parameter="epochs",
                    new_value=99,
                    reason="offline: aggressive epoch increase (expect rejection)",
                )
            return RemediationAction(  # type: ignore[return-value]
                action="set_epochs",
                parameter="epochs",
                new_value=16,
                reason="offline: epoch increase within policy",
            )
        if schema is SafetyReview:
            return SafetyReview(  # type: ignore[return-value]
                verdict="allow", risk_level="low", reasoning="offline canned review"
            )
        raise TypeError(f"OfflineLLM has no canned response for {schema}")
