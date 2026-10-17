"""Shared data models for AutoHeal.

Everything that crosses an agent / engine boundary is a typed Pydantic model so
the recursive loop, the policy engine, and the persisted trace all agree on
shape. These are the structures the Nemotron agents are asked to produce.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Workload desired-state                                                       #
# --------------------------------------------------------------------------- #
class TrainConfig(BaseModel):
    """Tunable desired-state for the PyTorch workload.

    This is the object the agents are allowed to mutate (via a RemediationAction
    validated by the policy engine). Defaults describe a *healthy* run; the demo
    scenarios start from a deliberately broken variant.
    """

    lr: float = 0.05
    batch_size: int = 64
    epochs: int = 20
    grad_clip: Optional[float] = None  # max grad norm; None disables clipping
    grad_accum: int = 1
    hidden_size: int = 128
    num_classes: int = 6
    n_features: int = 20
    n_samples: int = 4000
    class_sep: float = 0.9  # smaller = harder problem (needs more training)
    dataloader_workers: int = 0
    seed: int = 0

    def apply(self, parameter: str, value: Any) -> "TrainConfig":
        """Return a copy with one parameter changed."""
        return self.model_copy(update={parameter: value})


class DataPrepConfig(BaseModel):
    """Tunable desired-state for the data-preparation (Spark-style) workload.

    Models a distributed data-prep stage running under a per-executor memory
    budget, the way the Kubeflow Spark Operator would schedule it. The agent may
    raise executor memory (bounded by the node maximum) or repartition to shrink
    the per-partition working set — the two authentic Spark OOM remedies.
    """

    input_rows: int = 200_000
    row_bytes: int = 8192          # declared width of a raw record
    n_partitions: int = 4
    executor_memory_mb: int = 512  # configured per-executor memory limit
    overhead_factor: float = 3.0   # shuffle/serialization working-set multiplier
    seed: int = 0

    def apply(self, parameter: str, value: Any) -> "DataPrepConfig":
        return self.model_copy(update={parameter: value})



# --------------------------------------------------------------------------- #
# Execution evidence                                                           #
# --------------------------------------------------------------------------- #
class ExecutionResult(BaseModel):
    """Raw, trustworthy evidence from actually running the workload."""

    succeeded: bool
    exit_code: int
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> path

    def evidence_text(self) -> str:
        """Compact, model-readable rendering of the evidence."""
        lines = [
            f"exit_code={self.exit_code} succeeded={self.succeeded} "
            f"duration_s={self.duration_s:.1f}",
            f"metrics={self.metrics}",
            f"artifacts={list(self.artifacts.keys())}",
            "--- stdout (tail) ---",
            self.stdout_tail.strip(),
            "--- stderr (tail) ---",
            self.stderr_tail.strip(),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent outputs                                                                #
# --------------------------------------------------------------------------- #
class DoneCheck(BaseModel):
    name: str
    kind: Literal["process_success", "artifact_exists", "checkpoint_loadable",
                  "dataset_loadable", "metric_threshold"]
    detail: str = ""
    artifact: Optional[str] = None  # for artifact_exists: expected filename
    # for metric_threshold:
    metric: Optional[str] = None
    op: Optional[Literal[">=", ">", "<=", "<", "=="]] = None
    threshold: Optional[float] = None


class DoneDefinition(BaseModel):
    goal: str
    checks: list[DoneCheck]
    rationale: str = ""


class Diagnosis(BaseModel):
    failure_class: str = Field(description="Short machine-ish label, e.g. LR_DIVERGENCE")
    root_cause: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_cited: list[str] = Field(default_factory=list)


class RemediationAction(BaseModel):
    action: str = Field(description="e.g. set_learning_rate, set_epochs")
    parameter: str
    old_value: Any = None
    new_value: Any
    reason: str


class SafetyReview(BaseModel):
    verdict: Literal["allow", "block", "escalate"]
    risk_level: Literal["low", "medium", "high"] = "low"
    prompt_injection_detected: bool = False
    concerns: list[str] = Field(default_factory=list)
    reasoning: str = ""


class PolicyDecision(BaseModel):
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    reason: str = ""


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(
            f"{'PASS' if c.passed else 'FAIL'} {c.name}"
            + (f" ({c.detail})" if c.detail else "")
            for c in self.checks
        )
