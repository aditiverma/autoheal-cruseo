"""Definition-of-Done agent.

Turns a vague goal ("prepare a validated dataset" / "train a validated model")
into explicit, machine-checkable conditions. Critically, the agent is NOT
trusted to define success on its own: whatever it returns is reconciled against
a trusted minimum specification, so it can add stricter checks but can never
remove a required one, lower a threshold below the floor, or invent a check
against a metric the workload does not emit (which would make the goal
unreachable). This answers "who decides done?" for the Done-Means-Done track:
a trusted spec, not the agent's self-assessment.

The agent is stage-aware: the trusted minimum and the set of assertable metric
keys are supplied per stage (model training vs. data preparation), so the same
agent serves both workloads behind one API.
"""
from __future__ import annotations

from ..schemas import DoneCheck, DoneDefinition

SYSTEM = (
    "You are the Definition-of-Done agent in an autonomous ML pipeline healer. "
    "Given a goal, produce explicit, independently checkable completion criteria. "
    "Prefer objective checks a program can verify from artifacts and metrics."
)

# ---- training stage --------------------------------------------------------
TRAINING_MINIMUM = [
    DoneCheck(name="training process exits successfully", kind="process_success"),
    DoneCheck(name="model checkpoint is loadable", kind="checkpoint_loadable"),
    DoneCheck(name="validation accuracy meets floor", kind="metric_threshold",
              metric="val_accuracy", op=">=", threshold=0.90),
]
# Outcome metrics only; epochs_run is a tunable under budget, not a goal metric.
TRAINING_METRICS = {"val_accuracy", "final_train_loss"}

# ---- data-preparation stage ------------------------------------------------
DATAPREP_MINIMUM = [
    DoneCheck(name="data-prep process exits successfully", kind="process_success"),
    DoneCheck(name="processed dataset artifact exists", kind="artifact_exists",
              artifact="dataset.npz"),
    DoneCheck(name="processed dataset is loadable and non-empty",
              kind="dataset_loadable"),
]
DATAPREP_METRICS = {"rows_out", "n_partitions", "peak_mem_mb"}


class DefinitionOfDoneAgent:
    def __init__(self, llm, trusted_minimum=None, known_metrics=None,
                 stage: str = "model training"):
        self.llm = llm
        self.trusted_minimum = trusted_minimum or TRAINING_MINIMUM
        self.known_metrics = known_metrics or TRAINING_METRICS
        self.stage = stage

    def define(self, goal: str) -> DoneDefinition:
        required = ", ".join(c.name for c in self.trusted_minimum)
        user = (
            f"Goal: {goal}\n"
            f"This goal concerns the {self.stage} stage of an ML pipeline.\n\n"
            "Produce a DoneDefinition. It MUST include at least these checks: "
            f"{required}. You may add stricter checks.\n"
            "metric_threshold checks may ONLY reference these metric keys emitted "
            f"by the workload: {sorted(self.known_metrics)}. Do not invent keys."
        )
        proposed = self.llm.complete_json(
            system=SYSTEM, user=user, schema=DoneDefinition, role="reason"
        )
        return self._reconcile(goal, proposed)

    def _reconcile(self, goal: str, proposed: DoneDefinition) -> DoneDefinition:
        """Reconcile the agent's proposal against the trusted spec.

        The agent's ONLY latitude is to add stricter `metric_threshold` checks on
        known metrics. All structural checks (process success, which artifacts
        must exist, what must be loadable) are the workload's contract and come
        exclusively from the trusted minimum — the agent cannot invent them,
        weaken a floor, or (as a failure mode we actually hit) assert a check the
        workload can never satisfy and make the goal unreachable.
        """
        checks: list[DoneCheck] = [
            c for c in proposed.checks
            if c.kind == "metric_threshold" and c.metric in self.known_metrics
        ]

        def find_metric(metric: str | None):
            for c in checks:
                if c.kind == "metric_threshold" and c.metric == metric:
                    return c
            return None

        for req in self.trusted_minimum:
            if req.kind == "metric_threshold":
                existing = find_metric(req.metric)
                if existing is None:
                    checks.append(req)
                elif existing.threshold is None or existing.threshold < req.threshold:
                    existing.threshold = req.threshold  # never below the floor
                    existing.op = req.op or ">="
            else:
                checks.append(req)  # structural checks only from trusted spec
        return DoneDefinition(
            goal=goal, checks=checks,
            rationale=(proposed.rationale
                       + " [reconciled against trusted minimum spec]").strip(),
        )
