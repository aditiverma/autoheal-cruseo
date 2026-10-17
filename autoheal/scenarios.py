"""Demo scenarios: starting (broken) configurations.

Each scenario is a goal plus a deliberately broken starting TrainConfig that
produces a *real* failure the agent team must recover from.
"""
from __future__ import annotations

from .agents.definition_of_done import (
    DATAPREP_METRICS,
    DATAPREP_MINIMUM,
    TRAINING_METRICS,
    TRAINING_MINIMUM,
)
from .engine.policy import Policy
from .schemas import DataPrepConfig, TrainConfig

GOAL = "Prepare training data and train a validated PyTorch classifier."
DATAPREP_GOAL = "Prepare a validated training dataset from the raw input."

TRAIN_WORKLOAD = "autoheal.workloads.train"
DATAPREP_WORKLOAD = "autoheal.workloads.dataprep"

SCENARIOS: dict[str, dict] = {
    # Primary: learning rate so high the loss diverges to NaN. Real failure.
    "lr_divergence": {
        "goal": GOAL,
        "config": TrainConfig(lr=5.0, epochs=20, seed=0),
        "note": "lr=5.0 -> gradients explode -> NaN loss -> exit 1.",
    },
    # Underfit: learning rate far too low with too few epochs. The workload
    # 'succeeds' (exit 0) yet the independent verifier fails it on val-accuracy
    # (~0.62), forcing a replan. This is the Done-Means-Done beat: success!=done.
    "underfit": {
        "goal": GOAL,
        "config": TrainConfig(lr=0.002, epochs=3, seed=0),
        "note": "exits 0 but val_accuracy ~0.62, below the Done floor of 0.90.",
    },
    # Full arc: diverges first (lr), then may underfit (few epochs) after the lr
    # fix, exercising diagnosis, policy, verification-driven replan, recovery.
    "full": {
        "goal": GOAL,
        "config": TrainConfig(lr=5.0, epochs=3, seed=0),
        "note": "NaN first; after lr fix, may underfit until epochs are raised.",
    },
    # Policy-rejection demo: severely underfit (epochs=2) under a tight compute
    # budget (epochs ceiling = 8). The natural first fix is a big epoch jump that
    # exceeds the budget and is REJECTED by the deterministic policy gate; the
    # agent then replans within the budget (and raises lr) to recover.
    "rejection": {
        "goal": GOAL,
        "config": TrainConfig(lr=0.002, epochs=2, seed=0),
        "policy": Policy(bounds={
            "lr": (1e-5, 1.0),
            "batch_size": (1, 256),
            "epochs": (1, 8),          # tight compute budget -> forces replan
            "grad_clip": (0.1, 10.0),
            "grad_accum": (1, 16),
        }),
        "note": "epoch ceiling of 8 rejects the agent's large epoch jump.",
    },
    # Data-preparation OOM (Spark-style): a 4-partition job whose per-executor
    # working set (~1.2GB) exceeds the configured 512MB executor memory limit.
    # The agent must raise executor memory (bounded by the 2GB node max -> can be
    # rejected) or repartition to shrink the per-partition working set. Mirrors
    # the KubeCon abstract's "adjusting resources after an OOM failure".
    "dataprep_oom": {
        "goal": DATAPREP_GOAL,
        "config": DataPrepConfig(input_rows=200_000, row_bytes=8192,
                                 n_partitions=4, executor_memory_mb=512, seed=0),
        "workload": DATAPREP_WORKLOAD,
        "stage": "data preparation",
        "trusted_minimum": DATAPREP_MINIMUM,
        "known_metrics": DATAPREP_METRICS,
        "policy": Policy(
            allowed_actions={
                "set_executor_memory": "executor_memory_mb",
                "set_partitions": "n_partitions",
            },
            bounds={
                "executor_memory_mb": (256, 2048),  # 2GB = node memory ceiling
                "n_partitions": (1, 256),
            },
        ),
        "note": "executor OOM; raise memory (<=2GB node cap) or repartition.",
    },
}
