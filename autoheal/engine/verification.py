"""Independent verifier.

Decides whether the goal is actually done using real artifacts and metrics, not
the agent's narrative or confidence. Deterministic by design: it re-loads the
checkpoint from disk, re-reads metrics the workload wrote, and evaluates each
acceptance-criteria check. The supervisor is not allowed to advance a stage
until this passes.
"""
from __future__ import annotations

import operator
from pathlib import Path

from ..schemas import (
    CheckResult,
    AcceptanceCheck,
    AcceptanceSpec,
    ExecutionResult,
    VerificationResult,
)

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le,
        "<": operator.lt, "==": operator.eq}


def _checkpoint_loadable(result: ExecutionResult) -> tuple[bool, str]:
    path = result.artifacts.get("model.pt")
    if not path or not Path(path).exists():
        return False, "model.pt missing"
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        arch = ckpt["arch"]
        model = torch.nn.Sequential(
            torch.nn.Linear(arch["n_features"], arch["hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(arch["hidden"], arch["num_classes"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        # exercise it on a dummy forward pass to prove it is usable
        model(torch.zeros(1, arch["n_features"]))
        return True, f"loaded {path}"
    except Exception as e:
        return False, f"load failed: {e}"


def _dataset_loadable(result: ExecutionResult) -> tuple[bool, str]:
    path = result.artifacts.get("dataset.npz")
    if not path or not Path(path).exists():
        return False, "dataset.npz missing"
    try:
        import numpy as np
        with np.load(path) as data:
            feats = data["features"]
            if feats.size == 0:
                return False, "dataset is empty"
            return True, f"loaded {feats.shape[0]} rows x {feats.shape[1]} features"
    except Exception as e:
        return False, f"load failed: {e}"


def verify(done: AcceptanceSpec, result: ExecutionResult) -> VerificationResult:
    checks: list[CheckResult] = []
    for c in done.checks:
        checks.append(_run_check(c, result))
    return VerificationResult(passed=all(c.passed for c in checks), checks=checks)


def _run_check(c: AcceptanceCheck, result: ExecutionResult) -> CheckResult:
    if c.kind == "process_success":
        return CheckResult(name=c.name, passed=result.succeeded,
                           detail=f"exit_code={result.exit_code}")
    if c.kind == "artifact_exists":
        want = c.artifact or "model.pt"
        ok = want in result.artifacts and Path(result.artifacts[want]).exists()
        return CheckResult(name=c.name, passed=ok,
                           detail=f"{want} in {list(result.artifacts.keys())}")
    if c.kind == "checkpoint_loadable":
        ok, detail = _checkpoint_loadable(result)
        return CheckResult(name=c.name, passed=ok, detail=detail)
    if c.kind == "dataset_loadable":
        ok, detail = _dataset_loadable(result)
        return CheckResult(name=c.name, passed=ok, detail=detail)
    if c.kind == "metric_threshold":
        val = result.metrics.get(c.metric)
        if val is None:
            return CheckResult(name=c.name, passed=False,
                               detail=f"metric '{c.metric}' absent")
        op = _OPS.get(c.op or ">=", operator.ge)
        ok = bool(op(val, c.threshold))
        return CheckResult(name=c.name, passed=ok,
                           detail=f"{c.metric}={val} {c.op} {c.threshold}")
    return CheckResult(name=c.name, passed=False, detail=f"unknown check {c.kind}")
