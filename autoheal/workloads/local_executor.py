"""Local workload executor.

Runs the PyTorch training script in a subprocess, captures logs + exit code,
and returns typed ExecutionResult evidence. A hard wall-clock timeout guards
against hangs. This is the only component that touches the workload; swapping
in a Kubernetes Job / Kubeflow TrainJob adapter later means reimplementing just
this interface, not the agents.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from ..schemas import ExecutionResult


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:]


class LocalExecutor:
    def __init__(self, workdir: str | Path, timeout_s: int = 300):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    def run(self, config, attempt_id: str,
            module: str = "autoheal.workloads.train") -> ExecutionResult:
        attempt_dir = self.workdir / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = attempt_dir / "config.json"
        cfg_path.write_text(config.model_dump_json(indent=2))

        cmd = [
            sys.executable, "-m", module,
            "--config", str(cfg_path), "--out", str(attempt_dir),
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout_s,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + f"\n[executor] TIMEOUT after {self.timeout_s}s"
            code = 124
        duration = time.time() - start

        metrics: dict = {}
        mpath = attempt_dir / "metrics.json"
        if mpath.exists():
            try:
                metrics = json.loads(mpath.read_text())
            except Exception:
                pass

        artifacts: dict[str, str] = {}
        for p in sorted(attempt_dir.iterdir()):
            if p.is_file():
                artifacts[p.name] = str(p)

        return ExecutionResult(
            succeeded=(code == 0),
            exit_code=code,
            duration_s=duration,
            stdout_tail=_tail(stdout if isinstance(stdout, str) else ""),
            stderr_tail=_tail(stderr if isinstance(stderr, str) else ""),
            metrics=metrics,
            artifacts=artifacts,
        )
