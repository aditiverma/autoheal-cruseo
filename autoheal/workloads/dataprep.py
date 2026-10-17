"""A small, real data-preparation workload with Spark-style executor memory limits.

Models a distributed data-prep stage (as the Kubeflow Spark Operator would run
it): the data is split into partitions, and each executor has a configured
memory limit. If a partition's working set exceeds that limit, the executor is
OutOfMemory-killed and the job fails — exactly the failure the KubeCon abstract
calls out ("adjusting resources after an OOM failure").

What is real: the resource-budget enforcement and the two remedies (raise
executor memory, or repartition to shrink the per-partition working set), plus
the actual (chunked, lightweight) processing and the output dataset artifact.
What is modeled, not reproduced: a kernel/cgroup OOM kill — we enforce the
configured executor memory budget deterministically rather than exhausting the
host, which is both safe and faithful to how Spark/k8s executors behave.

Run standalone:
    python -m autoheal.workloads.dataprep --config cfg.json --out artifacts/run
Exits 1 on executor OOM so the executor sees an honest failure signal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    input_rows = int(cfg["input_rows"])
    row_bytes = int(cfg["row_bytes"])
    n_partitions = int(cfg["n_partitions"])
    executor_memory_mb = int(cfg["executor_memory_mb"])
    overhead = float(cfg.get("overhead_factor", 3.0))
    seed = int(cfg.get("seed", 0))

    per_partition_rows = math.ceil(input_rows / max(n_partitions, 1))
    peak_mb = per_partition_rows * row_bytes * overhead / 1e6

    print(f"[dataprep] input_rows={input_rows} partitions={n_partitions} "
          f"per_partition_rows={per_partition_rows}", flush=True)
    print(f"[dataprep] estimated per-executor working set={peak_mb:.0f}MB "
          f"limit={executor_memory_mb}MB", flush=True)

    if peak_mb > executor_memory_mb:
        # Authentic executor-OOM failure signal.
        msg = (f"[dataprep] FATAL: executor OutOfMemory — a partition needs "
               f"~{peak_mb:.0f}MB but the executor memory limit is "
               f"{executor_memory_mb}MB. Increase executor memory or repartition "
               f"(more partitions) to shrink the per-partition working set.")
        print(msg, flush=True)
        metrics = {"status": "executor_oom", "peak_mem_mb": round(peak_mb),
                   "executor_memory_mb": executor_memory_mb,
                   "n_partitions": n_partitions, "rows_out": 0}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[dataprep] METRICS={json.dumps(metrics)}", flush=True)
        return 1

    # --- real (chunked, memory-safe) processing -----------------------------
    rng = np.random.default_rng(seed)
    out_dim = 16
    processed = 0
    reduced = np.empty((input_rows, out_dim), dtype="float32")
    chunk = 20_000
    for start in range(0, input_rows, chunk):
        m = min(chunk, input_rows - start)
        raw = rng.standard_normal((m, out_dim)).astype("float32")
        reduced[start:start + m] = raw / (1.0 + np.abs(raw))  # a real transform
        processed += m

    np.savez(out / "dataset.npz", features=reduced)
    metrics = {"status": "completed", "rows_out": int(processed),
               "n_partitions": n_partitions, "peak_mem_mb": round(peak_mb),
               "features": out_dim}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[dataprep] done rows_out={processed} "
          f"partitions={n_partitions}", flush=True)
    print(f"[dataprep] METRICS={json.dumps(metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
