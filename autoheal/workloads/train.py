"""A small, real PyTorch training workload.

Deliberately simple and fast (CPU-only, trains in seconds) but genuinely real:
a synthetic multi-class classification problem learned by an MLP. The failures
AutoHeal recovers from are *emergent from the config*, not simulated:

  * lr too high  -> gradients explode -> loss becomes NaN -> process exits 1
  * too few epochs -> converges but validation accuracy stays below threshold

Run standalone:
    python -m autoheal.workloads.train --config cfg.json --out artifacts/run
It writes a checkpoint (model.pt) and metrics.json, and exits non-zero on
divergence so the executor sees an honest failure signal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    import numpy as np
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lr = float(cfg["lr"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    grad_clip = cfg.get("grad_clip")
    grad_accum = int(cfg.get("grad_accum", 1))
    hidden = int(cfg["hidden_size"])
    num_classes = int(cfg["num_classes"])
    n_features = int(cfg["n_features"])
    n_samples = int(cfg["n_samples"])
    class_sep = float(cfg.get("class_sep", 1.0))
    seed = int(cfg.get("seed", 0))

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- synthetic but learnable dataset ------------------------------------
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, n_features, generator=g) * class_sep
    y = torch.randint(0, num_classes, (n_samples,), generator=g)
    X = centers[y] + torch.randn(n_samples, n_features, generator=g)
    n_val = n_samples // 5
    X_tr, y_tr = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]

    model = nn.Sequential(
        nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[train] starting lr={lr} batch_size={batch_size} epochs={epochs} "
          f"grad_clip={grad_clip} grad_accum={grad_accum}", flush=True)

    n_tr = X_tr.shape[0]
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n_tr, generator=g)
        opt.zero_grad()
        for bi, start in enumerate(range(0, n_tr, batch_size)):
            idx = perm[start : start + batch_size]
            logits = model(X_tr[idx])
            loss = loss_fn(logits, y_tr[idx]) / grad_accum
            loss.backward()

            if (bi + 1) % grad_accum == 0:
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()
                opt.zero_grad()

            step += 1
            lv = loss.item() * grad_accum
            if math.isnan(lv) or math.isinf(lv):
                # Honest, real failure signal — not a simulated one.
                msg = (f"[train] FATAL: loss diverged to NaN/Inf at epoch={epoch} "
                       f"step={step} (lr={lr}). Gradients exploded.")
                print(msg, flush=True)
                metrics = {"status": "diverged", "epochs_run": epoch,
                           "diverged_at_step": step, "lr": lr}
                (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
                print(f"[train] METRICS={json.dumps(metrics)}", flush=True)
                return 1
        with torch.no_grad():
            tr_loss = loss_fn(model(X_tr), y_tr).item()
        print(f"[train] epoch={epoch} train_loss={tr_loss:.4f}", flush=True)

    # --- evaluation ---------------------------------------------------------
    with torch.no_grad():
        val_pred = model(X_val).argmax(dim=1)
        val_acc = (val_pred == y_val).float().mean().item()

    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "arch": {"n_features": n_features, "hidden": hidden,
                         "num_classes": num_classes}},
               out / "model.pt")
    metrics = {"status": "completed", "epochs_run": epochs,
               "val_accuracy": round(val_acc, 4),
               "final_train_loss": round(tr_loss, 4), "lr": lr}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] done val_accuracy={val_acc:.4f}", flush=True)
    print(f"[train] METRICS={json.dumps(metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
