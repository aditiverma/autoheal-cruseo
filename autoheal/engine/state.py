"""Persistent recovery state (SQLite).

Records the full recursive trace so judges can inspect exactly what happened and
so the supervisor can reason over prior attempts (shared memory / replanning).
One `runs` row per goal; one `attempts` row per loop iteration, capturing the
evidence, diagnosis, proposed action, safety review, policy decision, execution
outcome, and verification result.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                goal TEXT, status TEXT,
                done_definition TEXT, created_at REAL, updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT, n INTEGER, kind TEXT,
                config TEXT, evidence TEXT, diagnosis TEXT,
                proposal TEXT, safety TEXT, policy TEXT,
                verification TEXT, note TEXT, created_at REAL
            );
            """
        )
        self.conn.commit()

    # -- runs ---------------------------------------------------------------
    def create_run(self, goal: str) -> str:
        run_id = "run_" + uuid.uuid4().hex[:10]
        now = time.time()
        self.conn.execute(
            "INSERT INTO runs(run_id, goal, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (run_id, goal, "running", now, now),
        )
        self.conn.commit()
        return run_id

    def set_done_definition(self, run_id: str, done_json: str) -> None:
        self.conn.execute(
            "UPDATE runs SET done_definition=?, updated_at=? WHERE run_id=?",
            (done_json, time.time(), run_id),
        )
        self.conn.commit()

    def set_status(self, run_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
            (status, time.time(), run_id),
        )
        self.conn.commit()

    # -- attempts -----------------------------------------------------------
    def next_attempt_number(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM attempts WHERE run_id=?", (run_id,)
        ).fetchone()
        return int(row["c"]) + 1

    def record_attempt(self, run_id: str, n: int, kind: str, **fields: Any) -> str:
        attempt_id = f"{run_id}-a{n}"
        payload = {k: (self._dump(v)) for k, v in fields.items()}
        self.conn.execute(
            "INSERT OR REPLACE INTO attempts(attempt_id, run_id, n, kind, config, "
            "evidence, diagnosis, proposal, safety, policy, verification, note, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt_id, run_id, n, kind,
                payload.get("config"), payload.get("evidence"),
                payload.get("diagnosis"), payload.get("proposal"),
                payload.get("safety"), payload.get("policy"),
                payload.get("verification"), fields.get("note"),
                time.time(),
            ),
        )
        self.conn.commit()
        return attempt_id

    @staticmethod
    def _dump(v: Any) -> Optional[str]:
        if v is None:
            return None
        if hasattr(v, "model_dump_json"):
            return v.model_dump_json()
        if isinstance(v, str):
            return v
        return json.dumps(v, default=str)

    # -- read ---------------------------------------------------------------
    def history(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE run_id=? ORDER BY n", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def export_trace(self, run_id: str) -> dict:
        run = dict(self.conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
        attempts = []
        for r in self.history(run_id):
            a = dict(r)
            for k in ("config", "evidence", "diagnosis", "proposal",
                      "safety", "policy", "verification"):
                if a.get(k):
                    try:
                        a[k] = json.loads(a[k])
                    except Exception:
                        pass
            attempts.append(a)
        run["attempts"] = attempts
        if run.get("done_definition"):
            try:
                run["done_definition"] = json.loads(run["done_definition"])
            except Exception:
                pass
        return run
