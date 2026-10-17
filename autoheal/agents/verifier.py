"""Independent verifier agent.

A thin role wrapper over the deterministic verification engine. The "agent" here
coordinates verification, but the actual decision is made by deterministic code
(engine/verification.py) that re-loads artifacts and re-checks metrics — never
by an LLM judging its own work.
"""
from __future__ import annotations

from ..engine.verification import verify
from ..schemas import AcceptanceSpec, ExecutionResult, VerificationResult


def independently_verify(done: AcceptanceSpec,
                         result: ExecutionResult) -> VerificationResult:
    return verify(done, result)
