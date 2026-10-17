"""Recursive supervisor.

Owns the original goal and drives the reason -> act -> observe -> verify ->
re-plan loop. It never declares success itself: the independent verifier decides
when the acceptance criteria is met. It enforces budgets (max executions, max
rejected plans), refuses to re-run a configuration it has already tried (no
infinite identical retries), and escalates to a human when the evidence says it
is stuck. Every reasoning step goes through Crusoe/Nemotron; every binding
gate is deterministic.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..engine.policy import Policy
from ..engine.state import Store
from ..engine.verification import verify
from ..schemas import AcceptanceSpec, ExecutionResult, TrainConfig
from .acceptance import AcceptanceCriteriaAgent
from .diagnostician import DiagnosticAgent
from .remediation_planner import RemediationPlanner
from .safety_reviewer import SafetyReviewer


def _render_history(rows: list[dict]) -> str:
    out = []
    for r in rows:
        line = f"attempt {r['n']} [{r['kind']}]"
        if r.get("proposal"):
            line += f" proposal={r['proposal']}"
        if r.get("policy"):
            line += f" policy={r['policy']}"
        if r.get("verification"):
            line += f" verification={r['verification']}"
        if r.get("evidence"):
            # keep it short
            line += " evidence=<captured>"
        out.append(line)
    return "\n".join(out)


class Supervisor:
    def __init__(self, llm, store: Store, executor, policy: Optional[Policy] = None,
                 narrate: Callable[[str, dict], None] | None = None,
                 dod: Optional[AcceptanceCriteriaAgent] = None,
                 workload_module: str = "autoheal.workloads.train"):
        self.llm = llm
        self.store = store
        self.executor = executor
        self.policy = policy or Policy()
        self.workload_module = workload_module
        self.dod = dod or AcceptanceCriteriaAgent(llm)
        self.diagnostician = DiagnosticAgent(llm)
        self.planner = RemediationPlanner(llm)
        self.safety = SafetyReviewer(llm)
        self.narrate = narrate or (lambda event, data: None)

    def run_goal(self, goal: str, start_config: TrainConfig) -> dict:
        run_id = self.store.create_run(goal)
        self.narrate("run_start", {"run_id": run_id, "goal": goal})
        self.narrate("grant", {"grant": self.policy.grant()})

        done = self.dod.define(goal)
        self.store.set_acceptance_criteria(run_id, done.model_dump_json())
        self.narrate("acceptance", {"done": done.model_dump()})

        config = start_config
        executed_sigs: set[str] = set()
        executions = 0
        rejections = 0

        while True:
            if executions >= self.policy.max_attempts:
                return self._escalate(run_id, "budget_exhausted",
                                      f"reached max_attempts={self.policy.max_attempts}")

            sig = config.model_dump_json()
            if sig in executed_sigs:
                return self._escalate(run_id, "no_progress",
                                      "planner produced an already-tried config")
            executed_sigs.add(sig)

            executions += 1
            n = self.store.next_attempt_number(run_id)
            attempt_id = f"{run_id}-a{n}"
            self.narrate("execute", {"attempt": executions,
                                     "config": config.model_dump()})
            result = self.executor.run(config, attempt_id, self.workload_module)
            self.narrate("evidence", {"attempt": executions,
                                      "succeeded": result.succeeded,
                                      "exit_code": result.exit_code,
                                      "metrics": result.metrics})

            verification = None
            if result.succeeded:
                verification = verify(done, result)
                self.narrate("verify", {"attempt": executions,
                                        "passed": verification.passed,
                                        "summary": verification.summary()})
                if verification.passed:
                    self.store.record_attempt(
                        run_id, n, "execute_verified", config=config,
                        evidence=result, verification=verification)
                    self.store.set_status(run_id, "complete")
                    self.narrate("goal_complete",
                                 {"run_id": run_id, "attempts": executions})
                    return {"status": "complete", "run_id": run_id,
                            "executions": executions, "rejections": rejections}

            self.store.record_attempt(
                run_id, n, "execute", config=config, evidence=result,
                verification=verification)

            # ---- diagnose + plan until we get an approved action ----------
            ver_summary = verification.summary() if verification else "n/a"
            approved = None
            diag = safety_rev = decision = None
            while approved is None:
                if rejections >= self.policy.max_rejections:
                    return self._escalate(run_id, "rejections_exhausted",
                                          f"max_rejections={self.policy.max_rejections}")
                history_text = _render_history(self.store.history(run_id))
                diag = self.diagnostician.diagnose(result, history_text, ver_summary)
                self.narrate("diagnosis", {"diagnosis": diag.model_dump()})
                proposal = self.planner.plan(diag, config, self.policy, history_text)
                self.narrate("proposal", {"proposal": proposal.model_dump()})
                safety_rev = self.safety.review(proposal, result)
                self.narrate("safety", {"safety": safety_rev.model_dump()})
                decision = self.policy.check(proposal)
                self.narrate("policy", {"decision": decision.model_dump()})

                if safety_rev.verdict == "escalate":
                    self.store.record_attempt(
                        run_id, self.store.next_attempt_number(run_id),
                        "safety_escalation", diagnosis=diag, proposal=proposal,
                        safety=safety_rev, policy=decision)
                    return self._escalate(run_id, "safety_escalation",
                                          "safety reviewer requested human review")

                if decision.allowed and safety_rev.verdict == "allow":
                    approved = proposal
                    break

                rejections += 1
                reason = ("safety_block" if safety_rev.verdict == "block"
                          else "policy_rejected")
                self.store.record_attempt(
                    run_id, self.store.next_attempt_number(run_id), reason,
                    diagnosis=diag, proposal=proposal, safety=safety_rev,
                    policy=decision)
                self.narrate("rejected", {"reason": reason,
                                          "violations": decision.violations})

            # ---- apply approved action ------------------------------------
            self.store.record_attempt(
                run_id, self.store.next_attempt_number(run_id), "approved_plan",
                diagnosis=diag, proposal=approved, safety=safety_rev, policy=decision)
            self.narrate("approved", {"proposal": approved.model_dump()})
            config = config.apply(approved.parameter, approved.new_value)

    def _escalate(self, run_id: str, reason: str, detail: str) -> dict:
        self.store.set_status(run_id, f"escalated:{reason}")
        self.narrate("escalate", {"run_id": run_id, "reason": reason,
                                  "detail": detail})
        return {"status": "escalated", "reason": reason, "detail": detail,
                "run_id": run_id}
