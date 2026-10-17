# AutoHeal — Demo Scripts

## 1-minute submission video (storyboard)

Record a live `rejection` run (richest single-scenario flow) and narrate over it.

| Time | On screen | Say |
|---|---|---|
| 0–8s | Title + one line | "AutoHeal: recursive agents that don't just diagnose ML failures — they repair them, then independently *prove* the goal is actually done." |
| 8–20s | `python main.py run --scenario rejection`; Attempt 1 fails verification | "The job exits zero — but our independent verifier, checking a trusted definition of done, fails it. Nemotron on Crusoe diagnoses the real cause from the logs." |
| 20–32s | the `POLICY REJECT: epochs=10 outside [1,8]` line | "The agent proposes a fix. A deterministic policy gate — not the model — rejects it for exceeding the compute budget, and tells it the limit. The agent replans within budget." |
| 32–47s | Attempts 2–4 converging | "It re-diagnoses a *different* root cause, adjusts again, and converges — reasoning, not a retry loop." |
| 47–55s | final `verify PASS → GOAL COMPLETE` | "The independent verifier confirms every condition is met. Done means done." |
| 55–60s | architecture diagram | "Controllers reconcile. Agents adapt. Policies enforce. Independent tests decide when we're done." |

Keep the terminal font large. One clean take of a real run beats any edit.

## 3-minute live judging demo

**0:00–0:25 — the goal.** "A self-healing ML pipeline that keeps going until the
*original goal* is verifiably met, with hard guardrails on what the agent may
do." Show the reconciled Definition of Done.

**0:25–1:00 — real failure + delegated diagnosis.** Run `lr_divergence` (fast,
2 executions). Show the NaN crash and the live Nemotron diagnosis from real logs.

**1:00–1:50 — safety + policy + replanning.** Run (or replay) `rejection`. Show
the policy gate rejecting an over-budget proposal and the agent adapting.

**1:50–2:30 — second workload, same loop.** Run `dataprep_oom`. "Same agent loop,
a different workload — a Spark-style data-prep stage. Executor OOM; the agent
repartitions and recovers." This is the one-API-for-data-and-AI point.

**2:30–3:00 — verification + architecture.** Show a `trace` JSON and the
controller/agent/policy/verifier split. Close on the one-liner.

### Q&A — have these ready
- *Recursive vs retry loop?* It re-plans on its own output — evidence, the
  policy's rejection, and the verifier's verdict on its own prior fix — with a
  nested inner re-planning loop and dynamic depth. Shown live in `rejection`.
- *What prevents dangerous actions?* A deterministic policy engine (allowlisted
  actions, parameter ranges, compute budget) — independent of the LLM. Plus an
  advisory Nemotron safety reviewer on a separate model.
- *Who decides "done"?* A trusted minimum spec. The agent may tighten thresholds
  but cannot invent structural checks or lower a floor.
- *Where is Crusoe?* Every reasoning step (DoD, diagnosis, planning, safety) is a
  Crusoe-hosted Nemotron call, in the decision loop — not a final summary.
- *What did you build vs. frameworks?* See README "What we built vs. what we used".
- *What if the verifier fails?* Evidence returns to the supervisor for another
  bounded planning iteration; budgets/no-progress trigger escalation.
