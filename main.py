"""AutoHeal CLI.

Commands:
    python main.py models                 # discover Crusoe model ids
    python main.py run --scenario full    # run the self-healing loop
    python main.py run --scenario full --offline   # dev: no Crusoe calls
    python main.py trace <run_id>         # print a saved recovery trace

Only the model inference runs on Crusoe Managed Inference; the supervisor,
executor, policy engine, and verifier run locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from autoheal.agents.definition_of_done import DefinitionOfDoneAgent
from autoheal.agents.supervisor import Supervisor
from autoheal.engine.policy import Policy
from autoheal.engine.state import Store
from autoheal.scenarios import SCENARIOS
from autoheal.workloads.local_executor import LocalExecutor

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "runs" / "autoheal.sqlite"
ARTIFACTS = ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# pretty trace output                                                          #
# --------------------------------------------------------------------------- #
def _p(msg: str) -> None:
    print(msg, flush=True)


def make_narrator():
    def narrate(event: str, data: dict) -> None:
        if event == "run_start":
            _p(f"\n\033[1m● AutoHeal run {data['run_id']}\033[0m")
            _p(f"  goal: {data['goal']}")
        elif event == "definition_of_done":
            _p("\n\033[1m◆ Definition of Done\033[0m (reconciled vs trusted minimum)")
            for c in data["done"]["checks"]:
                extra = ""
                if c["kind"] == "metric_threshold":
                    extra = f"  [{c['metric']} {c['op']} {c['threshold']}]"
                _p(f"    - {c['name']}{extra}")
        elif event == "execute":
            cfg = data["config"]
            _p(f"\n\033[1m▶ Attempt {data['attempt']}: execute workload\033[0m")
            shown = {k: cfg[k] for k in (
                "lr", "batch_size", "epochs", "grad_clip",
                "input_rows", "n_partitions", "executor_memory_mb") if k in cfg}
            _p("    config: " + " ".join(f"{k}={v}" for k, v in shown.items()))
        elif event == "evidence":
            status = "\033[32msucceeded\033[0m" if data["succeeded"] \
                else "\033[31mFAILED\033[0m"
            _p(f"    evidence: {status} (exit={data['exit_code']}) "
               f"metrics={data['metrics']}")
        elif event == "verify":
            tag = "\033[32mPASS\033[0m" if data["passed"] else "\033[31mFAIL\033[0m"
            _p(f"    \033[1m⊙ independent verification: {tag}\033[0m")
            _p(f"      {data['summary']}")
        elif event == "diagnosis":
            d = data["diagnosis"]
            _p(f"    \033[36m⚕ diagnosis:\033[0m {d['failure_class']} — "
               f"{d['root_cause']} (conf={d['confidence']})")
        elif event == "proposal":
            pr = data["proposal"]
            _p(f"    \033[33m✎ proposal:\033[0m {pr['action']} "
               f"{pr['parameter']}: {pr['old_value']} -> {pr['new_value']}")
        elif event == "safety":
            s = data["safety"]
            _p(f"    \033[35m⛨ safety:\033[0m {s['verdict']} "
               f"(risk={s['risk_level']}, injection={s['prompt_injection_detected']})")
        elif event == "policy":
            d = data["decision"]
            tag = "\033[32mALLOW\033[0m" if d["allowed"] else "\033[31mREJECT\033[0m"
            _p(f"    \033[1m⚖ policy: {tag}\033[0m "
               + ("" if d["allowed"] else f"— {d['violations']}"))
        elif event == "rejected":
            _p(f"    \033[31m↺ plan rejected ({data['reason']}); replanning\033[0m")
        elif event == "approved":
            _p("    \033[32m✓ plan approved; applying and retrying\033[0m")
        elif event == "goal_complete":
            _p(f"\n\033[1;32m✅ GOAL COMPLETE\033[0m after {data['attempts']} "
               f"execution(s). run_id={data['run_id']}")
        elif event == "escalate":
            _p(f"\n\033[1;31m🚨 ESCALATED TO HUMAN: {data['reason']}\033[0m")
            _p(f"    {data['detail']}")
    return narrate


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_models(_args) -> int:
    from autoheal.llm import CrusoeLLM
    llm = CrusoeLLM()
    _p(f"base_url={llm.base_url}")
    for mid in llm.list_models():
        _p(f"  {mid}")
    return 0


def _build_llm(offline: bool):
    if offline:
        from autoheal.llm import OfflineLLM
        return OfflineLLM()
    from autoheal.llm import CrusoeLLM
    return CrusoeLLM()


def cmd_run(args) -> int:
    scenario = SCENARIOS.get(args.scenario)
    if scenario is None:
        _p(f"unknown scenario '{args.scenario}'. choices: {list(SCENARIOS)}")
        return 2
    llm = _build_llm(args.offline)
    _p(f"\033[2m(reasoning via {llm.name}, model={getattr(llm, 'model', '?')})\033[0m")

    store = Store(DB_PATH)
    executor = LocalExecutor(ARTIFACTS, timeout_s=args.timeout)
    policy = scenario.get("policy") or Policy()
    policy.max_attempts = args.max_attempts
    dod = DefinitionOfDoneAgent(
        llm,
        trusted_minimum=scenario.get("trusted_minimum"),
        known_metrics=scenario.get("known_metrics"),
        stage=scenario.get("stage", "model training"),
    )
    workload = scenario.get("workload", "autoheal.workloads.train")
    sup = Supervisor(llm, store, executor, policy, narrate=make_narrator(),
                     dod=dod, workload_module=workload)

    result = sup.run_goal(scenario["goal"], scenario["config"])

    trace_path = ROOT / "runs" / f"{result['run_id']}.json"
    trace_path.write_text(json.dumps(store.export_trace(result["run_id"]), indent=2))
    _p(f"\n\033[2mtrace written to {trace_path}\033[0m")
    _p(f"\033[2mreplay with: python main.py trace {result['run_id']}\033[0m")
    return 0 if result["status"] == "complete" else 1


def cmd_trace(args) -> int:
    store = Store(DB_PATH)
    trace = store.export_trace(args.run_id)
    _p(json.dumps(trace, indent=2))
    return 0


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="autoheal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list Crusoe model ids")

    r = sub.add_parser("run", help="run the self-healing loop")
    r.add_argument("--scenario", default="full", choices=list(SCENARIOS))
    r.add_argument("--offline", action="store_true",
                   help="use deterministic stub LLM (dev only, not for demo)")
    r.add_argument("--max-attempts", type=int, default=6, dest="max_attempts")
    r.add_argument("--timeout", type=int, default=300)

    t = sub.add_parser("trace", help="print a saved run trace")
    t.add_argument("run_id")

    args = ap.parse_args()
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "trace":
        return cmd_trace(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
