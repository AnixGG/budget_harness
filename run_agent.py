#!/usr/bin/env python3
"""Contest entrypoint and offline predict/evaluate/calibrate commands."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from harness.core import atomic_json, calibrate, load_config, predict, public_task, now, solver_task


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(runs):
    # Never mix mock and real calls in a single accuracy estimate.
    result = {}
    for backend in sorted({r["backend"] for r in runs}):
        group = [r for r in runs if r["backend"] == backend]
        complete = [r for r in group if r["actual"]["complete"]]
        observed = [r for r in complete if r["actual"]["physical_calls"] > 0]
        errors = [r["prediction"]["budget_units"] - r["actual"]["budget_units"] for r in observed]
        result[backend] = {"runs": len(group), "complete_usage_runs": len(complete),
                           "unknown_usage_runs": len(group) - len(complete),
                           "answered_ungraded": sum(r["status"] == "answered_ungraded" for r in group),
                           "mean_absolute_budget_error": sum(abs(e) for e in errors) / len(errors) if errors else None,
                           "mean_signed_budget_error": sum(errors) / len(errors) if errors else None,
                           "warning": "Diagnostic metrics, not official score. Answers require a separate grader.",
                           "comparisons": [{"run_id": r["run_id"], "predicted": r["prediction"]["budget_units"],
                                            "actual": r["actual"]["budget_units"], "status": r["status"]} for r in group]}
    return result


async def execute(args, config):
    if args.strategy != "baseline":
        from harness.runner import execute as execute_routed
        return await execute_routed(args, config)
    import importlib.metadata
    from harness.accounting import Ledger, StopBudget, initial_reserve
    from harness.agent import run_one
    from harness.provider import JiuwenProvider
    root, artifacts = Path(args.tasks_dir).resolve(), Path(args.artifacts).resolve()
    if args.out:
        args.out = str(Path(args.out).resolve())
    artifacts.mkdir(parents=True, exist_ok=True)
    # No silent restart with reset accounting; predictions may precede an interrupted request.
    if any((artifacts / name).exists() for name in ("predictions.jsonl", "calls.jsonl", "runs.jsonl")):
        raise ValueError("Use a fresh --artifacts directory; resume requires provider reconciliation")
    directories = sorted(p.parent for p in root.glob("*/task.json"))
    tasks = [(directory, solver_task(directory, config)) for directory in directories]
    if not tasks:
        raise ValueError("No task directories found")
    if len({t["task_id"] for _, t in tasks}) != len(tasks):
        raise ValueError("Duplicate task_id")
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    ranked = sorted(tasks, key=lambda pair: (-pair[1]["reward"] / max(1, predict(pair[1], config, calibration)["budget_units"]), pair[1]["task_id"]))
    atomic_json(artifacts / "manifest.json", {
        "created_at": now(), "python": sys.version, "config_id": config.config_id,
        "sufficiency_mode": getattr(args, 'sufficiency_mode', 'off'),
        "sufficiency_phase": getattr(args, 'sufficiency_phase', 'pre_api'),
        "backend": args.backend, "budget": args.budget, "max_physical_calls": args.max_physical_calls,
        "max_tasks": args.max_tasks, "wall_seconds": args.wall_seconds,
        "model_endpoint_host": urlsplit(os.getenv("LLM_BASE_URL", "")).hostname if args.backend == "real" else "mock.invalid",
        "dependencies": {p: importlib.metadata.version(p) for p in ("openjiuwen", "openai", "httpx")},
        "task_order": [t["task_id"] for _, t in ranked],
    })
    ledger = Ledger(artifacts / "calls.jsonl", args.budget, args.max_physical_calls, config.reasoning_in_completion)
    ledger.deadline_monotonic = time.monotonic() + args.wall_seconds
    # SDK defaults to ./logs. Keep all incidental writes off the read-only /app.
    os.chdir(tempfile.mkdtemp(prefix="jiuwen-runtime-"))
    provider = JiuwenProvider(config, ledger, mock=args.backend == "mock")
    started = time.monotonic()
    results = []
    try:
        for directory, task in ranked:
            if len(results) >= args.max_tasks or time.monotonic() - started >= args.wall_seconds:
                break
            if (directory / "answer.txt").exists():
                continue
            try:
                if max(predict(task, config, calibration)["budget_units"],initial_reserve(task,config)) > ledger.remaining:
                    continue
            except StopBudget:
                break
            remaining_time = args.wall_seconds - (time.monotonic() - started)
            if remaining_time <= 0:
                break
            try:
                result = await asyncio.wait_for(run_one(directory, config, provider, artifacts, calibration,
                                                        args.allow_local_exec,
                                                        sufficiency_mode=getattr(args, 'sufficiency_mode', 'off'),
                                                        sufficiency_phase=getattr(args, 'sufficiency_phase', 'pre_api')), timeout=remaining_time)
            except asyncio.TimeoutError:
                break  # started events retain evidence; never claim a full cost label.
            results.append(result)
            print(json.dumps({"task_id": task["task_id"], "status": result["status"], "actual": result["actual"]}))
            if args.out:
                atomic_json(args.out, results)
            if not result["actual"]["complete"]:
                break
    finally:
        await provider.close()
    report = evaluate(results)
    if ledger.path.exists():
        from harness.accounting import audit_events
        report["ledger_audit"] = audit_events(records(ledger.path))
    atomic_json(artifacts / "report.json", report)
    return 1 if any(r["status"] == "error" for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "predict", "evaluate", "calibrate"], nargs="?", default="run")
    parser.add_argument("--tasks-dir", default=os.getenv("TASKS"))
    parser.add_argument("--task-dir")
    parser.add_argument("--config")
    parser.add_argument("--calibration")
    parser.add_argument("--strategy", choices=["baseline", "router-greedy", "router-knapsack"],
                        help="Default: learned router-knapsack; baseline remains available explicitly")
    parser.add_argument("--forecaster", help="Selected cost/success bundle; defaults to configs/learned-router-r1-submit.json (checker off; reviewed submission wrapper); frozen v2 remains a separate control")
    parser.add_argument("--routing-config")
    parser.add_argument("--statistics", help="Frozen real train-only routing summaries")
    parser.add_argument("--reserve-fraction", type=float, default=0.1)
    parser.add_argument("--artifacts", default="/tmp/jiuwen-runs")
    parser.add_argument("--backend", choices=["real", "mock"], default="real")
    parser.add_argument("--budget", type=float, default=float(os.getenv("BUDGET", "100000")))
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--max-physical-calls", type=int, default=400)
    parser.add_argument("--wall-seconds", type=float, default=3500)
    parser.add_argument("--allow-local-exec", action="store_true")
    parser.add_argument('--sufficiency-mode', choices=['off', 'observe', 'skip'], default='off',
                        help='P2 candidate: observation preserves policy; skip only for complete pre-API Missing')
    parser.add_argument('--sufficiency-phase', choices=['pre_api', 'post_tool'], default='pre_api',
                        help='post_tool is diagnostic only: first model/tool call already paid')
    parser.add_argument("--runs")
    parser.add_argument("--calls", help="Optional physical ledger to audit interrupted requests")
    parser.add_argument("--out", default=os.getenv("OUT"))
    args = parser.parse_args()
    if args.strategy is None:
        args.strategy = 'router-knapsack'
    if args.strategy != 'baseline' and not args.forecaster and not args.statistics and not args.routing_config:
        from harness.deployed import DEFAULT_BUNDLE
        args.forecaster = str(DEFAULT_BUNDLE)
    if args.strategy == 'baseline' and args.forecaster:
        parser.error('--forecaster requires a router strategy')
    config = load_config(args.config)
    if args.command == "predict":
        if not args.task_dir:
            parser.error("predict requires --task-dir")
        value = predict(solver_task(args.task_dir, config), config,
                        json.loads(Path(args.calibration).read_text()) if args.calibration else None)
    elif args.command in {"evaluate", "calibrate"}:
        if not args.runs:
            parser.error("--runs required")
        value = evaluate(records(args.runs)) if args.command == "evaluate" else calibrate(records(args.runs), config.config_id)
        if args.command == "evaluate" and args.calls:
            from harness.accounting import audit_events
            value["ledger_audit"] = audit_events(records(args.calls))
    else:
        if not args.tasks_dir:
            parser.error("--tasks-dir or TASKS required")
        if args.max_tasks <= 0 or args.max_physical_calls <= 0 or args.wall_seconds <= 0:
            parser.error("Limits must be positive")
        return asyncio.run(execute(args, config))
    if args.out:
        atomic_json(args.out, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    from harness.deployed import BundleCompatibilityError
    try:
        raise SystemExit(main())
    except BundleCompatibilityError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        # Do not print exception payloads that may contain credentials/HTTP headers.
        print(f"Stopped: {type(exc).__name__}. Check configuration and local artifacts.", file=sys.stderr)
        raise SystemExit(2)
