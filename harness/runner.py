"""Execute selected plans against one shared physical ledger, then replan."""
import asyncio
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from .accounting import Ledger, ScopedLedger, StopBudget, aggregate, audit_events, initial_reserve
from .agent import run_one
from .core import append_json, atomic_json, now, public_task, solver_task, with_preview
from .planning import decide
from .routing import candidates, MODELS, PREVIEW_BYTES
from .statistics import Statistics
from .submission import test_hashes

SWITCH = {'invalid_response', 'submission_rejected', 'tool_error', 'tool_needs_continuation'}


async def run_plan(directory, candidate, ledger, artifacts, *, backend='real', allow_local_exec=False,
                   provider_factory=None, sufficiency_mode='off', sufficiency_phase='pre_api'):
    if provider_factory is None:
        from .provider import JiuwenProvider
        provider_factory = JiuwenProvider
    if candidate.forecast.get('execution_budget') is not None:
        ledger = ScopedLedger(ledger, candidate.forecast['execution_budget'], candidate.forecast['case_wall_seconds'])
    task = solver_task(directory, candidate.stages[0])
    solution_id = str(uuid4())
    started = time.monotonic()
    append_json(Path(artifacts)/'plan_predictions.jsonl', {
        'solution_id': solution_id, 'task_id': task['task_id'], 'task': task,
        'sufficiency_mode': sufficiency_mode, 'sufficiency_phase': sufficiency_phase,
        'forecast': candidate.forecast, 'plan_signature': candidate.signature,
        'configs': [asdict(c) for c in candidate.stages], 'created_at': now()})
    originals = test_hashes(Path(directory)/'workspace') if task['task_type']=='coding' else {}
    attempts = []
    switches = []
    feedback = None
    stop = 'plan_complete'
    cancellation = None
    for index, config in enumerate(candidate.stages):
        if index:
            previous = attempts[-1]
            if previous['status'] not in SWITCH:
                stop = previous['status']
                break
            if not previous['actual']['complete']:
                stop = 'unknown_usage'
                break
            feedback = {'previous_attempt_status': previous['status'],
                        'public_observation': previous['last_observation'],
                        'instruction': 'Continue from current workspace using this observed failure or incomplete tool result.'}
            # Both the expected price and the real request headroom must fit.
            from .core import predict
            try:
                next_task=solver_task(directory,config)
                if max(predict(next_task, config)['budget_units'],initial_reserve(next_task,config,feedback)) > ledger.remaining:
                    stop = 'insufficient_switch_budget'
                    break
            except StopBudget:
                stop = 'unknown_usage'
                break
            switches.append({'from': candidate.stages[index-1].model, 'to': config.model,
                             'trigger': previous['status']})
        provider = provider_factory(config, ledger, mock=backend=='mock')
        try:
            work = run_one(directory, config, provider, artifacts, allow_local_exec=allow_local_exec,
                           single_action=candidate.single_first and index==0,
                           original_tests=originals, feedback=feedback,
                           max_protocol_retries=candidate.forecast.get('max_protocol_retries', 0),
                           sufficiency_mode=sufficiency_mode, sufficiency_phase=sufficiency_phase)
            try:
                result = await work
            except asyncio.CancelledError as exc:
                cancellation = exc
                result = stopped_attempt(artifacts, provider.run_id)
        finally:
            await provider.close()
        attempts.append(result)
        stop = result['status']
        if cancellation is not None or result['answer'] is not None or not result['actual']['complete']:
            break
    run_ids = {r['run_id'] for r in attempts}
    result = {'solution_id': solution_id, 'task_id': task['task_id'], 'task_type': task['task_type'],
              'sufficiency_mode': sufficiency_mode, 'sufficiency_phase': sufficiency_phase,
              'plan_id': candidate.plan.plan_id, 'plan_signature': candidate.signature,
              'forecast': candidate.forecast, 'attempt_run_ids': [r['run_id'] for r in attempts],
              'actual': aggregate([c for c in ledger.calls if c['run_id'] in run_ids]),
              'status': attempts[-1]['status'] if attempts else 'not_started', 'stop_reason': stop,
              'answer_saved': any(r['answer'] is not None for r in attempts), 'success': None,
              'preexisting_answer_preserved': any(r.get('preexisting_answer_preserved', False) for r in attempts),
              'protocol_retries': sum(r.get('protocol_retries', 0) for r in attempts),
              'correctness': 'unknown', 'switches': switches,
              'duration_seconds': time.monotonic()-started, 'backend': backend}
    append_json(Path(artifacts)/'solutions.jsonl', result)
    if cancellation is not None:
        raise cancellation
    return result


def stopped_attempt(artifacts, run_id):
    # run_one finalizes only after its worker has killed the group/reaped the parent
    # and quarantined unaccepted answers. Use that durable result, not a guessed
    # timeout rollup that could disagree with files or lose known/unknown usage.
    runs = [json.loads(line) for line in (Path(artifacts)/'runs.jsonl').read_text().splitlines()]
    return next(r for r in reversed(runs) if r['run_id'] == run_id)


async def execute(args, config):
    root, artifacts = Path(args.tasks_dir).resolve(), Path(args.artifacts).resolve()
    out = Path(args.out).resolve() if args.out else None
    artifacts.mkdir(parents=True, exist_ok=True)
    if list(artifacts.glob('*.jsonl')):
        raise ValueError('Fresh artifacts required; interrupted usage must be reconciled')
    routing = json.loads(Path(args.routing_config).read_text()) if args.routing_config else {}
    models = routing.get('models', MODELS)
    reserve = routing.get('reserve_fraction', args.reserve_fraction)
    if not math_is_fraction(reserve):
        raise ValueError('Reserve fraction must be finite and in [0,1)')
    statistics = Statistics(json.loads(Path(args.statistics).read_text()) if args.statistics else None,
                            prior_success=routing.get('prior_success', .5))
    forecaster = None
    if getattr(args, 'forecaster', None):
        from .deployed import LearnedRouter
        forecaster = LearnedRouter(args.forecaster)
    pending = {t['task_id']: (d, t) for d in sorted(p.parent for p in root.glob('*/task.json'))
               if not (d/'answer.txt').exists() for t in [with_preview(d,public_task(d),PREVIEW_BYTES)]}
    if not pending:
        raise ValueError('No fresh tasks')
    all_dirs = list(root.glob('*/task.json'))
    if len({public_task(p.parent)['task_id'] for p in all_dirs}) != len(all_dirs):
        raise ValueError('Duplicate task_id')
    atomic_json(artifacts/'manifest.json', {'created_at': now(), 'strategy': args.strategy,
        'base_config': asdict(config), 'models': models, 'availability': 'allowed by local rules; proxy availability requires probe',
        'sufficiency_mode': getattr(args, 'sufficiency_mode', 'off'),
        'sufficiency_phase': getattr(args, 'sufficiency_phase', 'pre_api'),
        'reserve_fraction': reserve, 'statistics': args.statistics, 'backend': args.backend,
        'forecaster': str(args.forecaster) if forecaster else None,
        'forecaster_sha256': hashlib.sha256(Path(args.forecaster).read_bytes()).hexdigest() if forecaster else None,
        'runtime_binding': forecaster.bundle.get('runtime_binding') if forecaster else None,
        'selected_estimators': forecaster.bundle['selected'] if forecaster else None,
        'training_data_id': forecaster.bundle['data_id'] if forecaster else None,
        'budget': args.budget, 'max_physical_calls': args.max_physical_calls, 'wall_seconds': args.wall_seconds,
        'python': __import__('sys').version,
        'dependencies': {p: importlib.metadata.version(p) for p in ('openjiuwen','openai','httpx')}})
    ledger = Ledger(artifacts/'calls.jsonl', args.budget, args.max_physical_calls, config.reasoning_in_completion)
    ledger.deadline_monotonic = time.monotonic()+args.wall_seconds
    method = 'knapsack' if args.strategy=='router-knapsack' else 'success_greedy'
    results = []
    stop = 'tasks_complete'
    old = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='jiuwen-router-runtime-') as runtime:
        os.chdir(runtime)
        try:
            while pending and len(results)<args.max_tasks:
                seconds = ledger.deadline_monotonic-time.monotonic()
                if seconds<=0:
                    stop = 'wall_limit'; break
                try:
                    remaining = ledger.remaining
                except StopBudget:
                    stop = 'unknown_usage'; break
                if len(ledger.calls)>=ledger.limit:
                    stop = 'physical_call_limit'; break
                options = [c for _, t in pending.values() for c in candidates(t, config, models=models,
                           statistics=statistics, only=routing.get('only_plan'), forecaster=forecaster)]
                selected, chosen, selection_stop = decide(options, remaining, reserve, method)
                append_json(artifacts/'decisions.jsonl', {'created_at': now(), 'remaining': remaining,
                    'seconds_remaining': seconds, 'method': method,
                    'candidates': [{'task_id': c.plan.task_id, **c.forecast} for c in options],
                    'selected': [asdict(p) for p in selected]})
                if not selected:
                    stop = selection_stop; break
                # Execute one, then recompute from actual usage. Tie order is deterministic, not a quality claim.
                candidate = next(c for c in options if c.plan==chosen)
                directory, _ = pending.pop(chosen.task_id)
                # run_plan owns the ledger deadline and waits for tool cleanup.
                # A second timer here could interrupt its final accounting.
                result = await run_plan(directory, candidate, ledger, artifacts,
                    backend=args.backend, allow_local_exec=args.allow_local_exec,
                    sufficiency_mode=getattr(args, 'sufficiency_mode', 'off'),
                    sufficiency_phase=getattr(args, 'sufficiency_phase', 'pre_api'))
                results.append(result)
                if out:
                    atomic_json(out, results)
                if not result['actual']['complete']:
                    stop = 'unknown_usage'; break
                if time.monotonic() >= ledger.deadline_monotonic:
                    stop = 'wall_limit'; break
            if pending and len(results) >= args.max_tasks and stop == 'tasks_complete':
                stop = 'max_tasks'
        finally:
            os.chdir(old)
    events = [json.loads(s) for s in (artifacts/'calls.jsonl').read_text().splitlines()] if (artifacts/'calls.jsonl').exists() else []
    atomic_json(artifacts/'report.json', {'strategy': args.strategy, 'backend': args.backend,
        'solutions': results, 'stop_reason': stop, 'ledger_audit': audit_events(events),
        'task_count':len(all_dirs),'considered_tasks':len(all_dirs),
        'started_tasks':sum(r['actual']['physical_calls']>0 for r in results),
        'not_started_tasks':len(all_dirs)-sum(r['actual']['physical_calls']>0 for r in results),
        'coverage_complete':all(r['actual']['physical_calls']>0 for r in results) and len(results)==len(all_dirs),
        'warning': 'Saved answers are ungraded. This report contains no hidden correctness information.'})
    return 0


def math_is_fraction(value):
    import math
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<1
