"""Offline decision replay. Complete scripts are fixtures, never historical model code.

Historical terminal measurements cannot establish the cost/outcome of an interrupted
trajectory. We stop at the first unobserved branch rather than debit an invented cost.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from .accounting import Ledger, ScopedLedger, StopBudget, aggregate, audit_events
from .planning import Plan, decide

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 'offline-policy-replay-v1'
DEFAULTS = dict(method='knapsack', reserve_fraction=.1, unit=1.0,
                max_physical_calls=144, wall_seconds=3500., max_tasks=100)


def missing_decision(policy, reason, missing):
    """No admissible first decision can be established from these inputs."""
    return {'schema_version': SCHEMA, 'mode': 'historical_measurements',
            'fidelity': 'approximation_incomplete', 'policy': policy,
            'decisions': [], 'history': [], 'stop_reason': reason, 'complete': False,
            'known_spend': 0., 'spent': None, 'remaining': None, 'points': None,
            'physical_calls': 0, 'pending_tasks': [], 'unobserved_branches': [{'reason': reason}],
            'missing_data': missing, 'unknown_quantities': ['decisions', 'total spend', 'remaining', 'outcomes'],
            'limitation': 'Historical measurements are insufficient even for the first policy decision.'}


@dataclass(frozen=True)
class Option:
    plan: Plan
    forecast: dict
    configs: tuple = ()


def policy_values(budget, **kwargs):
    policy = dict(DEFAULTS, budget=budget)
    unknown = set(kwargs) - set(policy)
    if unknown:
        raise ValueError(f'Unknown policy parameters: {sorted(unknown)}')
    policy.update(kwargs)
    for key in ('budget', 'wall_seconds'):
        if isinstance(policy[key], bool) or not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError(f'Positive finite {key} required')
    for key in ('max_physical_calls', 'max_tasks'):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f'Positive integer {key} required')
    # Validate selection even if input is empty.
    decide([], policy['budget'], policy['reserve_fraction'], policy['method'], policy['unit'])
    return policy


def option_from_dict(row):
    option = Option(Plan(**row['plan']), dict(row['forecast']), tuple(row.get('configs', [])))
    minimum = option.forecast.get('minimum_start_budget')
    if minimum is None or not math.isfinite(minimum) or minimum < 0:
        raise ValueError('A saved minimum_start_budget is required')
    if 'execution_budget' in option.forecast and option.forecast['execution_budget'] <= 0:
        raise ValueError('Positive execution_budget required')
    return option


class Clock:
    """Synchronous fixture clock: no sleeps, worker code or network."""
    def __init__(self):
        self.seconds = 1000000.

    def monotonic(self):
        return self.seconds

    def advance(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid fixture duration')
        self.seconds += seconds


def _event_plan(option, trajectory, parent, clock):
    """Interpret a complete, deterministic single-stage script, using runtime admission.

    The script declares each full request and the terminal/continue event following
    its response. It contains no executable tools. Config changes invalidate it.
    """
    known_before = aggregate(parent.calls)['known_budget_units']
    result = dict(task_id=option.plan.task_id, plan_id=option.plan.plan_id, sends=[], events=[])
    def incomplete(reason, missing):
        result.update(status='unobserved', stop_reason=reason, complete=False, missing_data=missing,
                      known_spend=aggregate(parent.calls)['known_budget_units']-known_before)
        return result
    if trajectory is None:
        return incomplete('unobserved_plan', ['measured trajectory for this task and plan'])
    if len(option.configs) != 1:
        return incomplete('unobserved_cascade', ['conditional stateful cascade requests and stage events'])
    config = option.configs[0]
    contract = {'configs': list(option.configs), **{k: option.forecast.get(k) for k in
                ('execution_budget', 'case_wall_seconds', 'max_protocol_retries')}}
    if trajectory.get('contract') != contract:
        return incomplete('changed_execution_configuration', ['trajectory under the requested configuration/caps'])
    if not trajectory.get('complete_script'):
        return incomplete('incomplete_script', ['complete request/event sequence'])
    ledger = parent
    if option.forecast.get('execution_budget') is not None:
        ledger = ScopedLedger(parent, option.forecast['execution_budget'], option.forecast['case_wall_seconds'])
    status, reason = 'call_limit', None
    run_calls = 0
    for request in trajectory.get('requests', []):
        clock.advance(request.get('before_seconds', 0))
        if clock.monotonic() >= ledger.deadline_monotonic:
            status = 'case_wall_timeout'; break
        if run_calls >= config['max_calls']:
            break
        if not all(key in request for key in ('body', 'raw_usage', 'duration_seconds', 'event')):
            return incomplete('incomplete_request', ['body, usage, duration and following event'])
        body = request['body']
        if body.get('model') != config['model'] or not 0 < body.get('max_tokens', 0) <= config['max_retry_output_tokens']:
            return incomplete('changed_request_configuration', ['request matching the stage model and output cap'])
        try:
            reserve = ledger.authorize(body)
        except StopBudget as exc:
            status, reason = 'budget_stop', str(exc); break
        run_calls += 1
        result['sends'].append({'index': run_calls, 'reserve': reserve, 'max_tokens': body['max_tokens'],
                                'model': body['model'], 'raw_usage': request['raw_usage']})
        clock.advance(request['duration_seconds'])
        parent.record(attempt_id=str(len(parent.calls)), run_id=option.plan.task_id,
                      logical_call_id=str(run_calls), raw_usage=request['raw_usage'], status=200)
        event = request['event']
        if not isinstance(event, dict) or 'status' not in event or type(event.get('continue')) is not bool:
            return incomplete('missing_execution_event', ['status and continue flag after response'])
        result['events'].append(event)
        if clock.monotonic() >= ledger.deadline_monotonic:
            status = 'case_wall_timeout'; break
        status = event['status']
        if not event['continue']:
            break
    else:
        # A complete script must explicitly establish how the plan terminates.
        if run_calls < config['max_calls']:
            return incomplete('missing_terminal_event', ['terminal event or next physical request'])
    actual = aggregate(parent.calls)
    result.update(status=status, stop_reason=reason or status, complete=actual['complete'],
                  known_spend=actual['known_budget_units']-known_before,
                  missing_data=[] if actual['complete'] else ['provider usage'])
    return result


def simulate(options, outcome, policy, *, clock=None, measure_planner=False):
    """Shared forecast decisions; reveal each outcome only after selecting it.

    outcome receives the selected option, physical ledger and clock. It must return
    incomplete when evidence cannot determine the next event. No further planning
    is possible with an unknown remainder.
    """
    if len({(c.plan.task_id, c.plan.plan_id) for c in options}) != len(options):
        raise ValueError('Duplicate task/plan')
    clock = clock or Clock()
    pending = {c.plan.task_id for c in options}
    decisions, history = [], []
    stop, complete = 'tasks_complete', True
    with tempfile.TemporaryDirectory(prefix='offline-replay-') as temp:
        ledger = Ledger(Path(temp)/'calls.jsonl', policy['budget'], policy['max_physical_calls'], clock=clock.monotonic)
        ledger.deadline_monotonic = clock.monotonic()+policy['wall_seconds']
        while pending and len(history) < policy['max_tasks']:
            if clock.monotonic() >= ledger.deadline_monotonic:
                stop = 'wall_limit'; break
            try:
                remaining = ledger.remaining
            except StopBudget:
                stop, complete = 'unknown_usage', False; break
            if len(ledger.calls) >= ledger.limit:
                stop = 'physical_call_limit'; break
            available = [c for c in options if c.plan.task_id in pending]
            planner_started = time.perf_counter()
            selected, chosen, selection_stop = decide(available, remaining, policy['reserve_fraction'],
                                                      policy['method'], policy['unit'])
            decisions.append({'remaining': remaining, 'seconds_remaining': ledger.deadline_monotonic-clock.monotonic(),
                              'candidates': [{'task_id':c.plan.task_id, **c.forecast,
                                  'admitted':c.forecast['minimum_start_budget'] <= min(remaining,c.forecast.get('execution_budget',remaining))}
                                  for c in sorted(available,key=lambda c:(c.plan.task_id,c.plan.plan_id))],
                              'selected': [asdict(p) for p in selected],
                              'chosen': asdict(chosen) if chosen else None,
                              'stop_reason': selection_stop})
            if measure_planner:
                decisions[-1]['planner_seconds'] = time.perf_counter()-planner_started
            if chosen is None:
                stop = selection_stop; break
            option = next(c for c in available if c.plan == chosen)
            pending.remove(chosen.task_id)
            result = outcome(option, ledger, clock)
            history.append(result)
            if not result['complete']:
                stop = 'unknown_usage' if not aggregate(ledger.calls)['complete'] else 'unobserved'
                complete = False; break
            if clock.monotonic() >= ledger.deadline_monotonic:
                stop = 'wall_limit'; break
        if pending and len(history) >= policy['max_tasks'] and stop == 'tasks_complete':
            stop = 'max_tasks'
        actual = aggregate(ledger.calls)
        return {'schema_version': SCHEMA, 'policy': policy, 'decisions': decisions, 'history': history,
                'stop_reason': stop, 'complete': complete,
                'known_spend': actual['known_budget_units'],
                'spent': actual['budget_units'] if complete else None,
                'remaining': ledger.remaining if complete else None,
                'physical_calls': actual['physical_calls'], 'points': None,
                'unknown_quantities': [] if complete else ['total spend', 'remaining budget', 'outcome of unobserved branch'],
                'unobserved_branches': [h for h in history if not h['complete']],
                'pending_tasks': sorted(pending)}


def replay_fixture(data, policy):
    if data.get('schema_version') != 'controlled-replay-fixture-v1':
        raise ValueError('Expected controlled-replay-fixture-v1')
    options = [option_from_dict(c) for c in data['candidates']]
    # Do not inspect bodies, usage or events until after the plan has been chosen.
    def outcome(option, ledger, clock):
        matches = [t for t in data['trajectories'] if (t['task_id'], t['plan_id']) ==
                   (option.plan.task_id, option.plan.plan_id)]
        if len(matches) > 1:
            raise ValueError('Ambiguous fixture trajectory')
        return _event_plan(option, matches[0] if matches else None, ledger, clock)
    result = simulate(options, outcome, policy)
    result.update(mode='controlled_fixture', fidelity='exact_for_declared_fixture_events' if result['complete'] else 'incomplete',
                  limitation='Synchronous scripted single-stage events only; no claim about real HTTP cancellation, tools or hidden correctness.')
    return result


def terminal_outcome(option, measurement):
    """Keep aggregate observations as evidence, never as a simulated prefix."""
    return {'task_id': option.plan.task_id, 'plan_id': option.plan.plan_id,
            'status': 'unobserved', 'stop_reason': 'insufficient_historical_trajectory', 'complete': False,
            'known_spend': 0., 'sends': [], 'confirmed_prefix': [],
            'observed_terminal_evidence': measurement,
            'missing_data': ['full ordered request bodies/reserves', 'monotonic request/tool events and durations',
                             'stateful stage transitions', 'outcome under the requested budget/call/time/output caps']}


def replay_history(trace, policy, read):
    """Saved forecasts are inputs; loading a current forecaster is not historical replay."""
    trace = Path(trace)
    manifest = read(trace/'manifest.json')
    decisions = read(trace/'decisions.jsonl', lines=True)
    predictions = read(trace/'plan_predictions.jsonl', lines=True)
    solutions = read(trace/'solutions.jsonl', lines=True)
    calls = read(trace/'calls.jsonl', lines=True)
    runs = read(trace/'runs.jsonl', lines=True)
    events = read(trace/'events.jsonl', lines=True)
    def unavailable(reason, missing):
        return missing_decision(policy, reason, missing) | {'saved_policy': manifest,
                'available_evidence': {'solutions': solutions, 'calls': calls}}
    if not decisions or not decisions[0].get('candidates'):
        return unavailable('missing_pre_execution_decisions', ['saved candidate forecasts and public task rewards'])
    reward = {p['task_id']: p['reward'] for d in decisions for p in d.get('selected', [])}
    options = []
    for c in decisions[0]['candidates']:
        if c['task_id'] not in reward:
            return unavailable('missing_public_reward', ['public task reward for '+c['task_id']])
        if any(k not in c for k in ('minimum_start_budget', 'budget_units', 'p_success', 'plan_id')):
            return unavailable('missing_admission_forecast', ['saved pre-execution cost, success probability and start reserve'])
        options.append(option_from_dict({'plan': {'task_id':c['task_id'], 'plan_id':c['plan_id'],
            'reward':reward[c['task_id']], 'predicted_cost':max(.001,c['budget_units']), 'p_success':c['p_success']},
            'forecast':c}))
    def outcome(option, ledger, clock):
        matches = [p for p in predictions if (p['task_id'], p['forecast']['plan_id']) ==
                   (option.plan.task_id, option.plan.plan_id)]
        evidence = {'matching_measurements': len(matches), 'terminal_records': []}
        for prediction in matches:
            terminal = [s for s in solutions if s['solution_id'] == prediction['solution_id']]
            for s in terminal:
                ids = set(s.get('attempt_run_ids', []))
                physical = [c for c in calls if c.get('run_id') in ids]
                evidence['terminal_records'].append({'solution': s, 'prediction': prediction,
                    'physical_evidence': physical, 'runs': [r for r in runs if r['run_id'] in ids],
                    'events': [e for e in events if e['run_id'] in ids]})
        return terminal_outcome(option, evidence)
    result = simulate(options, outcome, policy)
    result.update(mode='historical_measurements', fidelity='approximation_incomplete',
                  saved_policy=manifest,
                  historical_ledger_evidence={
                      'scope': 'Saved original run only; never charged to the counterfactual replay.',
                      'audit': audit_events(calls) if (trace/'calls.jsonl').is_file() else None},
                  limitation='Decision-only approximation from saved forecasts. Historical spend/verdict are evidence, not counterfactual spend/success. Independent singles do not measure a stateful cascade.',
                  missing_data=['full request bodies and per-send reserves', 'monotonic event timing',
                                'trajectories for unmeasured task/plan pairs', 'version-bound hidden verdicts'])
    return result


class Sources:
    def __init__(self):
        self.files = {}

    def read(self, path, *, lines=False):
        path = Path(path).resolve()
        if not path.is_file():
            self.files[str(path)] = {'sha256': None, 'exists': False}
            return [] if lines else {}
        raw = path.read_bytes()
        self.files[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'exists': True}
        return [json.loads(s) for s in raw.splitlines() if s.strip()] if lines else json.loads(raw)

    def provenance(self):
        # Explicit allowlist: no credentials, environment, model responses or arbitrary Git files.
        paths = [*ROOT.glob('harness/*.py'), *ROOT.glob('experiments/*.py'),
                 *ROOT.glob('tests/test*.py'), ROOT/'tests/fixtures/budget-replay.json',
                 ROOT/'requirements.lock', ROOT/'requirements-ml.txt', ROOT/'run_agent.py',
                 ROOT/'configs/learned-router-v2.json', ROOT/'docs/offline-replay.md', ROOT/'plan.md']
        for path in paths:
            if path.is_file():
                raw = path.read_bytes()
                self.files[str(path.resolve())] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'exists': True}
        def git(*args):
            result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True)
            return result.stdout.strip() if result.returncode == 0 else None
        from .deployed import runtime_hash
        bundle = json.loads((ROOT/'configs/learned-router-v2.json').read_text())
        return {'files': self.files, 'git_head': git('rev-parse', 'HEAD'),
                'git_status': git('status', '--short', '--untracked-files=all'),
                'runtime_hash': runtime_hash(), 'control_bundle_runtime_hash': bundle.get('runtime_hash'),
                'control_bundle_compatible': bundle.get('runtime_hash') == runtime_hash(),
                'note': 'File SHA256 identifies exact local bytes; HEAD/status is supplementary. Control bundle was not rewritten.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fixture', type=Path)
    source.add_argument('--trace', type=Path)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--policy', type=Path, help='JSON policy overrides (no solver execution)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    sources = Sources()
    try:
        overrides = sources.read(args.policy) if args.policy else {}
        if args.policy and not args.policy.is_file():
            raise ValueError('Missing policy file')
        if 'budget' in overrides:
            raise ValueError('Set budget with --budget')
        input_path = args.fixture or args.trace
        output = args.out.resolve()
        protected = [input_path.resolve(), *( [args.policy.resolve()] if args.policy else [])]
        if any(output == p or (p.is_dir() and output.is_relative_to(p)) for p in protected):
            raise ValueError('Output must be separate from inputs')
        if output.exists():
            raise ValueError('Output must be a new file')
        if args.fixture:
            data = sources.read(args.fixture)
            policy = policy_values(args.budget, **(data.get('policy', {}) | overrides))
            result = replay_fixture(data, policy)
        else:
            # Manifest policy is saved data; never instantiate the frozen control.
            manifest = sources.read(args.trace/'manifest.json')
            saved = {key: manifest[key] for key in DEFAULTS if key in manifest}
            saved['method'] = 'knapsack' if manifest.get('strategy') == 'router-knapsack' else 'success_greedy'
            policy = policy_values(args.budget, **(saved | overrides))
            result = replay_history(args.trace, policy, sources.read)
        result['provenance'] = sources.provenance()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as file:
            json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write('\n')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output': str(output), 'mode': result['mode'], 'fidelity': result['fidelity'],
                      'stop_reason': result['stop_reason'], 'known_spend': result['known_spend'],
                      'spent': result['spent'], 'remaining': result['remaining']}, ensure_ascii=False))
    return 0 if result['complete'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
