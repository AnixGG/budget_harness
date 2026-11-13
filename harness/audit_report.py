"""Read-only reconciliation of saved evidence; never runs a solver or grader.

python -m harness.audit_report --trace TRACE --tasks-dir TASKS --grader GRADE --out NEW
Exit codes: 0 available checks passed, 1 contradictions, 2 CLI/input failure,
3 incomplete evidence (when no contradictions). See docs/offline-audit.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .accounting import audit_events, normalize
from .forecast_report import number
from .submission import read_answer, SubmissionError


TRACE_FILES = ('manifest.json', 'calls.jsonl', 'runs.jsonl', 'events.jsonl',
               'solutions.jsonl', 'predictions.jsonl', 'plan_predictions.jsonl', 'report.json')
TOTAL_FIELDS = ('complete', 'physical_calls', 'unknown_calls', 'known_budget_units',
                'input_tokens', 'output_tokens', 'total_tokens', 'budget_units')
STOPPED = {'case_wall_timeout', 'cancelled', 'wall_timeout', 'budget_stop'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def equal(a, b):
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) if number(a) and number(b) else a == b


class Audit:
    def __init__(self):
        self.sources = {}
        self.issues = []
        self.traces = {}
        self.attempts = []
        self.solutions = []
        self.answers = []

    def issue(self, severity, code, refs=(), **details):
        self.issues.append({'severity': severity, 'code': code, 'sources': list(refs), **details})

    def load(self, path, *, lines=False, binary=False, allow_list=False):
        path = Path(path).absolute()
        key = str(path)
        if key in self.sources:
            return self.sources[key]['value']
        item = {'path': key, 'sha256': None, 'exists': path.exists(), 'readable': False}
        self.sources[key] = item
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            item['value'] = None
            self.issue('incomplete', 'missing_source', [key])
            return None
        except OSError as exc:
            item['value'] = None
            self.issue('incomplete', 'unreadable_source', [key], error=str(exc))
            return None
        item.update(sha256=sha(data), size_bytes=len(data), readable=True)
        if binary:
            item['value'] = data
            return data
        try:
            if lines:
                value = []
                for n, line in enumerate(data.decode('utf-8').splitlines(), 1):
                    if line.strip():
                        row = json.loads(line, parse_constant=reject_constant)
                        if not isinstance(row, dict):
                            raise ValueError('Expected an object')
                        value.append((row, f'{key}#L{n}'))
            else:
                value = json.loads(data, parse_constant=reject_constant)
                if not isinstance(value, dict) and not (allow_list and isinstance(value, list)):
                    raise ValueError('Expected an object')
        except (ValueError, UnicodeError) as exc:
            self.issue('incomplete', 'invalid_source', [key], error=str(exc))
            value = None
        item['value'] = value
        return value

    def add_trace(self, trace):
        trace = Path(trace).resolve()
        if str(trace) in self.traces:
            return
        data = {name: self.load(trace / name, lines=name.endswith('.jsonl')) for name in TRACE_FILES}
        self.traces[str(trace)] = data

    def unique(self, rows, key, *, category):
        groups = defaultdict(list)
        for row, ref in rows or []:
            identity = row.get(key)
            if not isinstance(identity, str) or not identity:
                self.issue('incomplete', 'missing_identity', [ref], category=category, field=key)
                continue
            groups[identity].append((row, ref))
        result = {}
        for identity, copies in groups.items():
            if any(row != copies[0][0] for row, _ in copies[1:]):
                self.issue('contradiction', 'conflicting_duplicate', [ref for _, ref in copies],
                           category=category, id=identity)
                result[identity] = (None, [ref for _, ref in copies])
            else:
                result[identity] = (copies[0][0], [ref for _, ref in copies])
                if len(copies) > 1:
                    self.issue('info', 'identical_copy', result[identity][1], category=category, id=identity,
                               copies=len(copies))
        return result

    def ledger(self, rows, *, present=True):
        if not present:
            return {'state': 'missing_ledger', 'totals': None, 'sends': None, 'requests': []}
        physical = self.unique([(r, ref) for r, ref in rows if r.get('kind') == 'physical_request'],
                               'attempt_id', category='physical_request')
        starts = self.unique([(r, ref) for r, ref in rows if r.get('kind') == 'request_started'],
                             'attempt_id', category='request_started')
        bad_rows = [(r, ref) for r, ref in rows if r.get('kind') not in {'request_started', 'physical_request'}
                    or not isinstance(r.get('attempt_id'), str) or not r.get('attempt_id')
                    or not isinstance(r.get('run_id'), str) or not r.get('run_id')]
        if bad_rows:
            self.issue('incomplete', 'invalid_ledger_record', [ref for _, ref in bad_rows])
        events, requests = [], []
        component_in = component_out = 0
        for identity in sorted(set(physical) | set(starts)):
            terminal, refs = physical.get(identity, (None, []))
            start, start_refs = starts.get(identity, (None, []))
            conflicted = (identity in physical and terminal is None) or (identity in starts and start is None)
            if start and terminal and any(start.get(k) != terminal.get(k) for k in ('run_id', 'logical_call_id')):
                conflicted = True
                self.issue('contradiction', 'request_identity_mismatch', refs + start_refs, attempt_id=identity)
            usage = terminal.get('usage') if terminal else None
            valid = (isinstance(usage, dict) and type(usage.get('complete')) is bool
                     and all(usage.get(k) is None or number(usage[k]) for k in
                             ('input_tokens', 'output_tokens', 'total_tokens', 'budget_units')))
            if valid and usage['complete']:
                valid = (all(number(usage.get(k)) for k in ('input_tokens', 'output_tokens', 'total_tokens', 'budget_units'))
                         and equal(usage['total_tokens'], usage['input_tokens'] + usage['output_tokens'])
                         and equal(usage['budget_units'], usage['input_tokens'] / 10 + usage['output_tokens']))
            if terminal and not valid:
                self.issue('contradiction', 'invalid_normalized_usage', refs, attempt_id=identity)
                conflicted = True
            if terminal and 'raw_usage' in terminal and valid:
                expected = normalize(terminal['raw_usage'], terminal.get('reasoning_in_completion', True))
                if any(not equal(usage.get(k), expected[k]) for k in TOTAL_FIELDS if k in expected):
                    self.issue('contradiction', 'raw_usage_mismatch', refs, attempt_id=identity)
                    conflicted = True
            if conflicted or not terminal:
                usage = normalize(None)
            state = ('conflict' if conflicted else 'unfinished' if identity not in physical else
                     'known_zero' if usage['complete'] and usage['budget_units'] == 0 else
                     'known_usage' if usage['complete'] else 'unknown_usage')
            if state in {'unfinished', 'unknown_usage'}:
                self.issue('incomplete', state, refs + start_refs, attempt_id=identity)
            elif state == 'conflict':
                self.issue('incomplete', 'conflicting_usage_unavailable', refs + start_refs, attempt_id=identity)
            # Use the established accountant on one physical row per send. A conflicting
            # send gets unknown usage; no arbitrary duplicate is chosen as the truth.
            if identity in physical:
                events.append({'kind': 'physical_request', 'attempt_id': identity, 'usage': usage})
            if identity in starts:
                events.append({'kind': 'request_started', 'attempt_id': identity})
            component_in += usage.get('input_tokens') or 0
            component_out += usage.get('output_tokens') or 0
            requests.append({'attempt_id': identity, 'state': state, 'usage': usage,
                             'sources': refs + start_refs})
        result = audit_events(events)
        totals = result['totals']
        if bad_rows:
            totals.update(complete=False, **{k: None for k in
                          ('input_tokens', 'output_tokens', 'total_tokens', 'budget_units')})
        return {'state': 'unknown' if not totals['complete'] else 'no_sends' if not requests else
                        'known_zero' if totals['budget_units'] == 0 else 'known',
                'totals': totals, 'sends': len(requests), 'unfinished_attempts': result['unfinished_attempts'],
                'known_input_tokens': component_in, 'known_output_tokens': component_out,
                'known_component_budget_units': component_in / 10 + component_out, 'requests': requests}

    def rollup(self, stored, actual, refs, **identity):
        if not isinstance(stored, dict):
            self.issue('incomplete', 'missing_actual_rollup', refs, **identity)
            return
        if actual is None:
            self.issue('incomplete', 'unverifiable_actual_rollup', refs, **identity)
            return
        differences = {k: {'stored': stored[k], 'physical': actual[k]} for k in TOTAL_FIELDS
                       if k in stored and not equal(stored[k], actual[k])}
        if differences:
            self.issue('contradiction', 'actual_rollup_mismatch', refs, fields=differences, **identity)
        if any(k not in stored for k in ('complete', 'physical_calls', 'budget_units')):
            self.issue('incomplete', 'partial_actual_rollup', refs, **identity)

    def executions(self, trace, data):
        runs = self.unique(data['runs.jsonl'], 'run_id', category='run')
        predictions = self.unique(data['predictions.jsonl'], 'run_id', category='prediction')
        solutions = self.unique(data['solutions.jsonl'], 'solution_id', category='solution')
        plans = self.unique(data['plan_predictions.jsonl'], 'solution_id', category='plan_prediction')
        calls = data['calls.jsonl'] or []
        events = data['events.jsonl'] or []
        ids = set(runs) | set(predictions) | {r['run_id'] for r, _ in calls + events if isinstance(r.get('run_id'), str)}
        for run_id in sorted(ids):
            final, refs = runs.get(run_id, (None, []))
            pre, pre_refs = predictions.get(run_id, (None, []))
            selected = [(r, ref) for r, ref in calls if r.get('run_id') == run_id]
            actual = self.ledger(selected, present=data['calls.jsonl'] is not None)
            if final is None:
                self.issue('incomplete', 'missing_terminal_run', refs + pre_refs + [ref for _, ref in selected],
                           trace=trace, run_id=run_id)
            if final and pre and final.get('task_id') != pre.get('task_id'):
                self.issue('contradiction', 'task_identity_mismatch', refs + pre_refs, run_id=run_id)
            if final:
                self.rollup(final.get('actual'), actual['totals'], refs + [ref for _, ref in selected], run_id=run_id)
            self.attempts.append({'trace': trace, 'run_id': run_id, 'task_id': (final or pre or {}).get('task_id'),
                                  'status': (final or {}).get('status'), 'actual': actual, 'sources': refs + pre_refs,
                                  'record': final})
        owners = Counter(r for s, _ in solutions.values() if s and isinstance(s.get('attempt_run_ids'), list)
                         for r in set(x for x in s['attempt_run_ids'] if isinstance(x, str)))
        for sid in sorted(set(solutions) | set(plans)):
            final, refs = solutions.get(sid, (None, []))
            pre, pre_refs = plans.get(sid, (None, []))
            links = (final or {}).get('attempt_run_ids')
            valid = isinstance(links, list) and all(isinstance(r, str) for r in links)
            if not valid:
                self.issue('incomplete', 'missing_attempt_run_ids', refs + pre_refs, solution_id=sid)
                links = []
            if len(set(links)) != len(links) or any(owners[r] > 1 for r in links):
                valid = False
                self.issue('contradiction', 'ambiguous_attempt_ownership', refs, solution_id=sid, run_ids=links)
            if not final:
                valid = False
                self.issue('incomplete', 'missing_terminal_solution', refs + pre_refs, solution_id=sid)
            if final and pre:
                for key in ('task_id', 'plan_signature'):
                    if key in final and key in pre and final[key] != pre[key]:
                        valid = False
                        self.issue('contradiction', 'solution_identity_mismatch', refs + pre_refs, solution_id=sid, field=key)
            for rid in links:
                run, rrefs = runs.get(rid, (None, []))
                if not run:
                    self.issue('incomplete', 'missing_linked_run', refs + rrefs, solution_id=sid, run_id=rid)
                elif run.get('task_id') != final.get('task_id'):
                    valid = False
                    self.issue('contradiction', 'task_identity_mismatch', refs + rrefs, solution_id=sid, run_id=rid)
            selected = [(r, ref) for r, ref in calls if r.get('run_id') in links]
            actual = self.ledger(selected, present=data['calls.jsonl'] is not None and valid)
            if final:
                self.rollup(final.get('actual'), actual['totals'], refs + [ref for _, ref in selected], solution_id=sid)
                linked = [runs[r][0] for r in links if r in runs and runs[r][0]]
                if valid and len(linked) == len(links):
                    saved = any(r.get('answer') is not None or r.get('answer_saved') is True for r in linked)
                    if 'answer_saved' in final and final['answer_saved'] != saved:
                        self.issue('contradiction', 'answer_saved_mismatch', refs, solution_id=sid,
                                   stored=final['answer_saved'], linked=saved)
                    if linked and final.get('status') != linked[-1].get('status'):
                        self.issue('contradiction', 'solution_status_mismatch', refs, solution_id=sid)
            self.solutions.append({'trace': trace, 'solution_id': sid, 'task_id': (final or pre or {}).get('task_id'),
                                   'attempt_run_ids': links, 'identity_resolved': valid, 'actual': actual,
                                   'sources': refs + pre_refs, 'record': final})
        report = data['report.json']
        report_ref = str(Path(trace) / 'report.json')
        if report:
            totals = self.ledger(calls, present=data['calls.jsonl'] is not None)['totals']
            if any(k in report for k in ('budget_units', 'physical_calls', 'input_tokens', 'output_tokens')):
                self.rollup({k: report[k] for k in TOTAL_FIELDS if k in report}, totals, [report_ref], trace=trace)
            for key in ('actual', 'ledger_audit', 'audit'):
                if key in report:
                    self.rollup(report[key].get('totals') if key != 'actual' and isinstance(report[key], dict)
                                else report[key], totals, [report_ref + '#/' + key], trace=trace)
            for field, identity, terminal in (('solutions', 'solution_id', solutions), ('runs', 'run_id', runs)):
                saved_rows = report.get(field, [])
                if not isinstance(saved_rows, list):
                    self.issue('incomplete', 'invalid_report_records', [report_ref], field=field)
                    continue
                saved = self.unique([(r, f'{report_ref}#/{field}/{n}') for n, r in enumerate(saved_rows)
                                     if isinstance(r, dict)], identity, category='report_' + field)
                for ident in set(saved) | set(terminal):
                    row, refs = saved.get(ident, (None, []))
                    end, end_refs = terminal.get(ident, (None, []))
                    if field not in report:
                        continue
                    if row is None or end is None:
                        self.issue('contradiction', 'report_record_mismatch', refs + end_refs + [report_ref], id=ident)
                        continue
                    links = [ident] if field == 'runs' else end.get('attempt_run_ids')
                    if isinstance(links, list):
                        recalc = self.ledger([(r, ref) for r, ref in calls if r.get('run_id') in links],
                                             present=data['calls.jsonl'] is not None)['totals']
                        self.rollup(row.get('actual'), recalc, refs, **{identity: ident})
                    for key in ('task_id', 'attempt_run_ids', 'status', 'answer_saved', 'answer'):
                        if key in row and key in end and row[key] != end[key]:
                            self.issue('contradiction', 'report_record_mismatch', refs + end_refs, id=ident, field=key)

    def check_answers(self, tasks_dir):
        if tasks_dir is None:
            self.issue('incomplete', 'missing_tasks_dir')
            return
        root = Path(tasks_dir).resolve()
        metadata = {}
        for path in sorted(root.glob('*/task.json')):
            task = self.load(path)
            if task and isinstance(task.get('task_id'), str):
                metadata.setdefault(task['task_id'], []).append((path.parent, task))
        tids = set(metadata) | {a['task_id'] for a in self.attempts if a['task_id']}
        for tid in sorted(tids):
            candidates = metadata.get(tid, [])
            attempts = [a for a in self.attempts if a['task_id'] == tid]
            if len(candidates) != 1:
                self.issue('incomplete', 'ambiguous_task_directory', [str(root)], task_id=tid)
                continue
            directory, task = candidates[0]
            path = directory / 'answer.txt'
            data = self.load(path, binary=True) if path.exists() or path.is_symlink() else None
            if data is None and str(path) not in self.sources:
                self.sources[str(path)] = {'path': str(path), 'exists': False, 'readable': False,
                                           'sha256': None, 'value': None}
            text = None
            if data is not None:
                try:
                    text = read_answer(path).strip()
                    if not text:
                        raise SubmissionError('Empty answer')
                except (SubmissionError, UnicodeError, OSError) as exc:
                    self.issue('contradiction', 'invalid_answer_file', [str(path)], task_id=tid, error=str(exc))
            accepted, refs, stopped = [], [str(path)], []
            traces = {a['trace'] for a in attempts}
            for a in attempts:
                run = a['record'] or {}
                arefs = list(a['sources'])
                evs = [(r, ref) for r, ref in self.traces[a['trace']]['events.jsonl'] or [] if r.get('run_id') == a['run_id']]
                saves = [(r, ref) for r, ref in evs if r.get('kind') == 'answer_saved']
                if run.get('answer_saved') is True or run.get('answer') is not None or saves:
                    if data is None:
                        self.issue('contradiction', 'saved_answer_missing', arefs + [str(path)] + [ref for _, ref in saves], run_id=a['run_id'])
                if run.get('answer_saved') is False and (saves or run.get('answer') is not None):
                    self.issue('contradiction', 'answer_saved_mismatch', arefs + [ref for _, ref in saves], run_id=a['run_id'])
                if run.get('status') == 'answered_ungraded' and run.get('answer') is None and not saves:
                    self.issue('contradiction', 'answered_status_without_submission', arefs, run_id=a['run_id'])
                values = [r['answer'] for r, _ in saves if isinstance(r.get('answer'), str)]
                if isinstance(run.get('answer'), str):
                    values.append(run['answer'])
                if len(set(values)) > 1:
                    self.issue('contradiction', 'accepted_answer_mismatch', arefs + [ref for _, ref in saves], run_id=a['run_id'])
                if values:
                    accepted.append((a, values[-1]))
                if run.get('status') in STOPPED:
                    if values:
                        self.issue('contradiction', 'stopped_status_with_submission', arefs + [ref for _, ref in saves], run_id=a['run_id'])
                    preserved = run.get('preexisting_answer_preserved') is True or any(
                        r.get('kind') == 'answer_quarantined' and r.get('source') == 'task_file'
                        and r.get('restored_existing') is True for r, _ in evs)
                    if preserved:
                        self.issue('info', 'preexisting_answer_preserved', arefs, run_id=a['run_id'])
                    else:
                        stopped.append(a['run_id'])
                for event, ref in evs:
                    if event.get('kind') in {'answer_quarantined', 'attempt_stopped', 'submission_rejected'}:
                        refs.append(ref)
                        diagnostic = event.get('diagnostic_path')
                        if diagnostic:
                            # Historical container paths may no longer exist. Do not guess remapping.
                            self.load(Path(diagnostic) if Path(diagnostic).is_absolute() else Path(a['trace']) / diagnostic, binary=True)
                refs.extend(arefs + [ref for _, ref in saves])
            # task_id locates a directory, never chooses between competing executions.
            owners = [s for s in self.solutions if s['task_id'] == tid]
            single_owner = len(traces) <= 1 and (len(owners) <= 1 and all(s['identity_resolved'] for s in owners))
            if owners and any(a['run_id'] not in owners[0]['attempt_run_ids'] for a in attempts):
                single_owner = False
            if len(attempts) > 1 and not owners:
                single_owner = False
            if not single_owner:
                self.issue('incomplete', 'ambiguous_answer_identity', refs, task_id=tid)
            elif text is not None and accepted:
                order = owners[0]['attempt_run_ids'] if owners else [a['run_id'] for a, _ in accepted]
                accepted.sort(key=lambda item: order.index(item[0]['run_id']) if item[0]['run_id'] in order else -1)
                last = accepted[-1][1]
                if text != last:
                    self.issue('contradiction', 'answer_content_mismatch', refs, task_id=tid,
                               accepted=last, current=text)
            if data is not None and not accepted:
                self.issue('incomplete', 'unconfirmed_canonical_answer', refs, task_id=tid)
                if stopped:
                    self.issue('contradiction', 'stopped_attempt_unconfirmed_answer', refs, task_id=tid, run_ids=stopped)
            for s in owners:
                if (s['record'] or {}).get('answer_saved') is True and data is None:
                    self.issue('contradiction', 'saved_answer_missing', s['sources'] + [str(path)], solution_id=s['solution_id'])
            self.answers.append({'task_id': tid, 'task_type': task.get('task_type'), 'reward': task.get('reward'),
                                 'path': str(path), 'exists': data is not None, 'sha256': sha(data) if data is not None else None,
                                 'text': text, 'identity_resolved': single_owner, 'accepted_run_ids': [a['run_id'] for a, _ in accepted],
                                 'accepted_matches_current': bool(accepted and single_owner and accepted[-1][1] == text),
                                 'run_ids': [a['run_id'] for a in attempts], 'sources': refs,
                                 'correctness': 'unknown', 'grader_result': 'absent'})


def build_report(traces=(), *, tasks_dir=None, graders=(), observations=()):
    audit = Audit()
    for trace in traces:
        audit.add_trace(trace)
    observation_rows = []
    for path in observations:
        rows = audit.load(Path(path).resolve(), lines=True) or []
        observation_rows.extend(rows)
        for row, ref in rows:
            raw = row.get('raw_calls_path')
            if raw:
                origin = Path(raw)
                if not origin.is_absolute():
                    origin = Path(path).resolve().parent / origin
                audit.add_trace(origin.resolve().parent)
            else:
                audit.issue('incomplete', 'missing_observation_origin', [ref], run_key=row.get('run_key'))
    for trace, data in audit.traces.items():
        audit.executions(trace, data)
    all_calls = [r for data in audit.traces.values() for r in data['calls.jsonl'] or []]
    expense = audit.ledger(all_calls, present=bool(audit.traces) and all(d['calls.jsonl'] is not None for d in audit.traces.values()))
    # Preserve known evidence when a different source ledger is unavailable.
    if expense['totals'] is None and all_calls:
        known = audit.ledger(all_calls)
        expense['known_available_evidence'] = known
    audit.check_answers(tasks_dir)
    measurements = check_observations(audit, observation_rows)
    grades = check_graders(audit, graders)
    # Repeated rollup levels can encounter the same issue; retain one finding.
    unique_issues = {json.dumps(i, sort_keys=True): i for i in audit.issues}
    issues = list(unique_issues.values())
    counts = Counter(i['severity'] for i in issues)
    code = 1 if counts['contradiction'] else 3 if counts['incomplete'] else 0
    return {'schema_version': 'offline-consistency-audit-v1', 'exit_code': code,
            'status': 'contradictions_detected' if code == 1 else 'incomplete' if code == 3 else 'available_checks_passed',
            'all_checks_available': not counts['incomplete'],
            'sources': [{k: v for k, v in s.items() if k != 'value'} for s in audit.sources.values()],
            'expense': expense, 'attempts': audit.attempts, 'solutions': audit.solutions,
            'answers': audit.answers, 'measurements': measurements, 'graders': grades,
            'issues': issues, 'issue_counts': dict(counts),
            'unavailable_checks': [i for i in issues if i['severity'] == 'incomplete'],
            'limitations': ['Saved local grader results are not official hidden evaluation.',
                            'Current files alone cannot establish their historical contents.',
                            'Physical sends alone contribute to expense; rollups and observations are never added.']}


def check_observations(audit, rows):
    groups = defaultdict(list)
    unresolved = []
    for row, ref in rows:
        raw = row.get('raw_calls_path')
        if not raw:
            unresolved.append(ref)
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = Path(ref.split('#L')[0]).parent / path
        trace = str(path.resolve().parent)
        # Benchmark observations omit run_id; their origin has one solution.
        candidates = [s for s in audit.solutions if s['trace'] == trace and s['record']]
        if row.get('run_id'):
            candidates = [a for a in audit.attempts if a['trace'] == trace and a['run_id'] == row['run_id']]
        elif row.get('solution_id'):
            candidates = [s for s in candidates if s['solution_id'] == row['solution_id']]
        if len(candidates) != 1 or candidates[0]['task_id'] != row.get('task_id'):
            audit.issue('incomplete', 'ambiguous_observation_identity', [ref, str(path)], run_key=row.get('run_key'))
            unresolved.append(ref)
            continue
        candidate = candidates[0]
        identity = candidate.get('solution_id') or candidate.get('run_id')
        groups[(trace, identity)].append((row, ref, candidate))
    results = []
    for (trace, identity), copies in groups.items():
        first, ref, candidate = copies[0]
        # Continuation changes wrapper metadata and run_key, but not measured evidence.
        fields = ('task_id', 'config_id', 'solver_config_id', 'repeat_id', 'backend', 'actual',
                  'cost_label', 'usage_complete', 'success', 'grader', 'status', 'stop_reason')
        if any(any(r.get(k) != first.get(k) for k in fields) for r, _, _ in copies[1:]):
            audit.issue('contradiction', 'conflicting_observation', [ref for _, ref, _ in copies], id=identity)
        elif len(copies) > 1:
            audit.issue('info', 'continuation_observation_copy', [ref for _, ref, _ in copies], id=identity, copies=len(copies))
        for row, ref, _ in copies:
            if 'actual' in row:
                audit.rollup(row['actual'], candidate['actual']['totals'], [ref] + candidate['sources'], id=identity)
            total = candidate['actual']['totals']
            if 'cost_label' in row and total is not None and not equal(row['cost_label'], total['budget_units']):
                audit.issue('contradiction', 'observation_cost_mismatch', [ref] + candidate['sources'], id=identity)
        results.append({'trace': trace, 'id': identity, 'copies': len(copies),
                        'sources': [ref for _, ref, _ in copies]})
    return {'input_rows': len(rows), 'unique_measurements': len(results),
            'unresolved_rows': unresolved, 'records': results}


def saved_verdict(row):
    value = row.get('correctness')
    if value in {'verified_correct', 'verified_incorrect', 'unknown'}:
        return value
    flag = row.get('success', row.get('example_correct'))
    return 'verified_correct' if flag is True else 'verified_incorrect' if flag is False else 'unknown'


def check_graders(audit, paths):
    results = []
    if not paths:
        audit.issue('incomplete', 'missing_grader_results')
    by_answer = defaultdict(list)
    for path in paths:
        path = Path(path).resolve()
        data = audit.load(path, allow_list=True)
        if not data:
            continue
        root_list = isinstance(data, list)
        if root_list:
            data = {'tasks': data}
        field = 'per_task' if isinstance(data.get('per_task'), list) else 'tasks' if isinstance(data.get('tasks'), list) else None
        rows = data[field] if field else [data] if 'task_id' in data else []
        if not rows or any(not isinstance(r, dict) for r in rows):
            audit.issue('incomplete', 'unsupported_grader_format', [str(path)])
            continue
        verdicts = Counter(saved_verdict(r) for r in rows)
        point_values = [r.get('points', r.get('reward')) for r in rows]
        points = sum(point_values) if all(number(v) for v in point_values) else None
        refs = [str(path)]
        expected = {'points': points, 'solved': verdicts['verified_correct'],
                    'verified_correct': verdicts['verified_correct'], 'verified_incorrect': verdicts['verified_incorrect'],
                    'unknown': verdicts['unknown'], 'tasks': len(rows), 'task_count': len(rows)}
        rewards = []
        for row in rows:
            matches = [a for a in audit.answers if a['task_id'] == row.get('task_id')]
            rewards.append(matches[0]['reward'] if len(matches) == 1 else None)
        if all(number(v) for v in rewards):
            expected['available_task_reward'] = sum(rewards)
            if all(type(r.get('known_solvable')) is bool for r in rows):
                expected['max_solvable_points'] = sum(v for v, r in zip(rewards, rows) if r['known_solvable'])
        else:
            for key in ('available_task_reward', 'max_solvable_points'):
                if key in data:
                    audit.issue('incomplete', 'unverifiable_grader_summary', refs, field=key)
        for key, predicate in (('known_solvable', lambda r: r.get('known_solvable') is True),
                               ('underdetermined', lambda r: r.get('known_solvable') is False),
                               ('started_tasks', lambda r: r.get('started') is True)):
            row_field = 'started' if key == 'started_tasks' else 'known_solvable'
            if all(type(r.get(row_field)) is bool for r in rows):
                expected[key] = sum(predicate(r) for r in rows)
            elif key in data:
                audit.issue('incomplete', 'unverifiable_grader_summary', refs, field=key)
        for key, val in expected.items():
            if key in data and not isinstance(data[key], list):
                if val is None:
                    audit.issue('incomplete', 'unverifiable_grader_summary', refs, field=key)
                elif not equal(data[key], val):
                    audit.issue('contradiction', 'grader_summary_mismatch', refs, field=key, stored=data[key], recomputed=val)
        for key in ('actual', 'audit', 'ledger_audit'):
            # An external task-only summary does not identify a ledger scope.
            if key in data:
                scope = str(Path(data['trace']).resolve()) if isinstance(data.get('trace'), str) else None
                if scope in audit.traces:
                    actual = audit.ledger(audit.traces[scope]['calls.jsonl'] or [],
                                          present=audit.traces[scope]['calls.jsonl'] is not None)['totals']
                    stored = data[key] if key == 'actual' else data[key].get('totals')
                    audit.rollup(stored, actual, refs, trace=scope)
                else:
                    audit.issue('incomplete', 'unbound_grader_expense', refs)
        records = []
        for n, row in enumerate(rows):
            ref = f'{path}#/{n}' if root_list else f'{path}#/{field}/{n}' if field else str(path)
            tid = row.get('task_id')
            answer_candidates = [a for a in audit.answers if a['task_id'] == tid]
            answer = answer_candidates[0] if len(answer_candidates) == 1 else None
            verdict = saved_verdict(row)
            if verdict == 'unknown':
                audit.issue('incomplete', 'grader_verdict_unknown', [ref], task_id=tid)
            if 'correctness' in row and ('success' in row or 'example_correct' in row):
                flag = row.get('success', row.get('example_correct'))
                expected_verdict = 'verified_correct' if flag is True else 'verified_incorrect' if flag is False else 'unknown'
                if verdict != expected_verdict:
                    audit.issue('contradiction', 'grader_verdict_mismatch', [ref], task_id=tid)
            awarded = row.get('points', row.get('reward'))
            if answer and number(answer['reward']) and number(awarded) and verdict != 'unknown':
                expected_points = answer['reward'] if verdict == 'verified_correct' else 0
                if not equal(awarded, expected_points):
                    audit.issue('contradiction', 'grader_reward_mismatch', [ref, answer['path']], task_id=tid,
                                stored=awarded, expected=expected_points)
            candidates = []
            for item in audit.attempts if row.get('run_id') else audit.solutions if row.get('solution_id') else []:
                key = 'run_id' if row.get('run_id') else 'solution_id'
                if item[key] == row[key] and item['task_id'] == tid:
                    scope = row.get('trace', data.get('trace'))
                    if scope is None or str(Path(scope).resolve()) == item['trace']:
                        candidates.append(item)
            bound = (len(candidates) == 1 and candidates[0]['record'] is not None and answer is not None
                     and answer['identity_resolved'] and answer['accepted_matches_current'])
            if bound:
                linked = candidates[0].get('attempt_run_ids', [candidates[0].get('run_id')])
                bound = any(r in answer['accepted_run_ids'] for r in linked)
            reasons = []
            if not bound:
                reasons.append('grader_run_identity_unavailable')
            answer_hash = row.get('answer_sha256')
            if answer_hash is None or not answer or answer_hash != answer['sha256']:
                reasons.append('grader_answer_version_unavailable')
            if answer and answer['task_type'] == 'coding':
                snapshots = row.get('workspace_sha256')
                if not isinstance(snapshots, dict) or not snapshots:
                    reasons.append('grader_workspace_version_unavailable')
                else:
                    workspace = Path(answer['path']).parent / 'workspace'
                    current = {p.relative_to(workspace).as_posix(): sha(audit.load(p, binary=True) or b'')
                               for p in workspace.rglob('*.py') if '__pycache__' not in p.parts}
                    if snapshots != current:
                        reasons.append('grader_workspace_version_unavailable')
            for reason in reasons:
                audit.issue('incomplete', reason, [ref] + ([answer['path']] if answer else []), task_id=tid)
            bound = bound and not reasons
            if answer:
                answer['grader_result'] = 'present'
                by_answer[tid].append((verdict, bound, ref))
            records.append({'task_id': tid, 'saved_verdict': verdict, 'bound_to_execution_and_version': bound,
                            'binding_limitations': reasons, 'points': awarded, 'source': ref})
        results.append({'path': str(path), 'scope': data.get('scope'), 'saved_verdict_counts': dict(verdicts),
                        'recomputed_saved_points': points, 'records': records})
        audit.issue('info', 'local_grader_scope', [str(path)])
    for answer in audit.answers:
        rows = by_answer[answer['task_id']]
        bound = {v for v, ok, _ in rows if ok}
        if len(bound) == 1:
            answer['correctness'] = next(iter(bound))
        elif len(bound) > 1:
            audit.issue('contradiction', 'conflicting_grader_verdicts', [ref for _, ok, ref in rows if ok], task_id=answer['task_id'])
        if not rows:
            audit.issue('incomplete', 'missing_task_grader_result', [answer['path']], task_id=answer['task_id'])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', default=[])
    parser.add_argument('--tasks-dir')
    parser.add_argument('--grader', action='append', default=[])
    parser.add_argument('--observations', action='append', default=[], help='Saved JSONL; follows raw_calls_path only')
    parser.add_argument('--out', required=True, help='New report outside source traces and tasks')
    args = parser.parse_args()
    if not args.trace and not args.observations:
        parser.error('Supply --trace or --observations')
    out = Path(args.out).resolve()
    if out.exists() or Path(args.out).is_symlink():
        parser.error('--out must be a new file')
    report = build_report(args.trace, tasks_dir=args.tasks_dir, graders=args.grader, observations=args.observations)
    protected = [Path(s['path']).resolve() for s in report['sources']]
    roots = [Path(a['trace']) for a in report['attempts']] + [Path(p).resolve() for p in args.trace]
    roots.extend(p.parent for p in protected if p.name == 'calls.jsonl')
    if args.tasks_dir:
        roots.append(Path(args.tasks_dir).resolve())
    if out in protected or any(out.is_relative_to(p) for p in roots):
        parser.error('--out must be outside source traces/tasks and must not replace evidence')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another writer appearing during the audit.
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'out': str(out), 'status': report['status'], 'issue_counts': report['issue_counts'],
                      'expense': report['expense']['totals']}, ensure_ascii=False, indent=2))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
