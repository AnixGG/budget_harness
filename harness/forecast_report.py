"""Offline diagnostics from saved forecasts and physical usage; no runtime inference.

Run with python -m harness.forecast_report --trace TRACE --out NEW_REPORT.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .accounting import audit_events
from .core import atomic_json
from run_agent import evaluate, records


FILES = ('manifest.json', 'runs.jsonl', 'predictions.jsonl', 'plan_predictions.jsonl',
         'solutions.jsonl', 'calls.jsonl')


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def index(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[row.get(key)].append(row)
    return result


def metrics(rows):
    pairs = [r for r in rows if not r['excluded_reasons']]
    errors = [r['prediction'] - r['actual']['budget_units'] for r in pairs]
    n, total = len(pairs), len(rows)
    absolute = sum(abs(e) for e in errors)
    actual = sum(r['actual']['budget_units'] for r in pairs)
    return {
        'total_records': total, 'usable_pairs': n, 'excluded_records': total - n,
        'excluded_by_reason': dict(sorted(Counter(reason for r in rows for reason in r['excluded_reasons']).items())),
        'not_started_records': sum(r['not_started'] for r in rows),
        'known_zero_started_records': sum(r['actual'] is not None and r['actual']['complete']
                                          and r['actual']['physical_calls'] > 0
                                          and r['actual']['budget_units'] == 0 for r in rows),
        'mae': absolute / n if n else None, 'signed_bias': sum(errors) / n if n else None,
        'wape': absolute / actual if n and actual else None,
        'usable_pair_fraction': n / total if total else None,
        'denominators': {'mae_and_bias': n, 'wape_actual_budget_units': actual if n else None,
                         'usable_pair_fraction': total},
        'null_reasons': {**({'mae': 'no_usable_pairs', 'signed_bias': 'no_usable_pairs'} if not n else {}),
                         **({'wape': 'no_usable_pairs' if not n else 'zero_actual_sum'} if not n or not actual else {}),
                         **({'usable_pair_fraction': 'no_records'} if not total else {})}}


def provenance(record, forecast, manifest, *, plan=False):
    configs = record.get('configs') if plan else record.get('config')
    values = {'saved_forecast': forecast, 'configs' if plan else 'config': configs,
              'config_id': record.get('config_id'), 'plan_signature': record.get('plan_signature'),
              'data_id': forecast.get('data_id'), 'method': forecast.get('method')}
    # Keep all saved forecast/config version fields verbatim. Never open a current bundle.
    values['missing_fields'] = [k for k in ('method', 'data_id', 'plan_signature')
                                if values[k] is None]
    if configs is None:
        values['missing_fields'].append('configs' if plan else 'config')
    values['manifest_versions_available'] = bool(manifest)
    return values


def physical_actual(events, run_ids, ledger_present, reasons):
    if not ledger_present:
        reasons.append('missing_physical_ledger')
        return None
    selected = [e for e in events if e.get('run_id') in run_ids]
    try:
        audited = audit_events(selected)
    except (ValueError, KeyError, TypeError):
        reasons.append('invalid_physical_ledger')
        return None
    if audited['unfinished_attempts']:
        reasons.append('unfinished_requests')
    if not audited['totals']['complete'] and not audited['unfinished_attempts']:
        reasons.append('unknown_usage')
    return audited['totals']


def check_rollup(stored, actual, reasons):
    if not isinstance(stored, dict):
        reasons.append('missing_actual_rollup')
        return
    if stored.get('complete') is not True:
        reasons.append('unknown_usage' if stored.get('complete') is False else 'missing_usage_completeness')
    if actual is not None:
        for key in ('complete', 'physical_calls', 'budget_units'):
            a, b = stored.get(key), actual[key]
            equal = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) if number(a) and number(b) else a == b
            if key not in stored or not equal:
                reasons.append('actual_rollup_mismatch')
                break


def row_base(trace, identity, record, forecast, manifest, *, plan=False):
    return {'trace': str(trace), 'id': identity, 'task_id': record.get('task_id'),
            'status': record.get('status'), 'prediction': forecast.get('budget_units'),
            'provenance': provenance(record, forecast, manifest, plan=plan),
            'excluded_reasons': [], 'issues': [], 'not_started': False, 'actual': None}


def finish(row, backend):
    if backend == 'unknown':
        row['excluded_reasons'].append('missing_backend')
    if not number(row['prediction']):
        row['excluded_reasons'].append('missing_or_invalid_prediction')
    if row['actual'] is not None and row['actual']['complete']:
        if row['actual']['physical_calls'] == 0 and not any(
                r in row['excluded_reasons'] for r in ('missing_terminal_run', 'missing_terminal_solution',
                    'actual_rollup_mismatch', 'missing_actual_rollup', 'unknown_usage', 'missing_usage_completeness')):
            row['not_started'] = True
            row['excluded_reasons'].append('not_started')
    row['excluded_reasons'] = sorted(set(row['excluded_reasons']))
    row['error'] = row['prediction'] - row['actual']['budget_units'] if not row['excluded_reasons'] else None


def build_report(traces):
    groups = defaultdict(lambda: {'attempt_heuristic': [], 'selected_plan_forecast': [], 'admission_reserve': []})
    inputs, legacy = [], []
    for trace in traces:
        trace = Path(trace).resolve()
        if not trace.is_dir():
            raise ValueError(f'Trace directory does not exist: {trace}')
        loaded, hashes = {}, {}
        for name in FILES:
            path = trace / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            loaded[name] = (json.loads(path.read_text()) if name == 'manifest.json' else records(path)) if path.exists() else ({} if name == 'manifest.json' else [])
        manifest, runs, predictions, plans, solutions, events = (loaded[n] for n in FILES)
        groups[manifest.get('backend') or 'unknown']
        inputs.append({'trace': str(trace), 'sha256': hashes, 'manifest': manifest,
                       'missing_files': [n for n, h in hashes.items() if h is None]})
        ri, pi, si, fi = (index(rows, key) for rows, key in (
            (runs, 'run_id'), (predictions, 'run_id'), (solutions, 'solution_id'), (plans, 'solution_id')))
        ledger_present = hashes['calls.jsonl'] is not None

        def backend_for(*rows, run_ids=()):
            labels = {r.get('backend') for r in rows if r.get('backend')}
            labels.update(e['backend'] for e in events if e.get('run_id') in run_ids and e.get('backend'))
            if manifest.get('backend'):
                labels.add(manifest['backend'])
            return next(iter(labels)) if len(labels) == 1 else ('conflict' if labels else 'unknown')

        attempt_rows = {}
        # Include interrupted attempts with only a pre-run prediction or a started send.
        ids = set(ri) | set(pi) | {e.get('run_id') for e in events}
        for run_id in sorted(ids, key=str):
            finals, pre = ri.get(run_id, []), pi.get(run_id, [])
            record = finals[0] if finals else pre[0] if pre else {}
            forecast = record.get('prediction') or {}
            backend = backend_for(record, run_ids=[run_id])
            row = row_base(trace, run_id, record, forecast, manifest)
            reasons = row['excluded_reasons']
            if run_id is None or len(finals) > 1 or len(pre) > 1:
                reasons.append('missing_or_ambiguous_run_id')
            if not finals:
                reasons.append('missing_terminal_run')
            if pre and pre[0].get('prediction') != forecast:
                row['issues'].append('attempt_prediction_mismatch')
            if backend == 'conflict':
                reasons.append('backend_mismatch')
            row['actual'] = physical_actual(events, [run_id], ledger_present, reasons)
            check_rollup(record.get('actual'), row['actual'], reasons)
            finish(row, backend)
            groups[backend]['attempt_heuristic'].append(row)
            attempt_rows[run_id] = (row, backend)

        owners = Counter(run_id for s in solutions for run_id in set(s.get('attempt_run_ids') or []))
        for solution_id in sorted(set(si) | set(fi), key=str):
            finals, pre = si.get(solution_id, []), fi.get(solution_id, [])
            final, original = (finals[0] if finals else {}), (pre[0] if pre else {})
            forecast = original.get('forecast') or {}
            run_ids = final.get('attempt_run_ids')
            linked = [ri[r][0] for r in (run_ids or []) if ri.get(r)]
            backend = backend_for(final, original, *linked, run_ids=run_ids or [])
            row = row_base(trace, solution_id, original or final, forecast, manifest, plan=True)
            row.update(status=final.get('status'), attempt_run_ids=run_ids, final_forecast=final.get('forecast'))
            reasons = row['excluded_reasons']
            if solution_id is None or len(finals) > 1 or len(pre) > 1:
                reasons.append('missing_or_ambiguous_solution_id')
            if not pre or not forecast:
                reasons.append('missing_pre_execution_plan_forecast')
            if not finals:
                reasons.append('missing_terminal_solution')
            if pre and finals:
                if original.get('forecast') != final.get('forecast'):
                    row['issues'].append('plan_forecast_mismatch')
                if not original.get('plan_signature') or not final.get('plan_signature'):
                    reasons.append('missing_plan_signature')
                elif original['plan_signature'] != final['plan_signature']:
                    reasons.append('plan_signature_mismatch')
                if original.get('task_id') != final.get('task_id'):
                    reasons.append('task_identity_mismatch')
            signature = original.get('plan_signature')
            if forecast.get('plan_signature', signature) != signature:
                reasons.append('plan_signature_mismatch')
            if not isinstance(run_ids, list):
                reasons.append('missing_attempt_run_ids')
                run_ids = []
            elif len(set(run_ids)) != len(run_ids) or any(owners[r] > 1 for r in run_ids):
                reasons.append('ambiguous_attempt_ownership')
            for run_id in run_ids:
                if run_id not in attempt_rows:
                    reasons.append('missing_linked_run')
                else:
                    attempt, label = attempt_rows[run_id]
                    # Missing heuristic does not invalidate a plan cost label.
                    reasons.extend(r for r in attempt['excluded_reasons']
                                   if r not in {'missing_or_invalid_prediction', 'not_started'})
                    if attempt['task_id'] != final.get('task_id'):
                        reasons.append('task_identity_mismatch')
                    if label != backend:
                        reasons.append('backend_mismatch')
            if backend == 'conflict':
                reasons.append('backend_mismatch')
            row['actual'] = physical_actual(events, run_ids, ledger_present, reasons)
            check_rollup(final.get('actual'), row['actual'], reasons)
            finish(row, backend)
            groups[backend]['selected_plan_forecast'].append(row)
            reserve = forecast.get('minimum_start_budget')
            groups[backend]['admission_reserve'].append({
                'trace': str(trace), 'solution_id': solution_id, 'plan_signature': signature,
                'minimum_start_budget': reserve if number(reserve) else None,
                'missing_reason': None if number(reserve) else 'missing_or_invalid_saved_reserve',
                'source': 'plan_predictions.jsonl' if pre else None})
        try:
            legacy.append({'trace': str(trace), 'result': evaluate(runs)})
        except (KeyError, TypeError, ValueError):
            legacy.append({'trace': str(trace), 'result': None, 'reason': 'legacy_evaluate_requires_complete_old_schema'})

    for group in groups.values():
        for level in ('attempt_heuristic', 'selected_plan_forecast'):
            rows = group[level]
            group[level] = {'metrics': metrics(rows), 'records': rows}
        reserves = group['admission_reserve']
        group['admission_reserve'] = {'total_records': len(reserves),
                                     'available_records': sum(r['missing_reason'] is None for r in reserves),
                                     'purpose': 'Admission headroom only; no error metrics', 'records': reserves}
    return {'schema_version': 'offline-forecast-quality-v1', 'inputs': inputs,
            'warning': 'Diagnostic metrics, not official contest score. Correctness is never a filter.',
            'definitions': {'budget_unit': 'input_tokens / 10 + output_tokens, saved ledger normalization',
                            'error': 'prediction - actual', 'mae': 'sum(abs(error)) / usable_pairs; budget units',
                            'signed_bias': 'sum(error) / usable_pairs; budget units',
                            'wape': 'sum(abs(error)) / sum(actual) over usable pairs; fraction',
                            'usable_pair_fraction': 'usable_pairs / total_records at this level; fraction',
                            'exclusions': 'Reasons may overlap; excluded_records counts each row once.',
                            'actual': 'Unique physical sends for run_id, or union of attempt_run_ids for one solution; levels are never added.',
                            'historical_policy': 'Missing physical ledger or original plan forecast excludes the pair; no reconstruction.',
                            'mismatches': 'Original plan forecast retained on forecast mismatch; signature/identity/actual mismatch excludes pair.'},
            'interval_coverage': {'value': None, 'reason': 'Prediction intervals were not saved; usable_pair_fraction is not interval coverage.'},
            'backends': dict(sorted(groups.items())), 'legacy_attempt_evaluate': legacy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', required=True, help='Saved trace directory; repeat for separate runs')
    parser.add_argument('--out', required=True, help='New JSON file outside all source traces')
    args = parser.parse_args()
    traces = [Path(p).resolve() for p in args.trace]
    out = Path(args.out).resolve()
    if len(set(traces)) != len(traces):
        parser.error('Each trace may be supplied only once')
    if out.exists() or any(out.is_relative_to(p) for p in traces):
        parser.error('--out must be a new file outside source traces; historical files are read-only')
    report = build_report(traces)
    atomic_json(out, report)
    print(json.dumps({'out': str(out), 'backends': {b: {level: g[level]['metrics'] for level in
          ('attempt_heuristic', 'selected_plan_forecast')} for b, g in report['backends'].items()}}, indent=2))


if __name__ == '__main__':
    main()
