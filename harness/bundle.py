"""Check or bind inherited v2 forecasts to an explicitly reviewed runtime, offline.

No models are trained or unpickled, no datasets are read and no provider is used.
Binding preserves all estimator values; it is not a new real-quality evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .core import digest
from .deployed import (ROOT, CONTROL_BUNDLE, DEFAULT_BUNDLE, RUNTIME_REVIEW,
                       LearnedRouter, RUNTIME_FILES,
                       runtime_hash, runtime_manifest, solver_contract, validate_binding)

# These entire modules must still match the original control for this migration.
# Behavioral fixes in the other files are explicitly reviewed, never auto-approved.
UNCHANGED_MODULES = ('harness/core.py', 'harness/provider.py', 'harness/routing.py', 'harness/rate_limit.py')
VERIFICATION_FILES = ('harness/bundle.py', 'tests/test_bundle_binding.py',
                      'tests/test_deployed_router.py', 'tests/test_replay.py',
                      'tests/test_router.py', 'tests/test_cancellation.py',
                      'tests/fixtures/budget-replay.json', 'requirements.lock',
                      'tests/test_sufficiency.py', 'tests/test_submission.py',
                      'harness/sufficiency_report.py', 'corpus_lab/p2_fixtures.py',
                      'tests/fixtures/p2/public.json', 'tests/fixtures/p2/private.json',
                      'tests/test_sufficiency_text.py', 'harness/sufficiency_applicability.py',
                      'tests/fixtures/p2-text/public.json', 'tests/fixtures/p2-text/private.json',
                      'tests/test_p2_review.py', 'experiments/p2_review.py', 'experiments/p2_review_grader.py',
                      'tests/fixtures/p2-review/public.json', 'tests/fixtures/p2-review/private.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_bytes(revision, name):
    if not re.fullmatch('[0-9a-f]{40}', revision) or name not in (*RUNTIME_FILES, 'configs/learned-router-v2.json'):
        raise ValueError('Expected full control commit and an allowed source path')
    return subprocess.run(['git', 'show', revision+':'+name], cwd=ROOT,
                          check=True, capture_output=True).stdout


def check_review(review, parent):
    if review.get('schema_version') != 'runtime-compatibility-review-v1':
        raise ValueError('Expected runtime-compatibility-review-v1')
    if review.get('parent_bundle_sha256') != sha256(CONTROL_BUNDLE):
        raise ValueError('Frozen parent bundle differs from the review')
    revision = review['control_git_revision']
    if git_bytes(revision, 'configs/learned-router-v2.json') != CONTROL_BUNDLE.read_bytes():
        raise ValueError('Control bundle differs from its saved Git revision')
    baseline = {name: hashlib.sha256(git_bytes(revision, name)).hexdigest() for name in RUNTIME_FILES}
    if baseline != review.get('control_runtime_files') or digest(baseline) != parent['runtime_hash']:
        raise ValueError('Control source manifest does not reproduce the saved runtime hash')
    current = runtime_manifest()
    if current != review.get('reviewed_runtime_files'):
        raise ValueError('Runtime changed after review; inspect differences and create a new review')
    if any(current[name] != baseline[name] for name in UNCHANGED_MODULES):
        raise ValueError('Measured prompt/config/provider/routing modules changed; this migration does not cover them')
    changed = {name for name in current if current[name] != baseline.get(name)}
    if set(review.get('reviewed_changes', {})) != changed or any(not v for v in review['reviewed_changes'].values()):
        raise ValueError('Every runtime difference needs an explicit reviewed explanation')
    contract = solver_contract()
    if contract != review.get('solver_contract'):
        raise ValueError('Solver configuration changed since review')
    measured = {(p['task_type'], p['plan_id'], p['solver_identity']) for p in parent['plans']}
    actual = {(p['task_type'], p['plan_id'], p['solver_identity']) for p in contract}
    if len(measured) != 16 or actual != measured:
        raise ValueError('Current solver prompts/models/caps do not match all sixteen measured configurations')
    if (review.get('forecast_support') != 'inherited_v2_base_plan_estimates'
            or review.get('real_quality_revalidated') is not False or not review.get('limitations')):
        raise ValueError('Inherited estimates and unmeasured runtime changes must be explicit')
    if not set(VERIFICATION_FILES) <= set(review.get('verification_sources', {})):
        raise ValueError('Required verification sources are missing from review')
    for name, expected in review['verification_sources'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError('Verification source differs from review: '+name)
    return contract


def bind(output=DEFAULT_BUNDLE):
    """Read an already-reviewed manifest; never silently refresh it after code changes."""
    output = Path(output).resolve()
    protected = {CONTROL_BUNDLE.resolve(), RUNTIME_REVIEW.resolve()}
    if output in protected or output.exists():
        raise ValueError('A new output is required; frozen control and review cannot be overwritten')
    parent = json.loads(CONTROL_BUNDLE.read_text())
    review = json.loads(RUNTIME_REVIEW.read_text())
    check_review(review, parent)
    bundle = dict(parent, runtime_hash=runtime_hash(), runtime_binding={
        'schema_version': 'reviewed-runtime-binding-v1',
        'parent_bundle': 'configs/learned-router-v2.json',
        'parent_bundle_sha256': sha256(CONTROL_BUNDLE),
        'source_runtime_hash': parent['runtime_hash'],
        'control_git_revision': review['control_git_revision'],
        'review': RUNTIME_REVIEW.relative_to(ROOT).as_posix(),
        'review_sha256': sha256(RUNTIME_REVIEW),
        'forecast_support': 'inherited_v2_base_plan_estimates',
        'real_quality_revalidated': False,
        'limitations': review['limitations']})
    validate_binding(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        json.dump(bundle, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    return {'bundle': str(output), 'sha256': sha256(output), 'runtime_hash': bundle['runtime_hash'],
            'inherited_plans': len(bundle['plans']), 'source_bundle_sha256': sha256(CONTROL_BUNDLE),
            'review_sha256': sha256(RUNTIME_REVIEW), 'forecast_support': bundle['runtime_binding']['forecast_support'],
            'real_quality_revalidated': False, 'new_api_calls': 0}


def check(path=DEFAULT_BUNDLE):
    router = LearnedRouter(path)
    return {'bundle': str(path), 'sha256': sha256(path), 'runtime_hash': runtime_hash(),
            'runtime_matches': router.bundle['runtime_hash'] == runtime_hash(),
            'plans': len(router.entries), 'runtime_binding': router.bundle.get('runtime_binding'),
            'control_bundle_sha256': sha256(CONTROL_BUNDLE), 'new_api_calls': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('bind', help='Create a new bundle only for the checked-in reviewed runtime').add_argument('--out', type=Path, default=DEFAULT_BUNDLE)
    sub.add_parser('check', help='Validate hash and inherited contract without any provider').add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    try:
        result = bind(args.out) if args.command == 'bind' else check(args.bundle)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
