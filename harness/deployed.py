"""Frozen train-only selected estimators, exported as aggregate tables for stdlib inference."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .core import digest

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_FILES = ('harness/core.py', 'harness/agent.py', 'harness/provider.py', 'harness/submission.py',
                 'harness/accounting.py', 'harness/routing.py', 'harness/runner.py', 'harness/planning.py',
                 'harness/deployed.py', 'harness/rate_limit.py', 'run_agent.py')
CONTROL_BUNDLE = ROOT / 'configs/learned-router-v2.json'
# Keep RUNTIME_FILES as the original control manifest, reproducible from Git.
RUNTIME_EXTENSIONS = ('harness/sufficiency.py', 'harness/sufficiency_text.py',
                      'harness/table_engine.py', 'harness/table_actions.py', 'harness/query_policy.py')
DEFAULT_BUNDLE = ROOT / 'configs/learned-router-r1-submit.json'
RUNTIME_REVIEW = ROOT / 'configs/learned-router-r1-submit-review.json'


class BundleCompatibilityError(ValueError):
    """Safe startup diagnostic: paths/hashes only, never provider payloads."""


def runtime_manifest():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (*RUNTIME_FILES, *RUNTIME_EXTENSIONS)}


def runtime_hash():
    return digest(runtime_manifest())


def solver_identity(config):
    values = asdict(config) if not isinstance(config, dict) else dict(config)
    # Routing/accounting integration changes the whole-program hash, not the measured prompts/caps.
    values.pop('implementation_hash', None)
    return digest(values)


def solver_contract():
    """All sixteen public-input solver configurations, with implementation tracked separately."""
    from .core import Config
    from .routing import candidates, WORKFLOW
    contract = []
    for kind in sorted(WORKFLOW):
        task = dict(task_id='compatibility', task_type=kind, instruction='Compatibility fixture', reward=1, files=[])
        for option in candidates(task, Config()):
            if len(option.stages) != 1:
                continue
            config = asdict(option.stages[0])
            config.pop('implementation_hash')
            contract.append({'task_type': kind, 'plan_id': option.plan.plan_id,
                             'solver_identity': solver_identity(config), 'config': config})
    return sorted(contract, key=lambda row: (row['task_type'], row['plan_id']))


def validate_binding(bundle):
    """A reviewed derivative retains every learned value and binds exact runtime bytes."""
    binding = bundle.get('runtime_binding')
    if binding is None:
        return  # Original schema and separate synthetic fixtures remain supported.
    if binding.get('schema_version') != 'reviewed-runtime-binding-v1':
        raise BundleCompatibilityError('Unsupported runtime binding schema')
    parent_bytes = CONTROL_BUNDLE.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != binding.get('parent_bundle_sha256'):
        raise BundleCompatibilityError('Frozen v2 bundle changed since compatibility review')
    parent = json.loads(parent_bytes)
    inherited = {k: v for k, v in bundle.items() if k not in ('runtime_hash', 'runtime_binding')}
    if inherited != {k: v for k, v in parent.items() if k != 'runtime_hash'}:
        raise BundleCompatibilityError('Reviewed bundle must preserve every inherited estimate and execution cap')
    review_bytes = RUNTIME_REVIEW.read_bytes()
    if hashlib.sha256(review_bytes).hexdigest() != binding.get('review_sha256'):
        raise BundleCompatibilityError('Runtime compatibility review changed; review and bind again')
    review = json.loads(review_bytes)
    if (review.get('schema_version') != 'runtime-compatibility-review-v1'
            or review.get('reviewed_runtime_files') != runtime_manifest()
            or review.get('solver_contract') != solver_contract()
            or review.get('parent_bundle_sha256') != binding.get('parent_bundle_sha256')
            or binding.get('source_runtime_hash') != parent['runtime_hash']
            or binding.get('forecast_support') != 'inherited_v2_base_plan_estimates'
            or binding.get('execution_policy') != review.get('execution_policy')
            or binding.get('real_quality_revalidated') is not False):
        raise BundleCompatibilityError('Runtime/configuration differs from the reviewed compatibility contract')


class LearnedRouter:
    def __init__(self, path=DEFAULT_BUNDLE):
        self.bundle = json.loads(Path(path).read_text())
        b = self.bundle
        if b.get('schema_version') != 'selected-router-v1' or b.get('provenance') != 'real_provider_train_only':
            raise ValueError('Expected a frozen train-only routing bundle')
        if b.get('runtime_hash') != runtime_hash():
            raise BundleCompatibilityError('Runtime changed since export; verify compatibility and export again. '
                f'Bundle hash={b.get("runtime_hash")}; current hash={runtime_hash()}. '
                'Keep configs/learned-router-v2.json frozen; use a separately reviewed current bundle. '
                'Run python -m harness.bundle check for details.')
        validate_binding(b)
        if b.get('test_rows_consumed') != 0 or not b.get('plans'):
            raise ValueError('Invalid deployment provenance')
        self.max_protocol_retries = b.get('protocol_recovery', {}).get('max_retries', 0)
        if type(self.max_protocol_retries) is not int or not 0 <= self.max_protocol_retries <= 1:
            raise ValueError('Protocol recovery allows zero or one retry')
        self.entries = {}
        for entry in b['plans']:
            key = (entry['task_type'], entry['solver_identity'])
            cost, probability = entry['budget_units'], entry['p_success']
            if key in self.entries or not math.isfinite(cost) or cost <= 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid or duplicate learned plan')
            self.entries[key] = entry
        self.execution_budget = b['execution_budget']
        self.case_wall_seconds = b['case_wall_seconds']
        if self.execution_budget <= 0 or self.case_wall_seconds <= 0:
            raise ValueError('Positive measured per-plan caps required')

    def estimate(self, task, config):
        entry = self.entries.get((task['task_type'], solver_identity(config)))
        if entry is None:
            raise ValueError('No measured support for this model/workflow/prompt/caps')
        overlay = task['task_type']=='data_analysis' and config.model=='qwen3.8-27b-noreason'
        return {key: entry[key] for key in ('budget_units', 'p_success', 'n', 'n_groups')} | {
            'method': 'historical_base_plan_proxy' if overlay else 'selected_train_estimators',
            'execution_policy': 'table-query-direct-v1' if overlay else 'base',
            'estimates_calibrated_for_execution_policy': not overlay, 'cost_estimator': self.bundle['selected']['cost'],
            'success_estimator': self.bundle['selected']['success'], 'data_id': self.bundle['data_id'],
            'training_plan_signature': entry['training_plan_signature'],
            'solver_identity': entry['solver_identity'], 'execution_budget': self.execution_budget,
            'max_protocol_retries': self.max_protocol_retries,
            'forecast_support': 'historical_base_plan_proxy_for_query_overlay' if overlay else self.bundle.get('runtime_binding', {}).get('forecast_support', 'measured_v2_base_plans'),
            'real_quality_revalidated': False if self.bundle.get('runtime_binding') else None,
            'recovery_estimates_calibrated': False,
            'case_wall_seconds': self.case_wall_seconds, 'prediction_point': 'before_agent_start',
            'is_provider_usage': False, 'uncertainty': 'Base-plan train aggregates. Conditional format-retry cost/success not measured; probabilities not separately calibrated. Runtime fixes do not establish new real-provider quality.'}
