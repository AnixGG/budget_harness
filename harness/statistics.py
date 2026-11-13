"""Train-only summaries with explicit small-sample fallback; stdlib inference."""
from collections import defaultdict
import math
from statistics import mean, stdev

from .core import digest


def key(task_type, signature):
    return digest([task_type, signature])


def fit(rows, *, min_observations=5, min_groups=3):
    """Labels may train OFFLINE; only train, real, independently graded runs qualify."""
    groups = defaultdict(list)
    for row in rows:
        if row.get('split') != 'train' or row.get('backend') != 'real':
            continue
        if not row.get('usage_complete') or not row.get('eligible_for_cost_training'):
            continue
        if row.get('success') not in (True, False) or not isinstance(row.get('success'), bool):
            continue
        cost = row.get('cost_label')
        if not isinstance(cost, (float, int)) or not math.isfinite(cost) or cost < 0:
            continue
        signature = row.get('plan_signature')
        if not signature:
            continue  # A single-model trace is not a measurement of a new cascade.
        groups[key(row['task_type'], signature)].append(row)
    summaries = {}
    for k, observations in groups.items():
        values = [r['cost_label'] for r in observations]
        n = len(values)
        families = sorted({r['family_id'] for r in observations})
        # Repeats of one family do not create independent success observations.
        family_rates = [mean(float(r['success']) for r in observations if r['family_id'] == f) for f in families]
        p = (1 + sum(family_rates)) / (2 + len(families))
        summaries[k] = {'n': n, 'n_groups': len(families), 'mean_cost': mean(values),
                        'cost_stddev': stdev(values) if n > 1 else None,
                        'p_success': p, 'success_stddev': math.sqrt(p*(1-p)/(len(families)+3)),
                        'task_type': observations[0]['task_type'], 'plan_signature': observations[0]['plan_signature']}
    return {'schema_version': 1, 'provenance': 'real_provider_train_only',
            'min_observations': min_observations, 'min_groups': min_groups,
            'training_families': sorted({r['family_id'] for group in groups.values() for r in group}),
            'groups': summaries,
            'warning': 'Descriptive uncertainty; no optimal contest quantile or calibrated confidence interval claimed'}


class Statistics:
    def __init__(self, bundle=None, *, prior_success=.5):
        if not 0 <= prior_success <= 1:
            raise ValueError('Invalid success prior')
        self.bundle = bundle
        self.prior = prior_success
        if bundle and (bundle.get('schema_version') != 1 or bundle.get('provenance') != 'real_provider_train_only'):
            raise ValueError('Only real train summaries can drive routing')

    def estimate(self, task, signature, fallback_cost):
        group = (self.bundle or {}).get('groups', {}).get(key(task['task_type'], signature))
        enough = (group is not None and group['n'] >= self.bundle['min_observations']
                  and group['n_groups'] >= self.bundle['min_groups'])
        return {'budget_units': group['mean_cost'] if enough else fallback_cost,
                'p_success': group['p_success'] if enough else self.prior,
                'method': 'type_model_workflow_mean' if enough else 'heuristic_fallback',
                'n': group['n'] if group else 0, 'n_groups': group['n_groups'] if group else 0,
                'cost_stddev': group['cost_stddev'] if enough else None,
                'success_stddev': group['success_stddev'] if enough else None,
                'uncertainty': 'descriptive train dispersion' if enough else 'uncalibrated cost and explicit success prior',
                'prediction_point': 'before_agent_start', 'is_provider_usage': False}
