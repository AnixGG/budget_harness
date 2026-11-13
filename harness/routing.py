"""Public-input-only candidate plans; no workspace contents, labels or tool results."""
from dataclasses import dataclass, replace
import math

from .core import Config, digest, predict
from .planning import Plan
from .statistics import Statistics
from .accounting import initial_reserve

MODELS = ('qwen3.8-27b', 'qwen3.6-35b-a3b', 'qwen3.8-27b-noreason', 'qwen3.6-35b-a3b-noreason')
PREVIEW_BYTES = 2048
WORKFLOW = {'math': 'python_math', 'data_analysis': 'csv_python', 'search': 'file_search', 'coding': 'code_tests'}


@dataclass(frozen=True)
class Candidate:
    plan: Plan
    stages: tuple[Config, ...]
    single_first: bool
    forecast: dict
    signature: str


def candidates(task, base, *, models=MODELS, statistics=None, only=None, forecaster=None):
    task = {k: task[k] for k in ('task_id','task_type','instruction','reward','files','file_preview') if k in task}
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError('Choose nonempty unique allowed models')
    estimator = statistics or Statistics()
    workflow = WORKFLOW.get(task['task_type'], 'generic')
    stage_options = []
    for model in models:
        if model.endswith('-noreason'):
            # One full action; the second physical send is reserved for truncation recovery.
            cfg = replace(base, model=model, workflow=workflow, max_calls=2, expected_calls=1,
                          max_preview_bytes=PREVIEW_BYTES,single_action=True,auto_submit_coding=workflow=='code_tests')
            stage_options.append(('direct:' + model, (cfg,), True))
        else:
            cfg = replace(base, model=model, workflow=workflow,max_preview_bytes=PREVIEW_BYTES,single_action=False,
                          auto_submit_coding=workflow=='code_tests')
            stage_options.append(('reason:' + model, (cfg,), False))
    if forecaster is None:
        for first in [s for s in stage_options if s[2]]:
            for second in [s for s in stage_options if not s[2]]:
                stage_options.append(('cascade:' + first[1][0].model + '->' + second[1][0].model,
                                      first[1] + second[1], True))
    result = []
    for plan_id, stages, single in stage_options:
        if only and plan_id != only:
            continue
        signature = digest({'plan_id': plan_id, 'configs': [c.config_id for c in stages], 'single_first': single})
        # No measured switch rate yet: reserve the full fallback cost, not an invented saving.
        file_bytes = sum(f['bytes'] for f in task.get('files', []))
        # Public size-based correction to the old heuristic: account for a capped
        # file observation re-read by subsequent model turns. Coefficients remain
        # uncalibrated, identical for both model families.
        file_cost = sum(max(0, c.expected_calls-1)*min(file_bytes,c.max_tool_output_bytes)
                        / c.bytes_per_token / 10 for c in stages)
        handoff_cost = (stages[-1].expected_calls*(stages[0].max_tool_output_bytes+512)
                        / stages[-1].bytes_per_token / 10) if len(stages)>1 else 0
        heuristic = sum(predict(task, c)['budget_units'] for c in stages) + file_cost + handoff_cost
        forecast = forecaster.estimate(task, stages[0]) if forecaster is not None else estimator.estimate(task, signature, heuristic)
        if not math.isfinite(forecast['budget_units']) or forecast['budget_units'] < 0:
            raise ValueError('Invalid cost forecast')
        forecast.update(plan_id=plan_id, plan_signature=signature, workflow=workflow,
                        model_sequence=[c.model for c in stages],
                        minimum_start_budget=initial_reserve(task,stages[0]),
                        public_size_features={'text_bytes':len(task['instruction'].encode()),'file_bytes':file_bytes,
                                              'file_count':len(task.get('files',[]))},
                        heuristic_file_cost=file_cost, heuristic_handoff_cost=handoff_cost,
                        fallback_cascade='full sum of stage heuristics; switch rate unmeasured' if len(stages)>1 else None)
        plan = Plan(task['task_id'], plan_id, task['reward'], max(.001, forecast['budget_units']), forecast['p_success'])
        result.append(Candidate(plan, stages, single, forecast, signature))
    if not result:
        raise ValueError('Requested plan is unavailable')
    return result
