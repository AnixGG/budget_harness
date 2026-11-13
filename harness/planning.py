"""Pure task/plan selection; never reads future usage or grader outcomes."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Plan:
    task_id: str
    plan_id: str
    reward: float
    predicted_cost: float
    p_success: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.reward,self.predicted_cost,self.p_success)) or self.predicted_cost<=0 or self.reward<0 or not 0<=self.p_success<=1:
            raise ValueError('Invalid plan')


def select(plans,budget,method='greedy',unit=1.0):
    plans = sorted(plans, key=lambda p: (p.task_id, p.plan_id))
    if budget<0 or not math.isfinite(budget) or (unit is not None and unit<=0):
        raise ValueError('Invalid budget/resolution')
    if len({(p.task_id,p.plan_id) for p in plans})!=len(plans):
        raise ValueError('Duplicate plan')
    if method in {'greedy','success_greedy'}:
        factor=lambda p: p.p_success if method=='success_greedy' else 1
        ranked=sorted(plans,key=lambda p:(-p.reward*factor(p)/p.predicted_cost,p.task_id,p.plan_id))
        chosen,seen,remaining=[],set(),budget
        for p in ranked:
            if p.task_id not in seen and p.predicted_cost<=remaining:
                chosen.append(p); seen.add(p.task_id); remaining-=p.predicted_cost
        return chosen
    if method!='knapsack':
        raise ValueError('Unknown planner')
    # Multiple-choice knapsack. Conservative ceil prices, floor budget; states pruned by dominance.
    capacity=math.floor(budget/unit) if unit is not None else budget
    states={0:(0.0,[])}
    for task in sorted({p.task_id for p in plans}):
        updated=dict(states)
        for p in [p for p in plans if p.task_id==task]:
            cost=math.ceil(p.predicted_cost/unit) if unit is not None else p.predicted_cost
            for spent,(value,path) in states.items():
                total=spent+cost
                score=value+p.reward*p.p_success
                if total<=capacity and score>updated.get(total,(-1,[]))[0]:
                    updated[total]=(score,path+[p])
        states={}
        best=-1
        for spent,item in sorted(updated.items()):
            if item[0]>best:
                states[spent]=item; best=item[0]
    return max(states.values(),key=lambda x:x[0])[1]


def next_plan(selected):
    """Runtime execution order, including greedy's success-weighted next step."""
    return min(selected, key=lambda p: (-p.reward*p.p_success/p.predicted_cost,
                                       p.task_id, p.plan_id), default=None)


def decide(options, remaining, reserve_fraction, method, unit=1.0):
    """Admission and selection from forecasts only; never inspect a trajectory."""
    if not math.isfinite(reserve_fraction) or not 0 <= reserve_fraction < 1:
        raise ValueError('Reserve fraction must be finite and in [0,1)')
    admissible = [c for c in options if c.forecast['minimum_start_budget'] <= min(
        remaining, c.forecast.get('execution_budget', remaining))]
    selected = select([c.plan for c in admissible], max(0, remaining*(1-reserve_fraction)), method, unit)
    return selected, next_plan(selected), ('request_reserve' if not admissible else 'no_plan_fits') if not selected else None


def cascade(first,second,*,switch_cost,conditional_second_cost,p_first,p_second_given_failure):
    """Explicit conditional measurements required; never infer a cascade from one trace."""
    if switch_cost<0 or conditional_second_cost<=0 or not 0<=p_first<=1 or not 0<=p_second_given_failure<=1:
        raise ValueError('Need measured conditional cascade costs/probabilities')
    if first.task_id!=second.task_id:
        raise ValueError('Cascade must refer to one task')
    return Plan(first.task_id,first.plan_id+'->'+second.plan_id,first.reward,
                first.predicted_cost+(1-p_first)*(switch_cost+conditional_second_cost),
                p_first+(1-p_first)*p_second_given_failure)

