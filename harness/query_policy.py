"""Serial CSV execution overlay; historical routing forecasts are kept as proxies."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import MethodType

from . import agent, core
from .table_actions import SYSTEM, translate

POLICY = 'table-query-direct-v1'


def eligible(directory, candidate):
    return (len(candidate.stages)==1 and candidate.stages[0].workflow=='csv_python'
            and candidate.stages[0].model=='qwen3.8-27b-noreason'
            and core.public_task(directory)['task_type']=='data_analysis')


@contextmanager
def scope(allowed, artifacts, task_id=None):
    old_python, old_system, old_workflow, old_direct = agent.run_python, core.SYSTEM, core.WORKFLOWS['csv_python'], core.DIRECT_GUIDANCE
    source=Path(__file__).with_name('table_engine.py').read_text()
    def run_python(code, workspace, config, allow_local_exec, *, stop=None):
        with tempfile.TemporaryDirectory(prefix='table-query-metrics-') as tmp:
            metrics=Path(tmp)/'calls.jsonl'
            wrapped=(f"_tq={{}}\nexec(compile({source!r},'<table-query>','exec'),_tq)\n"
                f"_tq_allowed={tuple(allowed)!r}\n"
                "def table_query(q):\n"
                " import os,json,time\n"
                " t=time.monotonic()\n"
                " if not isinstance(q,dict) or q.get('table') not in _tq_allowed:\n"
                "  r={'ok':False,'result':None,'error':'Not a public task input'}\n"
                " else: r=_tq['execute'](q,os.getcwd())\n"
                f" with open({str(metrics)!r},'a') as f: f.write(json.dumps({{'ok':r['ok'],'error':r.get('error'),'seconds':time.monotonic()-t}})+'\\n')\n"
                " return r\n"
                f"exec(compile({code!r},'<agent-tool>','exec'))\n")
            obs=old_python(wrapped,workspace,config,allow_local_exec,stop=stop)
            calls=[json.loads(s) for s in metrics.read_text().splitlines()] if metrics.exists() else []
            core.append_json(Path(artifacts)/'table_metrics.jsonl',{'task_id':task_id,'calls':calls,'exit_code':obs['exit_code'],'timed_out':obs['timed_out']})
            return obs
    core.SYSTEM,core.WORKFLOWS['csv_python'],core.DIRECT_GUIDANCE=SYSTEM,'',''
    agent.run_python=run_python
    try:
        yield
    finally:
        agent.run_python=old_python
        core.SYSTEM,core.WORKFLOWS['csv_python'],core.DIRECT_GUIDANCE=old_system,old_workflow,old_direct


async def execute(directory,candidate,ledger,artifacts,ordinary,**kwargs):
    from .provider import JiuwenProvider
    factory=kwargs.pop('provider_factory',None) or JiuwenProvider
    allowed=[f['path'] for f in core.public_task(directory)['files']]
    def provider_factory(config,ledger,**options):
        provider=factory(config,ledger,**options)
        original=provider.invoke
        async def invoke(self,messages,**caps):
            messages=[dict(m,content='Invalid action. Return one JSON table_query, skip, or Python fallback with reason. No prose or fences.')
                if m['role']=='user' and m['content'].startswith('Your last response was not a complete valid JSON action') else m for m in messages]
            raw,finish=await original(messages,**caps)
            content,kind=translate(raw) if finish!='length' else (raw,'truncated')
            core.append_json(Path(artifacts)/'table_actions.jsonl',{'run_id':self.run_id,'kind':kind,'raw':raw,'finish':finish})
            return content,finish
        provider.invoke=MethodType(invoke,provider)
        return provider
    with scope(allowed,artifacts,core.public_task(directory)['task_id']):
        cfg=replace(candidate.stages[0],revision=candidate.stages[0].revision+'+'+POLICY)
        forecast=dict(candidate.forecast,execution_policy=POLICY,estimates_calibrated_for_execution_policy=False,
            forecast_support='historical_base_plan_proxy_for_query_overlay',historical_solver_identity=candidate.forecast.get('solver_identity'),
            execution_prompt_hash=cfg.prompt_hash)
        signature=core.digest({'plan_id':candidate.plan.plan_id,'configs':[cfg.config_id],'single_first':candidate.single_first})
        adapted=replace(candidate,stages=(cfg,),forecast=forecast,signature=signature)
        return await ordinary(directory,adapted,ledger,artifacts,provider_factory=provider_factory,**kwargs)
