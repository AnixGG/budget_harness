import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from uuid import uuid4

from harness import agent, core
from harness.accounting import Ledger
from harness.deployed import LearnedRouter, runtime_manifest, solver_contract
from harness.routing import candidates
from harness.runner import run_plan
from harness.query_policy import eligible


class ReleaseTests(unittest.IsolatedAsyncioTestCase):
    async def case(self,actions,kind='data_analysis',usage=True,budget=100000):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'task';work=task/'workspace';work.mkdir(parents=True)
            (work/'sales.csv').write_text('code,units,price\n007,2,0.1\n7,3,0.2\n')
            core.atomic_json(task/'task.json',dict(task_id='t',task_type=kind,instruction='Compute requested aggregate',reward=1))
            cfg=core.Config();router=LearnedRouter()
            option=next(c for c in candidates(core.public_task(task),cfg,forecaster=router) if c.stages[0].model=='qwen3.8-27b-noreason')
            ledger=Ledger(root/'calls.jsonl',budget,10);ledger.deadline_monotonic=time.monotonic()+60
            seen=[]
            class Provider:
                def __init__(self,config,ledger,**kwargs):
                    self.ledger=ledger;self.config=config;self.client=SimpleNamespace(base_url='https://mock.invalid/v1');self.stop_reason=None
                async def invoke(self,messages,**kwargs):
                    seen.append(messages[0]['content'])
                    self.ledger.authorize({'messages':messages,'max_tokens':2048})
                    self.run_calls+=1;aid=str(uuid4())
                    core.append_json(self.ledger.path,{'kind':'request_started','attempt_id':aid})
                    self.ledger.record(attempt_id=aid,run_id=self.run_id,logical_call_id='test',raw_usage={'prompt_tokens':100,'completion_tokens':20} if usage else None,status=200)
                    return actions[min(self.run_calls-1,len(actions)-1)],'stop'
                async def close(self): pass
            old=(core.SYSTEM,core.WORKFLOWS['csv_python'],core.DIRECT_GUIDANCE,agent.run_python)
            result=await run_plan(task,option,ledger,root/'trace',provider_factory=Provider,backend='mock',allow_local_exec=True)
            self.assertEqual((core.SYSTEM,core.WORKFLOWS['csv_python'],core.DIRECT_GUIDANCE,agent.run_python),old)
            answer=(task/'answer.txt').read_text().strip() if (task/'answer.txt').exists() else None
            runs=[json.loads(l) for l in (root/'trace/runs.jsonl').read_text().splitlines()]
            return result,answer,seen,runs

    async def test_query_actual_config_and_proxy_forecast(self):
        q={'action':'table_query','query':{'table':'sales.csv','schema':{'units':'number','code':'string'},
           'filters':[{'column':'code','op':'eq','value':'007'}],
           'aggregates':[{'as':'total','op':'sum','column':'units'}]},'column':'total'}
        r,answer,seen,runs=await self.case([json.dumps(q)])
        self.assertEqual(answer,'2');self.assertEqual(r['actual']['physical_calls'],1)
        self.assertFalse(r['forecast']['estimates_calibrated_for_execution_policy'])
        self.assertEqual(runs[0]['config']['prompt_hash'],core.digest(seen[0]))
        self.assertIn('table-query-direct-v1',runs[0]['config']['revision'])

    async def test_python_fallback(self):
        a={'action':'python','reason':'unsupported operation','code':'submit_answer(42)'}
        self.assertEqual((await self.case([json.dumps(a)]))[1],'42')

    async def test_python_skip_terminal(self):
        a={'action':'python','reason':'unknown required data','code':"skip_task('missing price')"}
        r,answer,_,_=await self.case([json.dumps(a)])
        self.assertIsNone(answer);self.assertEqual(r['status'],'agent_skip_unverified')

    async def test_invalid_json_retry_keeps_direct_contract(self):
        actions=['not JSON',json.dumps({'action':'skip','reason':'unknown'})]
        r,answer,_,_=await self.case(actions)
        self.assertEqual(r['actual']['physical_calls'],2);self.assertIsNone(answer)

    async def test_other_task_uses_original_agent(self):
        r,answer,seen,runs=await self.case(['{"action":"answer","answer":"42"}'],kind='math')
        self.assertEqual(answer,'42');self.assertEqual(seen[0],core.SYSTEM+core.WORKFLOWS['python_math']+core.DIRECT_GUIDANCE)
        self.assertNotIn('table-query-direct-v1',runs[0]['config']['revision'])

    async def test_unknown_usage_remains_unknown(self):
        r,_,_,_=await self.case(['{"action":"skip","reason":"unknown"}'],usage=False)
        self.assertFalse(r['actual']['complete'])
        self.assertEqual(r['actual']['physical_calls'],1)

    async def test_budget_reserve_stops_before_send(self):
        r,answer,_,_=await self.case(['{"action":"skip","reason":"unknown"}'],budget=1)
        self.assertEqual(r['actual']['physical_calls'],0)
        self.assertIsNone(answer)

    async def test_invalid_query_never_saves_answer(self):
        q={'action':'table_query','query':{'table':'sales.csv','schema':{'units':'unknown'},
           'aggregates':[{'as':'s','op':'sum','column':'units'}]},'column':'s'}
        r,answer,_,_=await self.case([json.dumps(q)])
        self.assertIsNone(answer);self.assertEqual(r['status'],'tool_error')

    def test_only_measured_model_is_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);core.atomic_json(d/'task.json',dict(task_id='t',task_type='data_analysis',instruction='Sum',reward=1))
            opts=candidates(core.public_task(d),core.Config(),forecaster=LearnedRouter())
            self.assertEqual(sum(eligible(d,c) for c in opts),1)

    def test_contract_and_source_binding(self):
        router=LearnedRouter()
        self.assertEqual(len(router.entries),16)
        self.assertEqual(router.bundle['runtime_binding']['execution_policy']['name'],'table-query-direct-v1')
        self.assertIn('harness/table_engine.py',runtime_manifest())
