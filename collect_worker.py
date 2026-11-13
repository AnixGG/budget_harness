"""Single-run isolated worker, using the existing agent/provider/physical ledger."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from harness.accounting import Ledger, audit_events
from harness.agent import run_one
from harness.core import Config, atomic_json
from harness.provider import JiuwenProvider


async def execute(task, trace, job, *, local_mock=False):
    trace = Path(trace).resolve()
    task = Path(task).resolve()
    if any(trace.glob('*.jsonl')):
        raise ValueError('Worker requires fresh trace, accounting is restored by campaign controller')
    config = Config(**job['config'])
    ledger = Ledger(trace/'calls.jsonl', job['remaining_budget'], job['remaining_calls'], config.reasoning_in_completion)
    # Pin the campaign wall deadline at the send boundary as well as the agent boundary.
    import time
    ledger.deadline_monotonic = time.monotonic() + job['remaining_seconds']
    old = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='jiuwen-collector-runtime-') as runtime:
        os.chdir(runtime)
        provider = None
        try:
            provider = JiuwenProvider(config, ledger, mock=job['backend'] == 'mock')
            try:
                await asyncio.wait_for(run_one(task, config, provider, trace,
                                               allow_local_exec=local_mock), job['remaining_seconds'])
            except asyncio.TimeoutError:
                atomic_json(trace/'worker_stop.json', {'reason':'campaign_wall_timeout'})
        finally:
            if provider is not None:
                await provider.close()
            os.chdir(old)
    events = [json.loads(x) for x in (trace/'calls.jsonl').read_text().splitlines()] if (trace/'calls.jsonl').exists() else []
    atomic_json(trace/'audit.json', audit_events(events))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--local-mock', action='store_true')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    if args.local_mock and job['backend'] != 'mock':
        parser.error('Local execution is restricted to the fixed mock fixture')
    asyncio.run(execute(args.task, args.trace, job, local_mock=args.local_mock))
