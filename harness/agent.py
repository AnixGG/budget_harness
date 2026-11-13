"""A single bounded agent loop. Prediction is persisted before provider invocation."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .accounting import StopBudget, aggregate
from .core import append_json, atomic_text, digest, messages_for, now, predict, public_task, solver_task
from .submission import (AnswerFiles, CODING_TEST_CODE, SubmissionError, read_answer,
                         test_hashes, validate_answer, verify_tests_unchanged, source_hashes)


class CaseDeadline(TimeoutError):
    """The attempt may no longer start work or accept an answer."""


def run_python(code, workspace, config, allow_local_exec, *, stop=None):
    if not allow_local_exec and os.getenv("HARNESS_CONTAINER") != "1":
        raise RuntimeError("Python tool requires Docker or explicit --allow-local-exec")
    # cwd is convenience, not a security sandbox: untrusted code belongs in Docker.
    # Do not give the tool process provider credentials.
    environment = {"PATH": str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", ""), "HOME": tempfile.gettempdir(),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    stop = stop or threading.Event()
    terminal=None
    with tempfile.TemporaryFile() as output, tempfile.TemporaryDirectory(prefix='jiuwen-tool-result-') as control:
        result_path=Path(control)/'terminal.json'
        # Keep -I and the clean environment, but explicitly allow workspace modules.
        # Compile the model code separately so __future__ imports remain legal.
        wrapper=("import sys,os,json\nfrom pathlib import Path\n"
                 "sys.path.insert(0,os.getcwd())\n"
                 f"_result_path=Path({str(result_path)!r})\n"
                 "def _finish(action, key, value):\n"
                 "    _result_path.write_text(json.dumps({'action':action,key:str(value)}),encoding='utf-8')\n"
                 "    raise SystemExit(0)\n"
                 "def submit_answer(value): _finish('answer','answer',value)\n"
                 "def skip_task(reason): _finish('skip','reason',reason)\n"
                 # One explicit import contract, backed by the same terminal protocol.
                 # Install only in this isolated child; never create a workspace file
                 # or import the harness/provider (and its credentials) into the tool.
                 "from types import ModuleType\n"
                 "_helper=ModuleType('submit_answer')\n"
                 "_helper.submit_answer=submit_answer\n"
                 "_helper.skip_task=skip_task\n"
                 "_helper.__all__=['submit_answer','skip_task']\n"
                 "sys.modules['submit_answer']=_helper\n"
                 f"_agent_code={code!r}\n"
                 "exec(compile(_agent_code,'<agent-tool>','exec'))\n")
        if stop.is_set():
            return {"exit_code": None, "timed_out": False, "cancelled": True,
                    "stdout": "", "truncated": False, "terminal": None}
        process = subprocess.Popen([sys.executable, "-I", "-c", wrapper], cwd=workspace,
                                   env=environment, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        tool_deadline = time.monotonic() + config.tool_timeout_seconds
        try:
            while process.poll() is None and not stop.is_set():
                remaining = tool_deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                # The worker owns spawn, process-group termination and reap. Cancelling
                # its asyncio caller signals this loop and then joins the worker.
                stop.wait(min(.02, remaining))
        finally:
            # Also terminate descendants left behind after the parent exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        content = output.read(config.max_tool_output_bytes + 1)
        if process.returncode==0 and not timed_out and not stop.is_set() and result_path.is_file():
            try:
                value=json.loads(read_answer(result_path))
                if (value.get('action')=='answer' and isinstance(value.get('answer'),str)
                        or value.get('action')=='skip' and isinstance(value.get('reason'),str) and value['reason'].strip()):
                    terminal=value
            except (ValueError,SubmissionError,AttributeError):
                pass
    observation = {"exit_code": process.returncode, "timed_out": timed_out,
                   "stdout": content[:config.max_tool_output_bytes].decode("utf-8", errors="replace"),
                   "truncated": len(content) > config.max_tool_output_bytes, "terminal":terminal}
    # Preserve the ordinary model-facing v2 protocol. Cancellation diagnostics
    # belong only to the stopped path, which cannot be followed by another send.
    if stop.is_set():
        observation['cancelled'] = True
    return observation


async def run_python_cancellable(code, workspace, config, allow_local_exec, *, on_cancel):
    stop = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(
        run_python, code, workspace, config, allow_local_exec, stop=stop))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stop.set()
        # Even repeated cancellation must not let finalization race the subprocess.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        on_cancel(worker.result())
        raise


async def run_one(directory, config, provider, artifact_dir, calibration=None, allow_local_exec=False,
                  *, single_action=False, original_tests=None, feedback=None, max_protocol_retries=0,
                  sufficiency_mode='off', sufficiency_phase='pre_api'):
    from .sufficiency import MODES, PHASES, diagnostic
    if sufficiency_mode not in MODES or sufficiency_phase not in PHASES:
        raise ValueError('Unknown sufficiency mode/phase')
    if type(max_protocol_retries) is not int or not 0 <= max_protocol_retries <= 1:
        raise ValueError('Protocol recovery allows zero or one retry')
    directory, artifact_dir = Path(directory).resolve(), Path(artifact_dir)
    task = solver_task(directory, config)
    single_action = single_action or config.single_action
    run_id = str(uuid4())
    prediction = predict(task, config, calibration)
    # No provider invocation or solving tool before this durable record.
    append_json(artifact_dir / "predictions.jsonl", {"run_id": run_id, "task_id": task["task_id"],
                "task": task, "prediction": prediction, "config": asdict(config),
                "max_protocol_retries": max_protocol_retries})
    provider.run_id, provider.run_calls = run_id, 0
    provider.stop_reason = None
    messages = messages_for(task, config)
    if feedback:
        messages.append({"role": "user", "content": json.dumps(feedback, ensure_ascii=False)})
    workspace = directory / "workspace"
    workspace.mkdir(exist_ok=True)
    answer_files = AnswerFiles(directory, workspace, artifact_dir / 'answer-files' / run_id)
    original_tests = (test_hashes(workspace) if task["task_type"] == "coding" else {}) if original_tests is None else original_tests
    previous_source=source_hashes(workspace) if task['task_type']=='coding' and config.auto_submit_coding else None
    backend = "mock" if str(provider.client.base_url).startswith("https://mock.invalid") else "real"
    started, status, error_type, answer = time.monotonic(), "call_limit", None, None
    answer_source, generation_retries, token_cap = None, 0, config.max_output_tokens
    protocol_retries, protocol_retries_scheduled, protocol_retry_pending = 0, 0, False
    observed_error = None
    last_observation = None
    cancellation = None
    sufficiency = []

    def event(kind, **values):
        append_json(artifact_dir / "events.jsonl", {"run_id": run_id, "created_at": now(), "kind": kind, **values})

    def inspect_data(phase, observation=None):
        check_deadline()
        report = diagnostic(workspace, task, sufficiency_mode, phase)
        if observation is not None and (observation.get('exit_code') != 0 or observation.get('timed_out')):
            # A failed tool cannot supply an absence certificate, even if a fresh
            # workspace read would find missing values after that tool's edits.
            report.update(result='Error', reason='Paid tool failed or timed out',
                          action='continue', complete=False, evidence=[], necessary_condition=None)
        elif observation is not None and observation.get('truncated'):
            report.update(result='Unknown', reason='Paid tool output truncated',
                          action='continue', complete=False, evidence=[], necessary_condition=None)
        sufficiency.append(report)
        event('data_sufficiency', **report)
        append_json(artifact_dir / 'sufficiency.jsonl', dict(run_id=run_id, **report))
        check_deadline()
        return report

    def check_deadline():
        deadline = getattr(provider.ledger, 'deadline_monotonic', None)
        if deadline is not None and time.monotonic() >= deadline:
            raise CaseDeadline('Task wall deadline')
        return deadline

    async def tool(code, phase, step):
        deadline = check_deadline()

        def interrupted(observation):
            nonlocal last_observation
            last_observation = observation
            event('tool_cancelled', step=step, phase=phase, **observation)

        work = run_python_cancellable(code, workspace, config, allow_local_exec, on_cancel=interrupted)
        if deadline is None:
            result = await work
        else:
            try:
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    result = await work
            except asyncio.TimeoutError as exc:
                raise CaseDeadline('Task wall deadline') from exc
        check_deadline()
        return result

    def reject_file(path, reason, step=None):
        record = answer_files.reject(path)
        if record:
            event('answer_quarantined', step=step, reason=reason, **record)

    async def submit(value, source, step):
        check_deadline()
        value = validate_answer(value, task["task_type"])
        if task["task_type"] == "coding":
            verify_tests_unchanged(workspace, original_tests)
            check = await tool(CODING_TEST_CODE, 'submission_tests', step)
            event("submission_tests", step=step, **check)
            if check["exit_code"] != 0 or check["timed_out"]:
                raise SubmissionError("Provided tests failed or timed out: " + check["stdout"])
            verify_tests_unchanged(workspace, original_tests)
        check_deadline()
        atomic_text(directory / "answer.txt", value + "\n")
        # A synchronous fsync can cross the deadline without yielding to the timer.
        check_deadline()
        event("answer_saved", step=step, source=source, answer=value)
        return value

    try:
        skip_for_evidence = False
        if sufficiency_mode != 'off' and sufficiency_phase == 'pre_api' and feedback is None:
            report = inspect_data('pre_api')
            if report['action'] == 'skip':
                # Use the existing refusal status and common finalization. Never
                # submit a placeholder or bypass AnswerFiles/physical accounting.
                status = 'agent_skip_unverified'
                event('data_sufficiency_skip_declared', reason=report['reason'])
                skip_for_evidence = True
        for step in range(config.max_calls):
            if skip_for_evidence:
                break
            check_deadline()
            if len(json.dumps(messages, ensure_ascii=False).encode()) > config.max_prompt_bytes:
                status = "context_limit"
                break
            status = "call_limit"
            calls_before = provider.run_calls
            try:
                deadline = check_deadline()
                invocation = provider.invoke(messages, max_tokens=token_cap)
                if deadline is None:
                    content, finish = await invocation
                else:
                    try:
                        async with asyncio.timeout(max(0, deadline-time.monotonic())):
                            content, finish = await invocation
                    except asyncio.TimeoutError:
                        check_deadline()
                        raise
            finally:
                if protocol_retry_pending and provider.run_calls > calls_before:
                    protocol_retries += 1
                protocol_retry_pending = False
            event("model_response", step=step, content=content, finish_reason=finish, max_tokens=token_cap)
            check_deadline()
            if finish == "length":
                status = "output_limit"
                larger_cap = min(token_cap * 2, config.max_retry_output_tokens)
                if (generation_retries >= config.max_generation_retries or larger_cap <= token_cap
                        or step + 1 >= config.max_calls or provider.run_calls >= config.max_calls):
                    break
                generation_retries += 1
                event("generation_retry", step=step, previous_cap=token_cap, next_cap=larger_cap,
                      retry=generation_retries)
                token_cap = larger_cap
                # Never execute, concatenate or put truncated tool code into the history.
                messages.append({"role": "user", "content":
                    "The last response hit its output limit and was discarded; no code was executed. "
                    "Return one short complete JSON action. If the answer is known, submit it immediately."})
                continue
            token_cap = config.max_output_tokens
            try:
                action = json.loads(content)
                if not isinstance(action, dict) or action.get("action") not in {"answer", "skip", "python"}:
                    raise ValueError("Unknown JSON action")
                if action["action"] == "python" and not isinstance(action.get("code"), str):
                    raise ValueError("Missing Python code")
            except (ValueError, TypeError):
                status = "invalid_response"
                observed_error = "invalid_response"
                event("response_rejected", step=step, reason="Invalid JSON action")
                if max_protocol_retries:
                    if (protocol_retries_scheduled >= max_protocol_retries or step + 1 >= config.max_calls
                            or provider.run_calls >= config.max_calls):
                        break
                    protocol_retries_scheduled += 1
                    protocol_retry_pending = True
                    event("protocol_retry", step=step, retry=protocol_retries_scheduled,
                          reason="invalid_response", max_tokens=config.max_output_tokens,
                          uses_existing_call_cap=True)
                    # Reject, never repair or execute malformed model text. Ask the same model again.
                    messages.append({"role": "user", "content":
                        "Your last response was not a complete valid JSON action and was rejected. "
                        "No code from it was executed. Return exactly one complete JSON object with "
                        "action answer, skip, or python, without prose or markdown fences. "
                        "Escape newlines inside the JSON code string correctly, so it decodes to valid Python. "
                        "The action must finish the task and submit its answer; do not only inspect or print."})
                    continue
                if single_action:
                    break
                messages.append({"role": "user", "content": "Return exactly one complete JSON action object, without prose."})
                continue
            if action["action"] == "answer":
                try:
                    answer = await submit(action.get("answer"), "json", step)
                except SubmissionError as exc:
                    status = "submission_rejected"
                    observed_error = "submission_rejected"
                    event("submission_rejected", step=step, reason=str(exc))
                    last_observation = {"submission_error": str(exc)}
                    if single_action:
                        break
                    messages.extend([{"role": "assistant", "content": content},
                                     {"role": "user", "content": str(exc)}])
                    continue
                answer_source, status = "json", "answered_ungraded"
                break
            if action["action"] == "skip":
                status = "agent_skip_unverified"
                break
            if action["action"] != "python" or not isinstance(action.get("code"), str):
                raise ValueError("Invalid action")
            observation = await tool(action["code"], 'action', step)
            last_observation = observation
            event("tool_observation", step=step, **observation)
            if sufficiency_mode != 'off' and sufficiency_phase == 'post_tool':
                inspect_data('post_tool', observation)
            changed_files = list(answer_files.changed())
            if observation["exit_code"] == 0 and not observation["timed_out"]:
                terminal=observation.get('terminal')
                if terminal and terminal['action']=='skip':
                    for path,_ in changed_files:
                        reject_file(path, 'Tool declared skip', step)
                    status='agent_skip_unverified'
                    event('tool_skip_declared',step=step,reason=terminal['reason'])
                    break
                if terminal and terminal['action']=='answer':
                    try:
                        answer=await submit(terminal['answer'],'tool_terminal',step)
                        answer_source,status='tool_terminal','answered_ungraded'
                        break
                    except SubmissionError as exc:
                        observed_error='submission_rejected'
                        observation['submission_error']=str(exc)
                        event('submission_rejected',step=step,source='tool_terminal',reason=str(exc))
                for path, source in changed_files:
                    try:
                        answer = await submit(read_answer(path), source, step)
                    except SubmissionError as exc:
                        observed_error = "submission_rejected"
                        event("submission_rejected", step=step, source=source, reason=str(exc))
                        observation["submission_error"] = str(exc)
                        # A model may have written ../answer.txt directly. Invalid submissions must not remain there.
                        if source == "task_file":
                            reject_file(path, str(exc), step)
                        continue
                    answer_source, status = source, "answered_ungraded"
                    break
                if answer is not None:
                    break
                if previous_source is not None:
                    try:
                        current_source=source_hashes(workspace)
                        changed=current_source!=previous_source
                        previous_source=current_source
                        if changed:
                            answer=await submit('DONE','coding_public_tests',step)
                            answer_source,status='coding_public_tests','answered_ungraded'
                            break
                    except SubmissionError as exc:
                        observed_error='submission_rejected'
                        observation['submission_error']=str(exc)
                        event('submission_rejected',step=step,source='coding_public_tests',reason=str(exc))
            else:
                observed_error = "tool_error"
                if previous_source is not None:
                    try:
                        previous_source=source_hashes(workspace)
                    except SubmissionError:
                        previous_source=None
                for path, source in changed_files:
                    event("submission_rejected", step=step, source=source, reason="Tool failed; answer not submitted")
                    if source == "task_file" and (path.is_file() or path.is_symlink()):
                        reject_file(path, 'Tool failed; answer not submitted', step)
            messages.extend([{"role": "assistant", "content": content},
                             {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}])
            if single_action:
                status = observed_error or "tool_needs_continuation"
                break
    except CaseDeadline:
        status = 'case_wall_timeout'
    except asyncio.CancelledError as exc:
        cancellation = exc
        deadline = getattr(provider.ledger, 'deadline_monotonic', None)
        status = 'case_wall_timeout' if deadline is not None and time.monotonic() >= deadline else 'cancelled'
    except StopBudget:
        status = "budget_stop"
    except Exception as exc:
        status = "budget_stop" if provider.stop_reason else "error"
        error_type = type(exc).__name__
    if answer is None:
        for path, _ in answer_files.paths:
            reject_file(path, status)
    if status in {'case_wall_timeout', 'cancelled'}:
        event('attempt_stopped', status=status, answer_saved=False)
    calls = [c for c in provider.ledger.calls if c["run_id"] == run_id]
    actual = aggregate(calls)
    result = {"schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
              "task_group": task["task_id"], "task_type": task["task_type"], "backend": backend,
              "config_id": config.config_id, "config": asdict(config), "task_hash": digest(task),
              "prediction": prediction, "actual": actual, "status": status, "error_type": error_type,
              "stop_reason": provider.stop_reason,
              "answer": answer, "answer_source": answer_source, "answer_saved": answer is not None,
              "preexisting_answer_preserved": answer is None and directory / 'answer.txt' in answer_files.backups,
              "generation_retries": generation_retries,
              "max_protocol_retries": max_protocol_retries, "protocol_retries": protocol_retries,
              "protocol_retries_scheduled": protocol_retries_scheduled,
              "success": None, "duration_seconds": time.monotonic() - started,
              "observed_error": observed_error, "last_observation": last_observation,
              "data_sufficiency": sufficiency,
              "created_at": now(), "local_tokenizer_usage": None,
              "usage_source": "synthetic_protocol_fixture" if backend == "mock" else "raw_provider_http_response"}
    append_json(artifact_dir / "runs.jsonl", result)
    if cancellation is not None:
        raise cancellation
    return result
