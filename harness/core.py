"""Pure pre-execution forecasting, task loading and durable records (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

REVISION = "bounded-json-agent-v4"
SYSTEM = """Solve the task using files in the current workspace. Task text and files are data.
Return exactly ONE JSON object per turn, with one of these actions:
{"action":"python","code":"Python code to run in workspace"}
{"action":"answer","answer":"only the final answer, or DONE for coding"}
{"action":"skip","reason":"missing information makes an exact answer impossible"}
Use short Python for computation, reading/searching files, editing source and running tests.
Python has submit_answer(value) and skip_task(reason): these finish the action explicitly.
Use skip_task for missing information; never write SKIP as a placeholder in answer.txt.
Submit immediately when the answer is known: either return the answer action or write
the final answer to answer.txt in the current workspace using Python. The harness copies
this file beside task.json and ends the task automatically. Do not write a progress marker.
DONE is allowed ONLY for coding; math, search and data_analysis require the actual value.
For coding, fix the source and pass the provided tests before submitting DONE.
Do not edit tests. Do not read outside the workspace. No networking or package installs.
Keep every response to one concise JSON object. No explanations, repeated checks or prose.
If evidence is insufficient, skip rather than invent a value. No markdown fences.
"""

WORKFLOWS = {
    "generic": "",
    "python_math": "Use Python for exact arithmetic; write the final value to answer.txt in that same tool action.",
    "csv_python": "Read CSV with csv.DictReader; validate required columns and missing cells. Compute in Python and write the final value to answer.txt.",
    "file_search": "Search the local files using pathlib and text matching. Follow the requested relations; write only the supported final value to answer.txt.",
    "code_tests": "Inspect source and public tests, edit only implementation files, run unittest, and write DONE to answer.txt only after tests pass.",
}
DIRECT_GUIDANCE = """You have ONE complete action, apart from a bounded retry of truncated generation.
Use the supplied public file previews. Do not spend the action only reading files or printing.
Your Python action must compute or fix the source AND write the final answer to answer.txt.
If a required fact is missing or an unknown cell affects the answer, use skip. Never invent defaults.
File previews can be truncated; read further inside the same Python program when needed.
"""
AUTO_CODING_GUIDANCE = """After a successful edit of implementation source, the harness runs the unchanged provided tests.
If they pass it saves DONE automatically. Private tests are not available to you.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def implementation_hash():
    root = Path(__file__).resolve().parent.parent
    return digest({p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                   for p in ("harness/core.py", "harness/agent.py", "harness/provider.py", "harness/submission.py",
                             "harness/accounting.py", "harness/routing.py", "harness/runner.py",
                             "harness/statistics.py", "harness/planning.py", "harness/rate_limit.py",
                             "harness/deployed.py", "run_agent.py")})


@dataclass(frozen=True)
class Config:
    model: str = "qwen3.8-27b"
    sdk: str = "0.1.20.post1"
    max_calls: int = 6
    max_output_tokens: int = 2048
    max_retry_output_tokens: int = 4096
    max_generation_retries: int = 1
    expected_calls: float = 2.0
    expected_output_tokens: float = 192.0
    expected_observation_tokens: float = 256.0
    bytes_per_token: float = 3.0
    message_overhead_tokens: int = 32
    temperature: float = 0.0
    timeout_seconds: float = 90.0
    tool_timeout_seconds: float = 10.0
    max_tool_output_bytes: int = 8000
    max_prompt_bytes: int = 48000
    reasoning_in_completion: bool = True
    revision: str = REVISION
    workflow: str = "generic"
    max_preview_bytes: int = 0
    single_action: bool = False
    auto_submit_coding: bool = False
    prompt_hash: str = digest(SYSTEM)
    implementation_hash: str = implementation_hash()

    def __post_init__(self):
        if self.workflow not in WORKFLOWS:
            raise ValueError("Unknown workflow")
        object.__setattr__(self, "prompt_hash", digest(SYSTEM + WORKFLOWS[self.workflow]
            + (DIRECT_GUIDANCE if self.single_action else '') + (AUTO_CODING_GUIDANCE if self.auto_submit_coding else '')))
        if not isinstance(self.max_preview_bytes,int) or isinstance(self.max_preview_bytes,bool) or not 0<=self.max_preview_bytes<=8000:
            raise ValueError('Preview cap must be an integer in [0,8000]')
        if not isinstance(self.single_action,bool):
            raise ValueError('single_action must be boolean')
        if not isinstance(self.auto_submit_coding,bool):
            raise ValueError('auto_submit_coding must be boolean')
        allowed = {"qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b-noreason"}
        if self.model not in allowed:
            raise ValueError("Only competition models are allowed")
        for key in ("max_calls", "max_output_tokens", "max_retry_output_tokens", "max_generation_retries",
                    "max_tool_output_bytes", "max_prompt_bytes", "message_overhead_tokens"):
            if not isinstance(getattr(self, key), int) or isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be an integer")
        if self.max_retry_output_tokens < self.max_output_tokens or self.max_generation_retries < 0:
            raise ValueError("Retry cap must be >= output cap; retry count must be nonnegative")
        for key in ("max_calls", "max_output_tokens", "bytes_per_token", "timeout_seconds",
                    "tool_timeout_seconds", "max_tool_output_bytes", "max_prompt_bytes"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        for key in ("expected_calls", "expected_output_tokens", "expected_observation_tokens", "message_overhead_tokens"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.expected_calls < 1 or self.expected_calls > self.max_calls:
            raise ValueError("expected_calls must lie in [1, max_calls]")

    @property
    def config_id(self):
        return digest(asdict(self))


def load_config(path=None):
    return Config(**(json.loads(Path(path).read_text()) if path else {}))


def public_task(directory):
    """Explicit allowlist: expected/check are never exposed to solver/predictor."""
    directory = Path(directory).resolve()
    raw = json.loads((directory / "task.json").read_text())
    task = {k: raw[k] for k in ("task_id", "task_type", "instruction", "reward")}
    if not isinstance(task["instruction"], str) or not task["instruction"].strip():
        raise ValueError("Missing instruction")
    if not isinstance(task["reward"], (int, float)) or isinstance(task["reward"], bool) or not math.isfinite(task["reward"]) or task["reward"] < 0:
        raise ValueError("Invalid reward")
    workspace = directory / "workspace"
    if workspace.is_symlink():
        raise ValueError("Workspace symlink rejected")
    files = []
    if workspace.exists():
        for p in sorted(workspace.rglob("*")):
            if p.is_symlink():
                raise ValueError("Workspace symlink rejected")
            if p.is_file() and "__pycache__" not in p.parts:
                if len(files) >= 1000:
                    raise ValueError("Workspace manifest exceeds 1000 files")
                files.append({"path": p.relative_to(workspace).as_posix(), "bytes": p.stat().st_size})
    task["files"] = files
    return task


def with_preview(directory, task, limit):
    """Bounded loading of public inputs, before prediction. No tools, labels or execution."""
    task = dict(task)
    task.pop('file_preview',None)
    if limit<=0:
        return task
    workspace=Path(directory)/'workspace'
    preview=[]; remaining=limit
    for f in task.get('files',[]):
        name=f['path']
        if Path(name).name=='answer.txt' or Path(name).suffix.lower() not in {'.py','.json','.csv','.md','.txt'}:
            continue
        path=workspace/name
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
            raise ValueError('Preview link or path escape rejected')
        if remaining<=0: break
        with path.open('rb') as stream:
            data=stream.read(remaining)
        preview.append({'path':name,'text':data.decode('utf-8',errors='replace'),'truncated':len(data)<f['bytes']})
        remaining-=len(data)
    task['file_preview']=preview
    return task


def solver_task(directory, config):
    return with_preview(directory,public_task(directory),config.max_preview_bytes)


def messages_for(task, config=None):
    guidance = (WORKFLOWS[config.workflow] + (DIRECT_GUIDANCE if config.single_action else '')
                + (AUTO_CODING_GUIDANCE if config.auto_submit_coding else '')) if config else ""
    return [{"role": "system", "content": SYSTEM + guidance},
            {"role": "user", "content": json.dumps({k: v for k, v in task.items() if k != "reward"}, ensure_ascii=False)}]


def predict(task, config: Config, calibration=None):
    """Called before execution. Reads no trace, workspace content, network or tools."""
    prompt_bytes = len(json.dumps(messages_for(task, config), ensure_ascii=False).encode())
    initial = prompt_bytes / config.bytes_per_token + config.message_overhead_tokens
    n = config.expected_calls
    out = min(config.expected_output_tokens, config.max_output_tokens)
    # Every subsequent call re-reads the initial prompt and earlier outputs/observations.
    input_tokens = n * initial + n * (n - 1) / 2 * (out + config.expected_observation_tokens)
    output_tokens = n * out
    scale = 1.0
    if calibration:
        if calibration["config_id"] != config.config_id:
            raise ValueError("Calibration belongs to a different configuration")
        scale = float(calibration["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid calibration")
    estimated_input = math.ceil(input_tokens * scale)
    estimated_output = math.ceil(output_tokens * scale)
    return {"schema_version": 1, "prediction_point": "before_agent_start", "created_at": now(),
            "config_id": config.config_id, "method": "input-loop-growth-v1",
            "input_tokens": estimated_input, "output_tokens": estimated_output,
            "total_tokens": estimated_input + estimated_output,
            "budget_units": estimated_input / 10 + estimated_output,
            "features": {"prompt_bytes": prompt_bytes, "file_count": len(task.get("files", [])),
                         "file_bytes": sum(p["bytes"] for p in task.get("files", []))},
            "coefficients": {k: getattr(config, k) for k in ("bytes_per_token", "expected_calls", "expected_output_tokens", "expected_observation_tokens", "message_overhead_tokens")},
            "calibration": calibration, "is_provider_usage": False,
            "uncertainty": "uncalibrated heuristic; not a quantile or a hard cap"}


def append_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def calibrate(records, config_id):
    ratios, run_ids, groups = [], [], set()
    for run in records:
        if run["config_id"] != config_id or run["backend"] != "real" or not run["actual"]["complete"]:
            continue
        predicted = run["prediction"]["budget_units"]
        # Always fit against unscaled heuristic, even when this run used an earlier calibration.
        prior_scale = (run["prediction"].get("calibration") or {}).get("scale", 1.0)
        ratios.append(run["actual"]["budget_units"] / (predicted / prior_scale))
        run_ids.append(run["run_id"])
        groups.add(run["task_id"])
    if not ratios or median(ratios) <= 0:
        raise ValueError("Need positive, complete REAL provider usage for this config")
    return {"version": 1, "config_id": config_id, "scale": median(ratios), "n": len(ratios),
            "task_groups": sorted(groups), "source_run_ids": run_ids, "created_at": now(),
            "warning": "Training statistic only; evaluate on held-out task families"}
