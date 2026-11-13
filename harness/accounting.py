"""Physical-request ledger. Rollups must never be inserted as calls."""
from __future__ import annotations

import math
from .core import append_json, now


class StopBudget(RuntimeError):
    pass


def request_reserve(body):
    """Admission headroom, NOT predicted spend or provider usage. Same rule at send and planning."""
    import json
    output_cap = count(body.get('max_tokens', body.get('max_completion_tokens')))
    if output_cap is None or output_cap <= 0 or body.get('stream'):
        raise StopBudget('Nonstreaming request with a positive output cap required')
    return len(json.dumps(body, ensure_ascii=False).encode()) / 10 + 1024 + output_cap


def initial_reserve(task, config, feedback=None):
    from .core import messages_for
    import json
    messages = messages_for(task, config)
    if feedback:
        messages.append({'role':'user','content':json.dumps(feedback,ensure_ascii=False)})
    return request_reserve({'model':config.model, 'messages':messages, 'temperature':config.temperature,
                            'max_tokens':config.max_output_tokens, 'stream':False})


def count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize(raw, reasoning_in_completion=True):
    if not isinstance(raw, dict):
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                "total_tokens": None, "budget_units": None, "complete": False}
    inp, out = count(raw.get("prompt_tokens")), count(raw.get("completion_tokens"))
    details = raw.get("completion_tokens_details") or {}
    reasoning = count(details.get("reasoning_tokens", raw.get("reasoning_tokens"))) if isinstance(details, dict) else None
    if not reasoning_in_completion:
        out = out + reasoning if out is not None and reasoning is not None else None
    complete = inp is not None and out is not None
    return {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": reasoning,
            "total_tokens": inp + out if complete else None,
            "budget_units": inp / 10 + out if complete else None, "complete": complete}


def aggregate(calls):
    # ID identifies a physical send, not provider response ID (cached IDs may repeat).
    unique = {}
    for c in calls:
        if c["kind"] != "physical_request":
            raise ValueError("Parent/rollup cannot enter the physical ledger")
        previous = unique.get(c["attempt_id"])
        if previous is not None and previous != c:
            raise ValueError("Conflicting duplicate attempt")
        unique[c["attempt_id"]] = c
    known = [c["usage"] for c in unique.values() if c["usage"]["complete"]]
    complete = len(known) == len(unique)
    return {"complete": complete, "physical_calls": len(unique),
            "unknown_calls": len(unique) - len(known),
            "known_budget_units": sum(u["budget_units"] for u in known),
            **{k: sum(u[k] for u in known) if complete else None
               for k in ("input_tokens", "output_tokens", "total_tokens", "budget_units")}}


class Ledger:
    def __init__(self, path, budget, max_physical_calls, reasoning_in_completion=True, *, clock=None):
        import time
        if not math.isfinite(budget) or budget <= 0 or max_physical_calls <= 0:
            raise ValueError("Positive budget/call limit required")
        self.path, self.budget, self.limit = path, budget, max_physical_calls
        self.calls = []
        self.reasoning_in_completion = reasoning_in_completion
        self.clock = clock or time.monotonic

    @property
    def remaining(self):
        actual = aggregate(self.calls)
        if not actual["complete"]:
            raise StopBudget("Unknown provider usage: reconciliation required")
        return self.budget - actual["budget_units"]

    def authorize(self, body):
        if getattr(self, "deadline_monotonic", None) is not None and self.clock() >= self.deadline_monotonic:
            raise StopBudget("Campaign wall deadline")
        if len(self.calls) >= self.limit:
            raise StopBudget("Physical request limit")
        if body.get("stream"):
            raise StopBudget("Streaming unsupported by this ledger")
        output_cap = count(body.get("max_tokens", body.get("max_completion_tokens")))
        if output_cap is None or output_cap <= 0:
            raise StopBudget("Output cap required")
        # Conservative local reserve, NOT a provider-confirmed token bound.
        reserve = request_reserve(body)
        if reserve > self.remaining:
            raise StopBudget("Insufficient reserve; proxy remains authoritative")
        return reserve

    def record(self, *, attempt_id, run_id, logical_call_id, raw_usage, status, provider_id=None, error_type=None,
               returned_model=None, system_fingerprint=None):
        record = {"kind": "physical_request", "attempt_id": attempt_id, "run_id": run_id,
                  "logical_call_id": logical_call_id, "created_at": now(), "provider_id": provider_id,
                  "status": status, "raw_usage": raw_usage,
                  "returned_model": returned_model, "system_fingerprint": system_fingerprint,
                  "usage": normalize(raw_usage, self.reasoning_in_completion),
                  "normalizer_version": "chat-completions-v1", "error_type": error_type,
                  "reasoning_in_completion": self.reasoning_in_completion}
        self.calls.append(record)
        append_json(self.path, record)
        return record


class ScopedLedger:
    """Per-plan allowance over the same physical journal; never resets global spending."""
    def __init__(self, parent, budget, wall_seconds):
        if not math.isfinite(budget) or budget <= 0 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError('Positive per-plan limits required')
        self.parent, self.budget = parent, budget
        self.initial_spend = parent.budget - parent.remaining
        self.deadline_monotonic = min(parent.deadline_monotonic, self.clock() + wall_seconds)

    @property
    def remaining(self):
        remaining = self.parent.remaining
        spent = self.parent.budget - remaining - self.initial_spend
        return min(remaining, self.budget - spent)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def authorize(self, body):
        if self.clock() >= self.deadline_monotonic:
            raise StopBudget('Per-plan wall deadline')
        if request_reserve(body) > self.remaining:
            raise StopBudget('Insufficient per-plan allowance')
        return self.parent.authorize(body)


def audit_events(events):
    started = {e["attempt_id"] for e in events if e["kind"] == "request_started"}
    settled = [e for e in events if e["kind"] == "physical_request"]
    settled_ids = {e["attempt_id"] for e in settled}
    totals = aggregate(settled)
    unfinished = sorted(started - settled_ids)
    if unfinished:
        totals.update(complete=False, budget_units=None, total_tokens=None, input_tokens=None, output_tokens=None)
    return {"unfinished_attempts": unfinished, "totals": totals,
            "warning": "Unfinished requests require proxy reconciliation; do not resume with reset budget"}
