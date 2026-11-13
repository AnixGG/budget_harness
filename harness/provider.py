"""Pinned openJiuwen adapter with instrumentation below its retry layers."""
from __future__ import annotations

import json
import os
from types import MethodType
from uuid import uuid4
from urllib.parse import urlsplit

from .accounting import StopBudget


class JiuwenProvider:
    def __init__(self, config, ledger, *, mock=False):
        from importlib.metadata import version
        if version("openjiuwen") != config.sdk:
            raise RuntimeError(f"Expected openjiuwen=={config.sdk}, installed {version('openjiuwen')}")
        import httpx
        from openai import AsyncOpenAI
        from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

        self.run_id = self.logical_call_id = None
        self.run_calls = 0
        self.stop_reason = None
        self.config, self.ledger = config, ledger
        self.pacer = None
        if mock:
            base, key = "https://mock.invalid/v1", "fixture-key-not-a-secret"
            inner = httpx.MockTransport(self._fixture)
        else:
            base, key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
            if not base or not key:
                raise ValueError("Set LLM_BASE_URL and LLM_API_KEY in environment")
            if not base.startswith("https://"):
                raise ValueError("Real proxy must use HTTPS")
            parsed_base = urlsplit(base)
            if parsed_base.username or parsed_base.password or parsed_base.query or parsed_base.fragment:
                raise ValueError("Proxy base URL must not contain credentials, query or fragment")
            inner = httpx.AsyncHTTPTransport(retries=0)
            from .rate_limit import RequestPacer
            self.pacer = RequestPacer(os.getenv("HARNESS_RATE_LIMIT_STATE", "/tmp/jiuwen-api-rate-limit.json"),
                                      int(os.getenv("HARNESS_REQUESTS_PER_MINUTE", "60")))

        provider = self

        class MeasuredTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                body = json.loads(request.content)
                try:
                    if provider.run_calls >= config.max_calls:
                        raise StopBudget("Per-task physical request limit")
                    ledger.authorize(body)
                    if provider.pacer is not None:
                        import time
                        deadline = getattr(ledger, "deadline_monotonic", None)
                        await provider.pacer.acquire(seconds_left=deadline-time.monotonic() if deadline is not None else None)
                        ledger.authorize(body)
                except StopBudget as exc:
                    provider.stop_reason = str(exc)
                    raise
                provider.run_calls += 1
                attempt_id = str(uuid4())
                # Persist before sending: a crash leaves a visibly unfinished attempt.
                from .core import append_json, now
                append_json(ledger.path, {"kind": "request_started", "attempt_id": attempt_id,
                            "run_id": provider.run_id, "logical_call_id": provider.logical_call_id,
                            "created_at": now(), "model": body.get("model"),
                            "max_tokens": body.get("max_tokens"), "backend": "mock" if mock else "real"})
                try:
                    response = await inner.handle_async_request(request)
                    await response.aread()
                    try:
                        payload = response.json()
                    except (ValueError, UnicodeError):
                        payload = {}
                    ledger.record(attempt_id=attempt_id, run_id=provider.run_id,
                                  logical_call_id=provider.logical_call_id,
                                  raw_usage=payload.get("usage"), status=response.status_code,
                                  provider_id=payload.get("id"), returned_model=payload.get("model"),
                                  system_fingerprint=payload.get("system_fingerprint"))
                    return response
                except BaseException as exc:
                    ledger.record(attempt_id=attempt_id, run_id=provider.run_id,
                                  logical_call_id=provider.logical_call_id,
                                  raw_usage=None, status=None, error_type=type(exc).__name__)
                    raise

            async def aclose(self):
                await inner.aclose()

        self.http = httpx.AsyncClient(transport=MeasuredTransport(), timeout=config.timeout_seconds,
                                     follow_redirects=False, trust_env=False)
        self.client = AsyncOpenAI(base_url=base, api_key=key, max_retries=0,
                                  timeout=config.timeout_seconds, http_client=self.http)
        client_config = ModelClientConfig(client_provider="OpenAI", api_base=base, api_key=key,
                                          verify_ssl=True, max_retries=0, timeout=config.timeout_seconds,
                                          use_shared_llm_http_client=True)
        self.model = Model(model_client_config=client_config,
                           model_config=ModelRequestConfig(model=config.model, temperature=config.temperature,
                                                           max_tokens=config.max_output_tokens))
        # SDK-specific seam, locked and covered by MockTransport integration test.
        self.model._client._create_async_openai_client = MethodType(
            lambda _client, timeout=None: self.client, self.model._client)

    @staticmethod
    def _fixture(request):
        """Protocol fixture, never an LLM or measured provider usage."""
        import httpx
        body = json.loads(request.content)
        messages = body["messages"]
        if len(messages) <= 2:
            action = {"action": "python", "code": "print(6 * 7)"}
        else:
            action = {"action": "answer", "answer": "42"}
        return httpx.Response(200, json={"id": "fixture-" + str(uuid4()), "object": "chat.completion",
                 "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(action)},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 320, "completion_tokens": 32, "total_tokens": 352,
                           "completion_tokens_details": {"reasoning_tokens": 8}}})

    async def invoke(self, messages, *, max_tokens=None):
        cap = self.config.max_output_tokens if max_tokens is None else max_tokens
        if not isinstance(cap, int) or isinstance(cap, bool) or not 0 < cap <= self.config.max_retry_output_tokens:
            raise ValueError("Output cap exceeds configured limit")
        self.logical_call_id = str(uuid4())
        response = await self.model.invoke(messages=messages, max_tokens=cap)
        return response.content, response.finish_reason

    async def close(self):
        await self.client.close()
