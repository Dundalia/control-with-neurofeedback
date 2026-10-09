"""Writers of Experiment 1: who generates the next sentence.

The subject writes for itself with greedy decoding on the local vLLM engine
(:class:`SelfWriter`). Every other writer is called through an OpenAI-compatible API
(:class:`APIWriter`): OpenRouter, or a self-served endpoint for OLMo. The score is always
read on the subject's activations, whoever wrote the sentence.
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

from .models import get_model

REPO = Path(__file__).resolve().parents[1]
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _status_of(exc):
    for attr in ("status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    v = getattr(getattr(exc, "response", None), "status_code", None)
    return v if isinstance(v, int) else None


def is_transient_error(exc) -> bool:
    """Rate limits, server errors and network errors are retried; other 4xx errors (e.g. a
    content filter) are not."""
    status = _status_of(exc)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__.lower()
    if any(t in name for t in ("ratelimit", "timeout", "connection", "apiconnection", "internalserver")):
        return True
    if any(t in name for t in ("badrequest", "permission", "authentication", "notfound", "unprocessable")):
        return False
    return True


def _retry_after(exc, fallback: float) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        for k in ("retry-after", "Retry-After"):
            try:
                v = headers.get(k)
                if v is not None:
                    return max(float(v), fallback)
            except (TypeError, ValueError):
                pass
    return fallback


def clean_reply(text: str | None) -> str:
    """Strip leaked reasoning (a think block, GPT-OSS Harmony channels), surrounding quotes,
    and repeated whitespace from an API reply."""
    t = (text or "").strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    if "<|channel|>final<|message|>" in t:
        t = t.split("<|channel|>final<|message|>")[-1]
    if "<|" in t:
        t = t.split("<|")[0].strip()
    if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1].strip()
    return " ".join(t.split())


class SelfWriter:
    def __init__(self, backend, max_new_tokens: int = 64):
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def write_batch(self, conversations, seeds) -> list[str]:
        return self.backend.generate_batch(conversations, max_new_tokens=self.max_new_tokens,
                                           seeds=seeds)

    def describe(self) -> dict:
        return {"type": "self", "max_new_tokens": self.max_new_tokens, "temperature": 0.0}


class APIWriter:
    """Greedy chat completions from an OpenAI-compatible endpoint. Transient errors are
    retried with exponential backoff; the caller bounds the whole call with a timeout."""

    retries = 3
    retry_base = 2.0
    retry_cap = 60.0

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", max_tokens: int = 128,
                 reasoning: dict | None = None, chat_template_kwargs: dict | None = None,
                 extra_body: dict | None = None, max_concurrency: int = 8):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.reasoning = reasoning or None
        self.chat_template_kwargs = chat_template_kwargs or None
        self.extra_body = dict(extra_body) if extra_body else {}
        self.max_concurrency = max_concurrency
        self._client = None
        self._sema = None
        self._sema_loop = None

    def _aclient(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key or "EMPTY")
        return self._client

    def _semaphore(self):
        loop = asyncio.get_running_loop()
        if self._sema_loop is not loop:
            self._sema = asyncio.Semaphore(self.max_concurrency)
            self._sema_loop = loop
        return self._sema

    def _request(self, conversation) -> dict:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, temperature=0.0,
                      messages=[{"role": m["role"], "content": m["content"]} for m in conversation])
        extra = dict(self.extra_body)
        if self.reasoning is not None:
            extra["reasoning"] = self.reasoning
        if self.chat_template_kwargs:
            extra["chat_template_kwargs"] = self.chat_template_kwargs
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    async def write(self, conversation) -> tuple[str, dict]:
        """Returns (reply, info) where info has the OpenRouter provider, the token usage and
        the finish reason."""
        kwargs = self._request(conversation)
        last = None
        for attempt in range(self.retries):
            try:
                async with self._semaphore():
                    resp = await self._aclient().chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                if not is_transient_error(e):
                    raise
                if attempt < self.retries - 1:
                    delay = _retry_after(e, min(self.retry_cap, self.retry_base * 2.0 ** attempt))
                    await asyncio.sleep(delay * random.uniform(0.7, 1.3))
        else:
            raise RuntimeError(f"{self.model}@{self.base_url} failed after {self.retries} tries: {last}")
        info = {"provider": getattr(resp, "provider", None)}
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                info["completion_tokens"] = getattr(u, "completion_tokens", None)
            info["finish_reason"] = resp.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            pass
        return clean_reply(resp.choices[0].message.content), info

    def describe(self) -> dict:
        return {"type": "api", "model": self.model, "base_url": self.base_url,
                "max_tokens": self.max_tokens, "temperature": 0.0, "reasoning": self.reasoning,
                "chat_template_kwargs": self.chat_template_kwargs,
                "extra_body": self.extra_body or None}


def api_writer(model_key: str, max_tokens: int = 128) -> APIWriter:
    """The API writer of a model, as configured in :mod:`neurofeedback.models`."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    w = dict(get_model(model_key).writer)
    if not w:
        raise ValueError(f"{model_key} has no writer configuration")
    common = dict(max_tokens=max_tokens, reasoning=w.get("reasoning"),
                  chat_template_kwargs=w.get("chat_template_kwargs"), extra_body=w.get("extra_body"))
    if w["api"] == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (environment or .env)")
        return APIWriter(w["model"], OPENROUTER_URL, api_key=key, max_concurrency=8, **common)
    base_url = os.environ.get(w["base_url_env"], w["base_url_default"])
    return APIWriter(w["model"], base_url, api_key="EMPTY", max_concurrency=64, **common)
