"""Client for a local OpenAI-compatible endpoint.

Runtime-neutral on purpose. rMLX (a single Rust binary, no Python in the serving
path) and mlx_lm.server expose the same `/v1/chat/completions` on the same port,
so this file does not know or care which one is running — exactly the reason
local-coding-agent treats the runtime as a registry field rather than a rewrite:

    agent-serve runtime            # which one is active
    agent-serve runtime mlx-lm     # fall back
    agent-serve restart

Note this experiment never asks for `tool_calls`. It only needs text
completions, so the tool-call gate that decides whether a model can drive
OpenCode or Pi does not apply here — a model that narrates instead of emitting
structured calls is still perfectly usable as a subject.
"""
from __future__ import annotations

import re
import sys
import time

import requests

from ..http import proxies_for
from ..types import Completion, Message

# Matches a parameter-count token: 1.5B, 7b, 14 B. Excludes quantisation
# suffixes, so "4bit" in Qwen3-8B-4bit is not read as a 4B model.
SIZE_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)


def size_tokens(model_id: str) -> set[str]:
    """Parameter counts mentioned in a model id, normalised ('7.0' -> '7')."""
    out = set()
    for raw in SIZE_TOKEN.findall(model_id or ""):
        f = float(raw)
        out.add(str(int(f)) if f == int(f) else str(f))
    return out


class ModelMismatch(RuntimeError):
    pass


class MLXProvider:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8080/v1",
        timeout_s: float = 600.0,
        name: str | None = None,
        registry_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = name or model
        # The key `agent-model use <key>` expects, which is usually shorter than
        # the Hugging Face repo id.
        self.registry_key = registry_key or model
        # What the server actually answers to. Filled in by resolve().
        self.served_model = model

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=10,
                             proxies=proxies_for(self.base_url))
            r.raise_for_status()
            return [m.get("id", "") for m in r.json().get("data", [])]
        except Exception:
            return []

    def resolve(self, strict: bool = True, verbose: bool = True) -> str:
        """Reconcile the configured model id with what the server answers to.

        Servers report whatever id they were launched with, which is often NOT
        the id you configured: rMLX here reports `Qwen/Qwen2.5-1.5B-Instruct`
        while the config names the MLX conversion
        `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. Sending the configured id
        gets a 404 from /v1/chat/completions even though the server is healthy
        and the right weights are loaded.

        Since exactly one model is resident, adopting the served id is the right
        repair -- but ONLY after checking the parameter counts agree. Silently
        adopting a mismatched id would record 3B results under the 1.5B key and
        quietly destroy the model axis, which is the one failure this experiment
        cannot survive.
        """
        available = self.list_models()
        if not available:
            raise ModelMismatch(
                f"{self.base_url}/models returned nothing. Is the server up? "
                f"Try `agent-serve status`."
            )
        if self.model in available:
            self.served_model = self.model
            return self.served_model

        if len(available) != 1:
            raise ModelMismatch(
                f"configured model {self.model!r} is not served, and the server "
                f"offers {len(available)} alternatives {available} -- cannot pick "
                f"unambiguously. Set `model` in config.json to one of them."
            )

        served = available[0]
        want, got = size_tokens(self.model), size_tokens(served)
        if want and got and not (want & got):
            raise ModelMismatch(
                f"REFUSING TO RUN: config wants {self.model!r} (~{'/'.join(sorted(want))}B) "
                f"but the server is serving {served!r} (~{'/'.join(sorted(got))}B).\n"
                f"Running anyway would file the wrong model's results under "
                f"{self.name!r} and invalidate the model axis.\n"
                f"Fix with: agent-model use {self.registry_key} && agent-serve restart"
            )
        if verbose:
            print(
                f"  [resolve] {self.name}: config says {self.model!r}, server "
                f"answers to {served!r} — parameter counts agree, using the "
                f"server's id.",
                file=sys.stderr,
            )
        self.served_model = served
        return served

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> Completion:
        body = {
            "model": self.served_model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s,
            proxies=proxies_for(self.base_url),
        )
        resp.raise_for_status()
        data = resp.json()
        dt = time.time() - t0

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=dt,
        )

    def healthcheck(self) -> str:
        c = self.generate(
            [Message("user", "Reply with the single word: ok")],
            temperature=0.0,
            max_tokens=8,
            seed=0,
        )
        return c.text.strip()
