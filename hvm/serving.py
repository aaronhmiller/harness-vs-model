"""Runtime control for a one-model-at-a-time server (rMLX or mlx-lm).

Why this file exists
--------------------
The first version of this project assumed one server could serve all three model
sizes by naming a different `model` in each request. That is wrong for the
local-coding-agent setup, and dangerously so on a 16GB machine: rMLX is launched
with `--max-loaded-models 1` precisely so a second set of weights cannot arrive
while the first is still resident. Switching models is
`agent-model use <key> && agent-serve restart` — a restart, not a request field.

So the runner treats the model axis as an OUTER loop with a real cost: swap the
weights, wait for the port to answer, then run every cell for that model. The
commands are config, not hard-coded, so the same code drives rMLX, mlx-lm, or
anything else that ends up serving an OpenAI-compatible port.

rMLX and mlx-lm serve byte-identical APIs, so nothing above this layer knows or
cares which runtime is active.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time

import requests

from .http import proxies_for


class SwitchError(RuntimeError):
    pass


def _fmt(cmd: list[str], **kw) -> list[str]:
    return [part.format(**kw) for part in cmd]


def server_models(base_url: str, timeout_s: float = 5.0) -> list[str]:
    """Ask the server what it currently has loaded via /v1/models."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s,
                            proxies=proxies_for(base_url))
        resp.raise_for_status()
        return [m.get("id", "") for m in resp.json().get("data", [])]
    except Exception:
        return []


def wait_ready(base_url: str, timeout_s: float = 180.0, poll_s: float = 2.0,
               verbose: bool = True) -> bool:
    """Poll /v1/models until the server answers.

    A cold start has to read several GB off disk, so the default timeout is
    generous. Weights loading is not a hang.
    """
    deadline = time.time() + timeout_s
    first = True
    while time.time() < deadline:
        if server_models(base_url):
            if verbose and not first:
                print(" ready", file=sys.stderr)
            return True
        if verbose and first:
            print("  waiting for the server to load weights", end="", file=sys.stderr)
            first = False
        elif verbose:
            print(".", end="", file=sys.stderr, flush=True)
        time.sleep(poll_s)
    if verbose:
        print(" TIMEOUT", file=sys.stderr)
    return False


class RuntimeSwitcher:
    """Swaps the resident model between cells, or does nothing if disabled."""

    def __init__(self, cfg: dict | None, base_url: str, enabled: bool = True) -> None:
        cfg = cfg or {}
        self.use_cmd: list[str] = cfg.get("use", [])
        self.restart_cmd: list[str] = cfg.get("restart", [])
        self.ready_timeout_s: float = cfg.get("ready_timeout_s", 180)
        self.base_url = base_url
        self.enabled = enabled and bool(self.use_cmd or self.restart_cmd)
        self._active: str | None = None

    def available(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "switching disabled; serve each model yourself"
        exe = (self.use_cmd or self.restart_cmd)[0]
        if shutil.which(exe) is None:
            return False, (
                f"{exe!r} is not on PATH. Either add ~/.local/bin (local-coding-agent's "
                f"install.sh prints the line) or pass --no-switch and serve one model "
                f"at a time yourself."
            )
        return True, f"{exe} found"

    def ensure(self, key: str, registry_key: str, verbose: bool = True) -> None:
        """Make `registry_key` the resident model. Idempotent within a run."""
        if not self.enabled or self._active == key:
            return
        if verbose:
            print(f"\n[switch] loading {key} ({registry_key})", file=sys.stderr)

        for cmd in (self.use_cmd, self.restart_cmd):
            if not cmd:
                continue
            full = _fmt(cmd, key=registry_key)
            proc = subprocess.run(full, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SwitchError(
                    f"`{' '.join(full)}` failed ({proc.returncode}):\n"
                    f"{(proc.stderr or proc.stdout)[-800:]}"
                )

        if not wait_ready(self.base_url, self.ready_timeout_s, verbose=verbose):
            raise SwitchError(
                f"server did not answer {self.base_url}/models within "
                f"{self.ready_timeout_s:.0f}s after loading {registry_key}. "
                f"Check `agent-serve status` and `agent-serve logs`."
            )
        self._active = key
