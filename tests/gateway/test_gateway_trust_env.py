"""gateway.trust_env — one config key controls aiohttp proxy-env honoring at every adapter site (#48820)."""
import re
from pathlib import Path

import pytest

from gateway.platforms import base as gw_base

REPO = Path(__file__).resolve().parents[2]
_ADAPTER_FILES = sorted(
    list((REPO / "gateway" / "platforms").rglob("*.py"))
    + list((REPO / "plugins" / "platforms").rglob("*.py"))
)


def _write_config(tmp_path, monkeypatch, body: str) -> None:
    # load_config caches on (path, mtime) — a fresh tmp HERMES_HOME per test is a fresh cache key.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(body)


@pytest.mark.parametrize(
    "yaml_body, expected",
    [("gateway:\n  trust_env: false\n", False), ("gateway:\n  trust_env: true\n", True), ("{}\n", True)],
)
def test_gateway_trust_env_reads_config(tmp_path, monkeypatch, yaml_body, expected):
    """gateway.trust_env in config.yaml drives the shared helper; absent → True (default)."""
    _write_config(tmp_path, monkeypatch, yaml_body)
    assert gw_base.gateway_trust_env() is expected
    # The generic-proxy discovery path is gated by the same knob; explicit per-platform vars are not.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert (gw_base.resolve_proxy_url() is not None) is expected
    monkeypatch.setenv("X_PLATFORM_PROXY", "http://127.0.0.1:1080")
    assert gw_base.resolve_proxy_url("X_PLATFORM_PROXY") == "http://127.0.0.1:1080"


def test_no_bare_trust_env_literal_in_adapters():
    """Every aiohttp session in gateway/ + plugins/platforms/ must go through gateway_trust_env()."""
    bare = re.compile(r"trust_env\s*=\s*(True|False)\b")
    constructor = re.compile(r"\b(aiohttp\.ClientSession|httpx\.(?:Async)?Client)\s*\(")
    offenders = []
    for path in _ADAPTER_FILES:
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, 1):
            if not bare.search(line):
                continue
            # trust_env is a constructor kwarg; the call can open several
            # lines above it in a multi-line
            # aiohttp.ClientSession(...)/httpx.Client(...), so a same-line
            # "httpx" check misses e.g. a2a/security.py's trust_env=False on
            # its own line inside a wrapped httpx.Client(...). Scan backward
            # (from this line, since a single-line call opens and sets the
            # kwarg on the same line) for the nearest opening constructor to
            # learn which library actually owns this literal.
            owner = None
            for prev in range(lineno, max(lineno - 25, 0), -1):
                match = constructor.search(lines[prev - 1])
                if match:
                    owner = match.group(1)
                    break
            if owner is None or owner.startswith("aiohttp"):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "hard-coded aiohttp trust_env literal(s); use gateway_trust_env():\n" + "\n".join(offenders)
