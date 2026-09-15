"""WSL-interop-aware PATH normalization for the systemd unit staleness check.

Split out of ``hermes_cli.gateway`` per AGENTS.md's god-file rule (that facade
is already ~6,500 lines) instead of appending new behaviour to it -- see
``hermes_cli.gateway.systemd_unit_is_current``, the sole caller.
"""

from __future__ import annotations

import re


def normalize_systemd_unit_for_comparison(text: str, *, is_wsl: bool, home_under_mnt: bool) -> str:
    """Normalize unit text for staleness checks, dropping only the WSL-interop entries from the PATH
    payload: ``_build_wsl_interop_paths()`` scrapes ``/mnt/...`` entries straight out of the invoking
    shell's live ``PATH`` (plus ``shutil.which()`` hits for powershell.exe/cmd.exe/etc., which resolve
    under the same ``/mnt/...`` prefix) only when running on WSL, so those entries genuinely differ
    across Windows sessions and would flag a perfectly current unit as outdated forever.

    Both flags are passed in as plain data by the caller (``hermes_cli.gateway``'s real ``is_wsl()``
    and ``Path.home()`` checks) rather than queried here, so this stays a pure, host-independent
    function -- AGENTS.md's "don't fake the host OS" rule: a test exercises every branch with a bool,
    never by making the interpreter believe it's on another platform.

    On a non-WSL host ``_build_wsl_interop_paths()`` never contributes anything, so any ``/mnt/...``
    entry there comes from a real source (a managed Node install, a mounted toolchain resolved via
    ``shutil.which()``) and must still be compared verbatim -- masking it there would let a genuine
    change go unrepaired by ``gateway start``/``restart``/``install``. The same is true on WSL when
    ``$HOME`` itself lives under ``/mnt/...`` (e.g. a checkout at ``/mnt/c/project``): every
    service-managed path -- the venv, a managed Node install, ``~/.local/bin`` -- would then ALSO sit
    under ``/mnt/...`` and be indistinguishable from interop noise by prefix alone, so masking is
    skipped there too rather than risk hiding a real change (#16 review). Masking only ever applies
    when ``is_wsl`` is true AND ``home_under_mnt`` is false; every other PATH entry is always compared
    verbatim regardless.
    """
    from hermes_cli.gateway import _normalize_service_definition

    normalized = _normalize_service_definition(text)
    if not is_wsl or home_under_mnt:
        return normalized

    def _drop_wsl_interop_entries(match: "re.Match[str]") -> str:
        prefix, path_value, suffix = match.group(1), match.group(2), match.group(3)
        kept = [entry for entry in path_value.split(":") if not entry.startswith("/mnt/")]
        return f"{prefix}{':'.join(kept)}{suffix}"

    return re.sub(r'(Environment="PATH=)(.*?)(")', _drop_wsl_interop_entries, normalized)
