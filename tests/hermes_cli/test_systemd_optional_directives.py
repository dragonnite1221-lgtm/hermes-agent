"""Tests for systemd optional-directive normalization (issue #41119) and PATH
normalization (issue #35240 follow-up).

On older systemd versions that don't support RestartMaxDelaySec /
RestartSteps, the installed unit file has those directives silently
dropped.  Without normalization, systemd_unit_is_current() would
perpetually report the unit as outdated because the strict text
comparison sees a difference.

The fix: _strip_optional_systemd_directives() removes those directives
from both the installed and expected text before comparison.

Separately, generate_systemd_unit() bakes _build_wsl_interop_paths()'s
/mnt/... entries -- scraped straight from the invoking shell's live PATH,
only when is_wsl() -- into the unit's Environment="PATH=..." directive. Two
shells on the same WSL host routinely carry different /mnt/... segments
(per-Windows-session interop PATH), so re-running `hermes gateway
status`/`restart` from a different shell than whichever last wrote the unit
made a perfectly healthy install look outdated forever.
hermes_cli.gateway_service_staleness.normalize_systemd_unit_for_comparison()
drops only those /mnt/... entries, and only under is_wsl() -- never the whole
PATH payload (a moved/removed managed Node directory must still trigger a
refresh, #35240 review), and never on a non-WSL host (there /mnt/... entries
are never interop noise -- they're real mounts a managed Node install or a
mounted toolchain can legitimately live under, #16 review).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _strip_optional_systemd_directives
# ---------------------------------------------------------------------------


class TestStripOptionalSystemdDirectives:
    def test_removes_restart_max_delay_sec(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = """[Service]
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
"""
        result = _strip_optional_systemd_directives(text)
        assert "RestartMaxDelaySec" not in result
        assert "RestartSteps" not in result
        assert "Restart=always" in result
        assert "RestartSec=5" in result





    def test_full_unit_comparison(self):
        """Simulate the full stale-check flow with an older systemd unit."""
        from hermes_cli.gateway import (
            _normalize_service_definition,
            _strip_optional_systemd_directives,
        )
        # What the installed unit looks like on older systemd (directives stripped)
        installed = """[Unit]
Description=Hermes Gateway
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python -m hermes_cli.main gateway run
Restart=always
RestartSec=5
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=default.target
"""
        # What generate_systemd_unit produces (with the directives)
        expected = """[Unit]
Description=Hermes Gateway
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python -m hermes_cli.main gateway run
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=default.target
"""
        # Without normalization, they differ
        assert _normalize_service_definition(installed) != _normalize_service_definition(expected)

        # With optional-directive stripping, they match
        norm_installed = _normalize_service_definition(
            _strip_optional_systemd_directives(installed)
        )
        norm_expected = _normalize_service_definition(
            _strip_optional_systemd_directives(expected)
        )
        assert norm_installed == norm_expected


# ---------------------------------------------------------------------------
# systemd_unit_is_current integration
# ---------------------------------------------------------------------------


class TestSystemdUnitIsCurrent:
    def test_unit_without_fatal_config_restart_policy_is_not_current(
        self, tmp_path, monkeypatch,
    ):
        from hermes_cli import gateway as gw

        expected = """[Service]
Restart=always
RestartForceExitStatus=75
RestartPreventExitStatus=78
"""
        installed = expected.replace("RestartPreventExitStatus=78\n", "")
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: expected,
        )

        assert gw.systemd_unit_is_current(system=False) is False

    def test_unit_without_optional_directives_is_current(self, tmp_path, monkeypatch):
        """Installed unit missing RestartMaxDelaySec/RestartSteps should be
        considered current when the generated unit includes them."""
        from hermes_cli import gateway as gw

        installed = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed + "\nRestartMaxDelaySec=300\nRestartSteps=5\n",
        )

        assert gw.systemd_unit_is_current(system=False) is True


    def test_nonexistent_unit_is_not_current(self, tmp_path, monkeypatch):
        from hermes_cli import gateway as gw
        unit_file = tmp_path / "nonexistent.service"
        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        assert gw.systemd_unit_is_current(system=False) is False

    def test_unit_is_current_ignores_wsl_interop_drift_but_not_real_changes(
        self, tmp_path, monkeypatch,
    ):
        """On WSL, /mnt/... PATH drift from a different invoking shell is ignored (#35240
        follow-up); a real change -- elsewhere, or a non-/mnt/... PATH entry such as a moved
        managed Node install -- must still be caught (#16 review)."""
        from hermes_cli import gateway as gw

        monkeypatch.setattr("hermes_constants.is_wsl", lambda: True)
        unit_file = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)

        installed = (
            '[Service]\n'
            'ExecStart=/usr/bin/python -m hermes_cli.main gateway run\n'
            'Environment="PATH=/home/user/.hermes/node/bin:/usr/bin:/bin"\n'
        )
        unit_file.write_text(installed)

        # A different invoking shell regenerates the same deployment with different /mnt/...
        # interop noise -- everything that actually matters is unchanged.
        monkeypatch.setattr(
            gw, "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed.replace(
                "PATH=", "PATH=/mnt/c/some/other/tool:"
            ),
        )
        assert gw.systemd_unit_is_current(system=False) is True

        # The managed Node directory itself changed -- a real deployment change, not noise.
        monkeypatch.setattr(
            gw, "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed.replace(
                "/home/user/.hermes/node/bin", "/home/user/.hermes/node/v2/bin"
            ),
        )
        assert gw.systemd_unit_is_current(system=False) is False

        # ExecStart itself changed -- unrelated to PATH, must still be caught.
        monkeypatch.setattr(
            gw, "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed.replace(
                "gateway run", "gateway run --profile jarvis"
            ),
        )
        assert gw.systemd_unit_is_current(system=False) is False


# ---------------------------------------------------------------------------
# hermes_cli.gateway_service_staleness.normalize_systemd_unit_for_comparison
# ---------------------------------------------------------------------------


class TestNormalizeSystemdUnitForComparison:
    def test_masks_mnt_entries_only_under_is_wsl(self, monkeypatch):
        """The /mnt/... masking is WSL-interop-specific: on a non-WSL host, `_build_wsl_interop_paths()`
        never contributes anything, so a /mnt/... entry there is a real mount (e.g. a managed Node
        install or mounted toolchain) that must be compared verbatim, not masked away (#16 review)."""
        from hermes_cli.gateway_service_staleness import normalize_systemd_unit_for_comparison

        text = '[Service]\nEnvironment="PATH=/a:/mnt/c/windows/thing:/b"\n'

        monkeypatch.setattr("hermes_constants.is_wsl", lambda: True)
        assert "/mnt/c/windows/thing" not in normalize_systemd_unit_for_comparison(text)

        monkeypatch.setattr("hermes_constants.is_wsl", lambda: False)
        assert "/mnt/c/windows/thing" in normalize_systemd_unit_for_comparison(text)
