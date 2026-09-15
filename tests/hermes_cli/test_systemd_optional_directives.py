"""Tests for systemd optional-directive normalization (issue #41119) and PATH
normalization (issue #35240 follow-up).

On older systemd versions that don't support RestartMaxDelaySec /
RestartSteps, the installed unit file has those directives silently
dropped.  Without normalization, systemd_unit_is_current() would
perpetually report the unit as outdated because the strict text
comparison sees a difference.

The fix: _strip_optional_systemd_directives() removes those directives
from both the installed and expected text before comparison.

Separately, generate_systemd_unit() bakes the invoking shell's PATH into
the unit's Environment="PATH=..." directive. Two shells on the same host
routinely carry different PATHs (WSL interop entries, per-tool installer
dirs, etc.), so re-running `hermes gateway status`/`restart` from a
different shell than whichever last wrote the unit made a perfectly
healthy install look outdated forever. _normalize_systemd_unit_for_comparison()
masks that one payload before comparing, mirroring the launchd twin
(_normalize_launchd_plist_for_comparison) which already does the same for
its <key>PATH</key> entry.
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

    def test_unit_differing_only_by_path_is_current(self, tmp_path, monkeypatch):
        """A unit installed from one shell, then checked from another shell with a
        different PATH, must still read as current -- see module docstring (#35240
        follow-up)."""
        from hermes_cli import gateway as gw

        installed = (
            '[Service]\n'
            'ExecStart=/usr/bin/python -m hermes_cli.main gateway run\n'
            'Environment="PATH=/home/user/.hermes/current/.venv/bin:/usr/bin:/bin"\n'
            'Environment="VIRTUAL_ENV=/home/user/.hermes/current/.venv"\n'
        )
        # A different invoking shell would regenerate the same unit with a
        # differently-ordered/differently-populated PATH -- everything else is identical.
        expected = (
            '[Service]\n'
            'ExecStart=/usr/bin/python -m hermes_cli.main gateway run\n'
            'Environment="PATH=/home/user/.hermes/current/.venv/bin:/mnt/c/some/other/tool:/usr/bin:/bin"\n'
            'Environment="VIRTUAL_ENV=/home/user/.hermes/current/.venv"\n'
        )
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(gw, "generate_systemd_unit", lambda system=False, run_as_user=None: expected)

        assert gw.systemd_unit_is_current(system=False) is True

    def test_unit_differing_by_more_than_path_is_still_stale(self, tmp_path, monkeypatch):
        """PATH is masked, but a real difference elsewhere must still be caught."""
        from hermes_cli import gateway as gw

        installed = (
            '[Service]\n'
            'ExecStart=/usr/bin/python -m hermes_cli.main gateway run\n'
            'Environment="PATH=/usr/bin:/bin"\n'
        )
        expected = (
            '[Service]\n'
            'ExecStart=/usr/bin/python -m hermes_cli.main gateway run --profile jarvis\n'
            'Environment="PATH=/mnt/c/some/other/tool:/usr/bin:/bin"\n'
        )
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(gw, "generate_systemd_unit", lambda system=False, run_as_user=None: expected)

        assert gw.systemd_unit_is_current(system=False) is False


# ---------------------------------------------------------------------------
# _normalize_systemd_unit_for_comparison
# ---------------------------------------------------------------------------


class TestNormalizeSystemdUnitForComparison:
    def test_masks_path_payload_only(self):
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison

        text = (
            '[Service]\n'
            'Environment="PATH=/a:/b:/c"\n'
            'Environment="VIRTUAL_ENV=/venv"\n'
        )
        result = _normalize_systemd_unit_for_comparison(text)
        assert "/a:/b:/c" not in result
        assert "__HERMES_PATH__" in result
        assert 'Environment="VIRTUAL_ENV=/venv"' in result

    def test_two_different_paths_normalize_identically(self):
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison

        a = '[Service]\nEnvironment="PATH=/a:/b"\n'
        b = '[Service]\nEnvironment="PATH=/x:/y:/z"\n'
        assert _normalize_systemd_unit_for_comparison(a) == _normalize_systemd_unit_for_comparison(b)
