"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )


@pytest.fixture
def isolate_update_host_runtime(monkeypatch):
    """Prevent updater unit tests from discovering live host runtimes.

    Successful update paths purge cached Hermes modules before lazily importing
    gateway helpers. Without disabling that purge, per-test gateway mocks are
    evicted and the later restart phase can rediscover and signal real Hermes
    processes on a developer machine. Keep discovery and the final fleet probe
    hermetic for tests that opt into this fixture.
    """
    from hermes_cli import gateway, main, update_receipt

    monkeypatch.setattr(main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda *args, **kwargs: [])
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(
        gateway, "find_profile_gateway_processes", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(gateway, "_get_service_pids", lambda *args, **kwargs: set())
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions", lambda *args, **kwargs: []
    )
