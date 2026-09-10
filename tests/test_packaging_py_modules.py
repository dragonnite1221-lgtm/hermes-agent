"""Packaging invariant: every root-level module that packaged code imports ships in the wheel.

``packages.find`` only sees directories with ``__init__.py``; root single-file modules
(``run_agent``, ``hermes_state``, ``toolsets``...) reach the wheel through
``setup.py::_root_py_modules()``, which derives the list from the tree at build time. A
static list drifted every time the root layout changed and broke installed wheels with
``ModuleNotFoundError`` on ``import hermes_state``. This test pins the two halves of that
contract: the derived list covers every root module packaged code can import, and the
helper stays the single source (no static ``py-modules`` creeping back into pyproject).

Coverage is behavioral, not textual: it builds the real wheel via the same
``HERMES_NIX_BUILD=1`` marker path ``test_packaging_build_guard.py`` uses, then imports
every packaged module straight out of that artifact in an isolated subprocess. A root
module packaged code needs but the wheel didn't ship fails as the actual
``ModuleNotFoundError`` an installed user would hit, not a source-text guess.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("agent", "tools", "hermes_cli", "gateway", "tui_gateway", "cron", "acp_adapter", "plugins", "providers")

_IMPORT_WHEEL_PACKAGES_SCRIPT = """
import importlib
import pkgutil
import sys

wheel_path, packages_csv = sys.argv[1], sys.argv[2]
sys.path.insert(0, wheel_path)  # a .whl is a zip; CPython imports straight out of it
sys.argv = ["python"]  # some modules parse argv at import time (e.g. hermes-acp's entry point)

missing_root_modules = set()
other_errors = []
for pkg_name in packages_csv.split(","):
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as exc:  # the package itself failed to import
        other_errors.append(f"{pkg_name}: {exc!r}")
        continue
    if not hasattr(pkg, "__path__"):
        continue
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            # Only a MISSING SHIPPED FILE is this test's concern. A module
            # failing to import for an unrelated reason (an optional
            # third-party dependency not installed in this ad hoc
            # environment) is not a packaging bug and is not this test's job.
            if exc.name and "." not in exc.name:
                missing_root_modules.add(exc.name)
            else:
                other_errors.append(f"{info.name}: {exc!r}")
        except Exception as exc:
            other_errors.append(f"{info.name}: {exc!r}")

print("MISSING_ROOT_MODULES", sorted(missing_root_modules))
print("OTHER_ERRORS", other_errors)
"""


def _root_py_modules() -> set[str]:
    spec = importlib.util.spec_from_file_location("_hermes_setup_py", REPO_ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["setup.py", "--name"]  # setup() must not try to build anything on import
    try:
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sys.argv = saved
    return set(mod._root_py_modules())


def _build_wheel(tmp_path: Path) -> Path:
    """Invoke the real PEP 517 build_wheel hook, same allowed-marker path as
    ``test_packaging_build_guard.py``'s ``_build_artifact`` helper."""
    env = os.environ.copy()
    env["HERMES_NIX_BUILD"] = "1"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    result = subprocess.run(
        [sys.executable, "-c", f"from setuptools.build_meta import build_wheel; build_wheel(r'{tmp_path}')"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(tmp_path.glob("hermes_agent-*.whl"))
    assert wheels, f"build_wheel produced no artifact: {result.stdout}\n{result.stderr}"
    return wheels[0]


def test_pyproject_has_no_static_py_modules_list():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "py-modules" not in cfg["tool"]["setuptools"], (
        "root modules are derived in setup.py::_root_py_modules(); a static py-modules list drifts "
        "from the tree and breaks installed wheels. Do not add it back."
    )


def test_every_root_module_imported_by_packaged_code_is_shipped(tmp_path):
    shipped = _root_py_modules()
    assert "hermes_state" in shipped and "setup" not in shipped

    wheel_path = _build_wheel(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WHEEL_PACKAGES_SCRIPT, str(wheel_path), ",".join(PACKAGES)],
        cwd=tmp_path,  # not REPO_ROOT: importing must resolve only through the wheel on sys.path
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"import sweep crashed:\n{result.stdout}\n{result.stderr}"

    # ast.literal_eval, not eval: the subprocess prints repr() of plain
    # list/str literals, so this only ever needs to parse data, not execute it.
    missing: set[str] = set()
    other_errors: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("MISSING_ROOT_MODULES "):
            missing = set(ast.literal_eval(line[len("MISSING_ROOT_MODULES ") :]))
        elif line.startswith("OTHER_ERRORS "):
            other_errors = ast.literal_eval(line[len("OTHER_ERRORS ") :])

    assert not missing, (
        f"packaged code imports root modules the wheel would not ship: {sorted(missing)}\n"
        f"(unrelated import errors seen during the sweep, for context: {other_errors})"
    )
