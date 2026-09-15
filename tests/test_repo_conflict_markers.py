"""Guard against unresolved git merge-conflict markers landing on ``main``.

``_validate_critical_files_syntax`` (see ``test_update_post_pull_syntax_guard.py``,
PR #28452) already catches orphan conflict markers in the small set of Python
files ``hermes`` imports at CLI startup -- a marker there makes the CLI
unbootable, so it fails loudly and immediately. But that guard only looks at
that fixed file list, and only checks Python syntax validity. A marker left
in anything else (docs, configs, non-critical modules) imports fine and ships
silently: PR #11's 7085-commit upstream merge left a bare ``>>>>>>>`` line in
``website/docs/reference/optional-skills-catalog.md`` that rendered as broken
Markdown in the published skills catalog for days before a Codex post-hoc
review caught it.

This test closes that gap with a repo-wide sweep: any tracked file with a
line starting with ``<<<<<<<`` or ``>>>>>>>`` is almost certainly an orphan
conflict marker, on `main` or not. (The equivalent ``=======`` marker is
intentionally NOT checked here -- it is indistinguishable from an ordinary
Markdown/reST section-underline, which several docstrings and docs pages use
legitimately; the two distinctive markers below are enough to catch a botched
merge without false-flagging every underlined heading in the repo.)
"""

from __future__ import annotations

import subprocess

from hermes_cli.gateway import PROJECT_ROOT

_CONFLICT_MARKER_PATTERN = r"^(<<<<<<<|>>>>>>>)"


def test_no_unresolved_merge_conflict_markers_in_tracked_files():
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", _CONFLICT_MARKER_PATTERN, "--", "."],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when it finds nothing to report -- that's the success case here.
    assert result.returncode in (0, 1), (
        f"git grep failed unexpectedly (rc={result.returncode}): {result.stderr}"
    )
    assert result.stdout == "", (
        "Found what looks like an unresolved merge-conflict marker in a tracked "
        f"file -- resolve the merge properly instead of committing the marker:\n{result.stdout}"
    )
