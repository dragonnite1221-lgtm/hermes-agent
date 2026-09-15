"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    build_edit_proposal,
    set_edit_approval_requester,
    should_auto_approve_edit,
)
from model_tools import handle_function_call


def teardown_function() -> None:
    set_edit_approval_requester(None)


def _fake_walkup_exec(command: str, overrides: dict[str, str] | None = None):
    """Test double for a non-host backend's ``_exec``, matching
    ``tools.file_tools._verify_realpath_within_any``'s combined walk-up
    script: it resolves the target AND every boundary in ONE round-trip,
    each tagged (``T`` for the target, ``B0``/``B1``/... for the
    boundaries, in order) and emitted as ``TAG\\t<value>`` lines.

    By default every requested path "resolves" to itself (no symlink
    involved) -- pass ``overrides`` (``{requested_path: real_path}``) to
    simulate a symlink for a specific one (typically the target, to
    exercise the escape-detection path; a boundary override exercises the
    boundary-is-itself-a-symlink case).
    """
    import re

    from tools.file_operations import ExecuteResult

    overrides = overrides or {}
    tags = re.findall(r"printf '(\w+)\\t", command)
    requested = re.findall(r"p='([^']*)';", command)
    lines = [f"{tag}\t{overrides.get(raw, raw)}" for tag, raw in zip(tags, requested)]
    return ExecuteResult(stdout="\n".join(lines) + "\n", exit_code=0)


def test_acp_permission_tool_call_uses_edit_kind_and_diff_content():
    proposal = EditProposal(
        tool_name="write_file",
        path="demo.txt",
        old_text="old\n",
        new_text="new\n",
        arguments={"path": "demo.txt", "content": "new\n"},
    )

    tool_call = build_acp_edit_tool_call(proposal)

    assert tool_call.kind == "edit"
    assert tool_call.status == "pending"
    assert tool_call.rawInput == {"tool": "write_file", "arguments": proposal.arguments}
    assert len(tool_call.content) == 1
    diff = tool_call.content[0]
    assert diff.path == "demo.txt"
    assert diff.oldText == "old\n"
    assert diff.newText == "new\n"








def test_requester_exception_denies_and_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    def boom(_proposal):
        raise RuntimeError("zed disconnected")

    set_edit_approval_requester(boom)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-exception",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_patch_replace_rejection_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "beta\n",
                "new_string": "gamma\n",
            },
            task_id="acp-patch-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"








def test_workspace_auto_approval_allows_workspace_and_tmp_but_not_sensitive(tmp_path):
    workspace_file = tmp_path / "src.py"
    # Use tempfile.gettempdir() so this test exercises the same code path on
    # Linux (`/tmp`), macOS (`/private/var/folders/...`) and Windows
    # (`%LOCALAPPDATA%\Temp`). Before the fix this branch only worked on Linux.
    tmp_file = Path(tempfile.gettempdir()) / "hermes-acp-auto-approve-test.txt"
    env_file = tmp_path / ".env"

    assert should_auto_approve_edit(
        EditProposal("write_file", str(workspace_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert should_auto_approve_edit(
        EditProposal("write_file", str(tmp_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert not should_auto_approve_edit(
        EditProposal("write_file", str(env_file), None, "SECRET=x", {}),
        "session",
        str(tmp_path),
    )


def test_multi_target_v4a_patch_with_outside_path_is_not_auto_approved(tmp_path, monkeypatch):
    # Keep the global-tmp exemption out of the way so only the workspace
    # boundary check matters for this test.
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "a.txt"
    inside.write_text("inside\n", encoding="utf-8")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "b.txt"
    outside.write_text("outside\n", encoding="utf-8")

    # A V4A patch touching an in-workspace file first and an out-of-workspace
    # file second. The display string joins both paths with ", "; a decision
    # based on that joined string (instead of the real, individually resolved
    # targets) can misclassify the whole patch as workspace-local.
    patch_body = (
        f"*** Update File: {inside}\n"
        "@@\n"
        "-inside\n"
        "+inside changed\n"
        f"*** Update File: {outside}\n"
        "@@\n"
        "-outside\n"
        "+outside changed\n"
    )
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body})

    assert should_auto_approve_edit(proposal, "workspace_session", str(workspace)) is False


def test_preview_expands_tilde_using_the_writers_profile_home(tmp_path, monkeypatch):
    """A leading ``~`` must expand through the same profile-aware resolver
    the real write uses (tools.file_tools._expand_tilde ->
    get_subprocess_home()), not this ACP process's raw ``$HOME``
    (``Path.expanduser()``).

    Expanding ``~`` locally before deciding whether the path is absolute
    made every ``~/...`` path skip ``_resolve_path_for_task`` entirely
    (an expanded tilde path is always absolute), so a gateway/cron context
    where the profile home differs from the raw process ``$HOME`` could
    preview one file while the real write -- which always goes through
    ``_resolve_path_for_task`` -- targets another.
    """
    from tools.file_operations import ReadResult

    profile_home = tmp_path / "profile-home"
    os_home = tmp_path / "os-home"

    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setattr(
        "tools.file_tools.get_subprocess_home", lambda: str(profile_home), raising=False
    )
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda: str(profile_home))

    captured_paths = []

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            captured_paths.append(path)
            return ReadResult(content="irrelevant\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    build_edit_proposal(
        "write_file",
        {"path": "~/notes.txt", "content": "new content\n"},
        task_id="some-task",
    )

    assert captured_paths == [str(profile_home / "notes.txt")]


def test_preview_reads_relative_path_from_registered_session_cwd_not_process_cwd(
    tmp_path, monkeypatch
):
    # ACP is a long-lived server: its OS process cwd is wherever it was
    # launched from, not the per-session project directory. The actual
    # write engine (tools.file_tools.write_file_tool / patch) resolves a
    # relative tool-call path via tools.file_tools._resolve_path_for_task,
    # which falls back to the task's registered cwd override (what
    # acp_adapter/session.py's _register_task_cwd registers at session
    # create/load/resume) when there is no live-tracked terminal cwd yet.
    # The approval preview must resolve the SAME relative path through the
    # SAME resolver, or the diff shown to the user can come from a
    # completely different file than the one that is about to be written.
    import tools.terminal_tool as terminal_tool

    task_id = "preview-cwd-test-session"

    process_cwd = tmp_path / "server-launch-dir"
    process_cwd.mkdir()
    # A same-named file sits under the wrong (process) cwd with different
    # content, so a resolution bug reads this instead of the real target.
    (process_cwd / "relative.txt").write_text("WRONG FILE\n", encoding="utf-8")
    monkeypatch.chdir(process_cwd)

    session_cwd = tmp_path / "session-project-dir"
    session_cwd.mkdir()
    (session_cwd / "relative.txt").write_text("correct existing content\n", encoding="utf-8")

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(session_cwd)})
    try:
        proposal = build_edit_proposal(
            "write_file",
            {"path": "relative.txt", "content": "new content\n"},
            task_id=task_id,
        )
    finally:
        terminal_tool.clear_task_env_overrides(task_id)

    assert proposal.old_text == "correct existing content\n"


def test_preview_follows_live_terminal_cwd_after_cd(tmp_path, monkeypatch):
    """A mid-session `cd` must move the preview along with the real write.

    tools.file_tools._resolve_path_for_task treats the terminal's *live*
    tracked cwd (tools.terminal_tool.record_session_cwd -- updated on every
    completed terminal command) as more authoritative than the cwd the
    session was originally registered with. A preview resolver that only
    ever consulted the original registered/session cwd (and never the live
    tracked one) would keep diffing the session's ORIGINAL directory after
    the agent cd'd elsewhere, while the real write followed the terminal to
    its new location -- reopening the exact "preview shows a different file
    than the one written" bug one layer down.
    """
    import tools.terminal_tool as terminal_tool

    task_id = "preview-live-cwd-test-session"

    original_cwd = tmp_path / "original-session-dir"
    original_cwd.mkdir()
    (original_cwd / "relative.txt").write_text("STALE ORIGINAL DIR\n", encoding="utf-8")

    live_cwd = tmp_path / "worktree-after-cd"
    live_cwd.mkdir()
    (live_cwd / "relative.txt").write_text("current live directory\n", encoding="utf-8")

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(original_cwd)})
    terminal_tool.record_session_cwd(task_id, str(live_cwd))
    try:
        proposal = build_edit_proposal(
            "write_file",
            {"path": "relative.txt", "content": "new content\n"},
            task_id=task_id,
        )
    finally:
        terminal_tool.clear_task_env_overrides(task_id)
        terminal_tool.clear_session_cwd(task_id)

    assert proposal.old_text == "current live directory\n"


def test_preview_reads_through_the_selected_file_backend_not_local_fs(tmp_path, monkeypatch):
    """The preview must read via tools.file_tools._get_file_ops(task_id) --
    the same ShellFileOperations backend write_file_tool/patch_tool use --
    not a direct local filesystem read.

    For an SSH/container/sandbox-backed task, the resolved path lives in
    THAT backend's namespace, which is meaningless to read directly off
    this process's local filesystem: a local read could wrongly report a
    remote-only path as "does not exist", or worse, silently read an
    unrelated file that happens to exist at the same-looking local path on
    the ACP server's own host.
    """
    from tools.file_operations import ReadResult

    # A file exists locally at the resolved path with DIFFERENT content
    # than what the "remote" backend reports. A resolution bug that reads
    # the local filesystem directly would return this WRONG content.
    local_decoy = tmp_path / "remote-looking-path.txt"
    local_decoy.write_text("WRONG: local host file\n", encoding="utf-8")

    class FakeRemoteBackend:
        def read_file_raw(self, path, **kwargs):
            assert path == str(local_decoy)
            return ReadResult(content="correct remote content\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeRemoteBackend()
    )

    proposal = build_edit_proposal(
        "write_file",
        {"path": str(local_decoy), "content": "new content\n"},
        task_id="remote-task",
    )

    assert proposal.old_text == "correct remote content\n"


def test_preview_denies_instead_of_masking_unreadable_existing_file_as_new(monkeypatch):
    """A backend read error for an EXISTING file must not be shown as "no
    prior content" (i.e. as if write_file were creating a brand-new file).

    ShellFileOperations.read_file_raw() sets ``.error`` for permission
    failures, non-regular files, binary content, and transport failures --
    not only for a genuinely missing path. Collapsing all of those to "no
    old text" would let an approved write silently overwrite content the
    user was never shown a diff of. Only the confirmed-missing case (the
    exact "File not found: " prefix _suggest_similar_files always uses)
    may return None; anything else must raise so the proposal fails and
    maybe_require_edit_approval denies by default.
    """
    from tools.file_operations import ReadResult

    class FakeBackendPermissionDenied:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(error="Cannot read '/etc/shadow': Permission denied")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops",
        lambda task_id="default": FakeBackendPermissionDenied(),
    )

    import pytest

    with pytest.raises(OSError):
        build_edit_proposal(
            "write_file",
            {"path": "/etc/shadow", "content": "new content\n"},
            task_id="some-task",
        )


def test_preview_treats_confirmed_not_found_as_new_file(monkeypatch):
    """The one error shape that DOES mean "no old text": a confirmed
    missing path, signalled by the exact "File not found: " prefix that
    ShellFileOperations._suggest_similar_files always uses.
    """
    from tools.file_operations import ReadResult

    class FakeBackendMissing:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(error=f"File not found: {path}", similar_files=[])

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackendMissing()
    )

    proposal = build_edit_proposal(
        "write_file",
        {"path": "brand/new/file.txt", "content": "new content\n"},
        task_id="some-task",
    )

    assert proposal.old_text is None


def test_preview_denies_existing_image_instead_of_showing_it_as_empty(monkeypatch):
    """An existing image file must not be previewed as an empty/new file.

    ShellFileOperations.read_file_raw() reports an image with is_image=True,
    is_binary=True, and NO .error at all -- content stays at its default
    "". Checking only .error would fall through and show old_text="",
    making an approver think write_file is creating a brand-new file when
    it is actually about to overwrite an existing image.
    """
    from tools.file_operations import ReadResult

    class FakeBackendImage:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(is_image=True, is_binary=True, file_size=12345)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackendImage()
    )

    import pytest

    with pytest.raises(OSError):
        build_edit_proposal(
            "write_file",
            {"path": "photo.png", "content": "not actually an image"},
            task_id="some-task",
        )




def test_multi_target_v4a_patch_with_no_space_header_is_not_auto_approved(tmp_path, monkeypatch):
    # tools/patch_parser.py's real executor matches headers with `\s*` after
    # `***` (zero or more spaces), so a no-space header like
    # "***Update File:" still runs. Target extraction must use the exact
    # same parser, or a stricter regex here (e.g. requiring `\s+`) would
    # silently drop a no-space-header target from `target_paths` while the
    # patch still executes against it -- reopening the same auto-approval
    # bypass this module exists to close.
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "a.txt"
    inside.write_text("inside\n", encoding="utf-8")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "b.txt"
    outside.write_text("outside\n", encoding="utf-8")

    patch_body = (
        f"***Update File: {inside}\n"
        "@@\n"
        "-inside\n"
        "+inside changed\n"
        f"***Update File: {outside}\n"
        "@@\n"
        "-outside\n"
        "+outside changed\n"
    )
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body})

    assert proposal.target_paths is not None
    assert str(outside) in proposal.target_paths
    assert should_auto_approve_edit(proposal, "workspace_session", str(workspace)) is False


def test_v4a_delete_only_preview_allows_deleting_an_image(monkeypatch):
    """A V4A delete-only patch targeting an image must not be denied by the
    preview's binary/image guard (added for #5's "don't mask an existing
    image as an empty new file ahead of a content-overwriting write"
    protection -- see test_preview_denies_existing_image_instead_of_showing_it_as_empty).

    The real V4A executor's delete path (tools/patch_parser.py's
    _apply_delete) already deletes binary/image targets fine -- it only
    checks read_file_raw()'s .error, never .is_binary/.is_image -- so
    requiring a successful TEXT preview before a pure delete can run would
    deny a delete the real executor allows.
    """
    from tools.file_operations import ReadResult

    class FakeBackendImage:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(is_image=True, is_binary=True, file_size=54321)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackendImage()
    )

    patch_body = "*** Delete File: photo.png\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id="some-task")

    assert proposal is not None
    assert proposal.old_text is not None
    assert "image" in proposal.old_text
    assert "54321" in proposal.old_text


def test_v4a_update_on_image_still_denied(monkeypatch):
    """The delete-only relaxation above must not weaken #5's protection for
    an operation that DOES overwrite content: a single-target V4A UPDATE
    against an existing image must still deny, exactly like
    test_preview_denies_existing_image_instead_of_showing_it_as_empty does
    for write_file.
    """
    from tools.file_operations import ReadResult

    class FakeBackendImage:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(is_image=True, is_binary=True, file_size=54321)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackendImage()
    )

    patch_body = "*** Update File: photo.png\n@@\n-old\n+new\n"

    import pytest

    with pytest.raises(OSError):
        build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id="some-task")


def test_patch_replace_preview_reads_unstripped_content_like_the_real_write(monkeypatch):
    """patch_replace()'s real write (tools/file_operations.py) reads OLD
    content via a bare shell ``cat`` with no ``_strip_terminal_fence_leaks``
    cleanup applied -- unlike the default (stripped) ``read_file_raw()``
    every other proposal kind uses. If the "replace" patch-mode preview read
    the default stripped content, a file whose real bytes happen to match
    the stripper's regexes (an OSC escape sequence, or a
    ``__HERMES_FENCE_<id>__``-looking marker) would preview -- and fuzzy-match
    old_string against -- a laundered version of the file, computing a
    new_text that diverges from what patch_replace() actually produces once
    approved.
    """
    from tools.file_operations import ReadResult

    # Content containing a literal substring the fence-marker regex treats
    # as leaked terminal wrapper noise -- nothing in the codebase emits this
    # marker today, but the file's real on-disk bytes can still
    # coincidentally contain it (this very fixture file does).
    real_content = "before __HERMES_FENCE_deadbeef__ after\n"

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            assert kwargs.get("strip_fence_leaks") is False, (
                "patch-replace preview must request the UNSTRIPPED read"
            )
            return ReadResult(content=real_content)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    proposal = build_edit_proposal(
        "patch",
        {"mode": "replace", "path": "sample.txt", "old_string": "after", "new_string": "AFTER"},
        task_id="some-task",
    )

    assert proposal.old_text == real_content
    assert "__HERMES_FENCE_deadbeef__" in proposal.old_text


def test_workspace_auto_approval_judges_relative_target_against_task_cwd_not_process_cwd(
    tmp_path, monkeypatch
):
    """should_auto_approve_edit must judge a RELATIVE target against the
    task's own live/registered cwd (the same resolver _resolve_edit_path/
    _resolve_path_for_task the real write uses) -- not this ACP process's own
    ``os.getcwd()``.

    Before the fix, should_auto_approve_edit resolved the raw target path
    with a bare ``Path(raw_path).resolve()``, which falls back to
    ``os.getcwd()``. If the ACP server process happens to have been launched
    from a directory that coincides with the configured workspace boundary
    while the task's REAL live/registered cwd is a different, out-of-
    workspace directory (e.g. after a `cd`, or a session registered
    elsewhere), a relative write that actually lands OUTSIDE the workspace
    could be misjudged as workspace-local and silently skip the approval
    prompt -- the exact class of bug this PR fixes for the preview, one
    layer down in the auto-approval policy check.
    """
    import tools.terminal_tool as terminal_tool

    # Keep the global-tmp auto-approve exemption out of the way (both
    # tmp_path and "outside" below otherwise live under the real system
    # temp root and would auto-qualify regardless of the workspace check --
    # see test_multi_target_v4a_patch_with_outside_path_is_not_auto_approved
    # for the same setup).
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    task_id = "auto-approve-cwd-test"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # The ACP process's own os.getcwd() coincides with the workspace
    # boundary -- exactly the situation that made the bug invisible to a
    # naive Path(raw_path).resolve().
    monkeypatch.chdir(workspace)

    outside = tmp_path / "outside-the-workspace"
    outside.mkdir()

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(outside)})
    try:
        proposal = build_edit_proposal(
            "write_file", {"path": "relative.txt", "content": "x"}, task_id=task_id,
        )
        assert should_auto_approve_edit(proposal, "workspace_session", str(workspace)) is False
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_preview_reads_through_a_real_unmocked_shell_backend(tmp_path, monkeypatch):
    """The only prior coverage claiming backend routing
    (test_preview_reads_through_the_selected_file_backend_not_local_fs)
    fully mocks ``_get_file_ops``, so it cannot catch a regression that
    bypasses ``ShellFileOperations`` entirely (e.g. a stray direct
    ``pathlib.Path`` read reintroduced somewhere in the call chain) -- the
    exact class of bug this PR fixes. Wire a REAL ``ShellFileOperations``
    over a REAL ``LocalEnvironment`` (the same construction
    ``tests/tools/test_file_tools_live.py`` uses for its own no-mocks
    coverage), rooted in an isolated ``tmp_path``, and drive the preview
    through it end-to-end: real shell ``cat``, real existence probe, real
    resolver.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    real_ops = ShellFileOperations(env, cwd=str(tmp_path))

    target = tmp_path / "real_backend.txt"
    target.write_text("real content on disk\n", encoding="utf-8")

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": real_ops)

    proposal = build_edit_proposal(
        "write_file", {"path": str(target), "content": "new content\n"}, task_id="real-backend-task",
    )

    assert proposal.old_text == "real content on disk\n"


def test_write_file_preview_normalizes_new_text_to_match_existing_crlf_ending(monkeypatch):
    """ShellFileOperations.write_file() preserves an existing CRLF file's
    line ending when the model supplies the usual bare-LF content (see
    write_file()'s original_ending handling in tools/file_operations.py).
    The preview's new_text must match that same normalization, or the ACP
    diff compares bare-LF new_text against CRLF old_text and shows every
    UNCHANGED line as modified, even though those exact bytes will not be
    touched by the real write.
    """
    from tools.file_operations import ReadResult

    crlf_content = "line one\r\nline two\r\nline three\r\n"

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=crlf_content)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    proposal = build_edit_proposal(
        "write_file",
        {"path": "sample.txt", "content": "line one\nline two changed\nline three\n"},
        task_id="some-task",
    )

    assert proposal.new_text == "line one\r\nline two changed\r\nline three\r\n"


def test_patch_replace_preview_normalizes_new_text_to_match_existing_crlf_ending(monkeypatch):
    """patch_replace() itself normalizes its fuzzy-replace result to the
    file's dominant line ending (file_ending = _detect_line_ending(content);
    new_content = _normalize_line_endings(new_content, file_ending)) after
    computing it -- a step this preview skipped, so a replacement
    introducing a bare-LF newline (a realistic case: a multi-line
    new_string) would preview a MIXED-ending result the real write will
    never actually produce.
    """
    from tools.file_operations import ReadResult

    crlf_content = "alpha\r\nbeta\r\ngamma\r\n"

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=crlf_content)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    proposal = build_edit_proposal(
        "patch",
        {"mode": "replace", "path": "sample.txt", "old_string": "beta", "new_string": "beta\nextra"},
        task_id="some-task",
    )

    assert proposal.new_text == "alpha\r\nbeta\r\nextra\r\ngamma\r\n"


def test_absolute_path_preview_also_routes_through_the_shared_resolver(monkeypatch):
    """``_resolve_edit_path`` must route an ABSOLUTE path through
    ``tools.file_tools._resolve_path_for_task`` too, not return it
    unresolved. That resolver does more than anchor relative inputs -- it
    also runs a host absolute path through ``Path.resolve()`` (following
    symlinks, e.g. macOS's ``/tmp`` -> ``/private/tmp``) and a container
    path through ``posixpath.normpath`` -- so short-circuiting on
    ``Path(path).is_absolute()`` skipped that normalization entirely,
    letting the preview and the real write resolve the identical-looking
    absolute string to different underlying paths.
    """
    calls = []

    def fake_resolve(path, task_id):
        calls.append((path, task_id))
        return f"/resolved{path}"

    monkeypatch.setattr("tools.file_tools._resolve_path_for_task", fake_resolve)

    from tools.file_operations import ReadResult

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            assert path == "/resolved/tmp/x.txt"
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "/tmp/x.txt", "content": "new\n"}, task_id="abs-path-task",
    )

    # Called twice (once for resolved_target_paths, once inside
    # _read_text_if_exists) -- both with the SAME absolute input, which is
    # exactly what proves it is no longer short-circuited.
    assert calls == [("/tmp/x.txt", "abs-path-task")] * 2
    assert proposal.old_text == "old\n"
    assert proposal.resolved_target_paths == ("/resolved/tmp/x.txt",)


def test_write_file_preview_reads_unstripped_content_like_the_real_write(monkeypatch):
    """``write_file()``'s real write never sanitizes the EXISTING file's
    content -- it fully overwrites, with no dependency on old content at
    all -- so ``old_text`` is pure display and must show the literal bytes
    on disk, not a version with terminal-fence-leak patterns stripped out
    (which would hide real existing content from the approval review).
    """
    from tools.file_operations import ReadResult

    real_content = "before __HERMES_FENCE_deadbeef__ after\n"

    class FakeBackend:
        def read_file_raw(self, path, **kwargs):
            assert kwargs.get("strip_fence_leaks") is False
            return ReadResult(content=real_content)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "sample.txt", "content": "new content\n"}, task_id="some-task",
    )

    assert proposal.old_text == real_content


def test_workspace_auto_approval_resolves_cwd_boundary_via_task_base_dir(tmp_path, monkeypatch):
    """When ``task_id`` is given, ``should_auto_approve_edit`` must judge the
    workspace boundary via ``tools.file_tools._resolve_base_dir(task_id)``
    -- the SAME base a relative target is anchored onto -- not the raw
    ``cwd`` argument alone, which (for a non-local backend, or simply a
    stale client report) can be a different namespace/value than the
    task-resolved target.
    """
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir()
    (real_workspace / "x.txt").write_text("x", encoding="utf-8")

    import tools.terminal_tool as terminal_tool

    task_id = "boundary-namespace-test"
    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(real_workspace)})
    try:
        proposal = build_edit_proposal(
            "write_file", {"path": "x.txt", "content": "y"}, task_id=task_id,
        )
        # A deliberately WRONG boundary (an unrelated directory) is passed as
        # `cwd` -- if should_auto_approve_edit used it directly (as it did
        # before this fix), the edit would be denied even though the target
        # is genuinely inside the real, task-resolved workspace.
        wrong_cwd = str(tmp_path / "an-unrelated-directory")
        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=wrong_cwd, task_id=task_id,
        ) is True
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_workspace_auto_approval_boundary_survives_a_cd_outside_the_original_workspace(
    tmp_path, monkeypatch
):
    """The AUTO_APPROVE_WORKSPACE boundary must stay anchored to the
    ORIGINAL registered session workspace, not the task's live (post-cd)
    terminal cwd.

    The prior fix (resolving the boundary via
    tools.file_tools._resolve_base_dir(task_id)) accidentally tracked the
    LIVE cwd, same as a relative target does -- so once the agent cd'd
    outside the original workspace, the target and the boundary were both
    resolved against the SAME (now out-of-workspace) live directory,
    making the check a tautology that always passed. This is exactly the
    "cd outside the workspace, then auto-approve a write there" scenario
    the boundary check exists to catch: register the ORIGINAL workspace,
    cd elsewhere, then write a relative path.
    """
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    import tools.terminal_tool as terminal_tool

    task_id = "cd-outside-workspace-test"

    original_workspace = tmp_path / "original-workspace"
    original_workspace.mkdir()

    outside = tmp_path / "outside-the-workspace"
    outside.mkdir()

    # Register the session's cwd at "create" time (original_workspace) --
    # what acp_adapter/session.py's _register_task_cwd does -- then the
    # agent `cd`s outside it (tracked separately as the LIVE cwd).
    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(original_workspace)})
    terminal_tool.record_session_cwd(task_id, str(outside))
    try:
        proposal = build_edit_proposal(
            "write_file", {"path": "relative.txt", "content": "x"}, task_id=task_id,
        )
        # `cwd` here is exactly the original workspace (what state.cwd would
        # still report) -- the boundary must stay there, not follow the live
        # cwd the target itself was resolved against.
        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(original_workspace), task_id=task_id,
        ) is False
    finally:
        terminal_tool.clear_task_env_overrides(task_id)
        terminal_tool.clear_session_cwd(task_id)


def test_v4a_preview_on_non_host_backend_does_not_resolve_via_host_resolver(monkeypatch):
    """V4A's real write (tools/file_tools.py's
    ``_rewrite_v4a_patch_paths_for_host``) rewrites patch headers to
    host-resolved paths ONLY for a host-paths backend
    (``_file_ops_uses_host_paths``); for a non-host (SSH/container/sandbox)
    backend it leaves headers untouched and lets THAT backend's own shell
    resolve them against its own live cwd. The preview must never call the
    HOST resolver (``_resolve_edit_path``/``_resolve_path_for_task``) for
    such a target -- that would preview a path the real V4A apply never
    even looks at.

    The preview instead reads via the SAME backend-canonical path
    (``tools.file_tools._resolve_v4a_policy_target``, joining the raw
    header onto the backend's own live ``env.cwd`` with a plain
    ``posixpath`` join, no shell round-trip) that
    ``resolved_target_paths`` uses for the policy check -- resolving ONCE
    and reading that SAME resolved value (rather than handing the backend
    the raw relative header to resolve itself, independently, at read
    time) closes a shared-backend race between the preview read and the
    policy/freeze resolution; see
    ``test_v4a_preview_reads_the_same_snapshot_used_for_policy_resolution``.
    The patch BODY sent to the backend for actual execution stays
    untouched either way (asserted via the patch string itself below).
    """
    from tools.file_operations import ReadResult

    class FakeNonHostEnv:
        """Not a LocalEnvironment -- _file_ops_uses_host_paths() reads this."""

        cwd = "/remote/base"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            # The backend-canonical resolved path, NOT the raw header.
            assert path == "/remote/base/relative.txt"
            return ReadResult(content="container content\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("must not host-resolve a path on a non-host backend")

    monkeypatch.setattr("tools.file_tools._resolve_path_for_task", _must_not_resolve)

    patch_body = "*** Update File: relative.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal(
        "patch", {"mode": "patch", "patch": patch_body}, task_id="container-task",
    )

    assert proposal.old_text == "container content\n"
    # The policy-check target is backend-canonical ("/remote/base/relative.txt"),
    # NOT the raw un-anchored header -- and the patch BODY handed to the
    # backend for actual execution stays untouched (relative header intact;
    # freezing only happens after approval, in maybe_require_edit_approval).
    assert proposal.resolved_target_paths == ("/remote/base/relative.txt",)
    assert proposal.new_text == patch_body


def test_v4a_auto_approval_on_non_host_backend_uses_backend_cwd_not_host_resolve(
    tmp_path, monkeypatch,
):
    """A ``workspace_session`` on a non-host (SSH/container/sandbox) backend
    must judge a V4A patch's auto-approval boundary against where the target
    actually lands IN THE BACKEND'S OWN NAMESPACE, not against a host-side
    ``pathlib.Path.resolve()`` of the raw header.

    Before the fix, ``_proposal_for_patch_v4a`` stored the raw, un-anchored
    V4A header as the policy-check target for a non-host backend, and
    ``should_auto_approve_edit`` fed that raw string straight to
    ``_is_single_path_auto_approvable``'s HOST ``Path(...).resolve()`` --
    resolving it against wherever this ACP process itself happens to be
    running (simulated below via ``monkeypatch.chdir`` into the registered
    workspace, matching "the ACP process was launched inside the
    workspace"), not against the backend's actual live cwd. A relative
    header that the backend would apply somewhere ENTIRELY different (here,
    a simulated container path with no relation to the host workspace at
    all) was misclassified as workspace-local and silently auto-approved.

    This reproduces exactly that setup: a `workspace_session` whose
    registered/original cwd is the client's HOST workspace path, but whose
    backend (`_get_file_ops`) is a non-host, SSH/container-style
    environment with its OWN, entirely different live cwd (as if the agent
    had `cd`-ed there, or the backend was never rooted at the host path to
    begin with) -- then a relative V4A header that resolves, in the
    backend's namespace, to a location with no relation to the registered
    host workspace. The patch must NOT be auto-approved.
    """
    import tools.terminal_tool as terminal_tool
    from tools.file_operations import ReadResult

    task_id = "non-host-cd-outside-workspace-test"

    # The ACP client's registered/original workspace -- a HOST path, and
    # (critically for reproducing the pre-fix bug) also this test process's
    # own cwd, simulating "the ACP process was launched inside the
    # workspace" from the finding.
    host_workspace = tmp_path / "host-workspace"
    host_workspace.mkdir()
    monkeypatch.chdir(host_workspace)

    class FakeNonHostEnv:
        """Not a LocalEnvironment -- _file_ops_uses_host_paths() reads this."""

        # The backend's OWN live cwd: a different namespace entirely from
        # `host_workspace` above, with no path relationship to it.
        cwd = "/backend/elsewhere"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(host_workspace)})
    try:
        patch_body = "*** Update File: secret.txt\n@@\n-old\n+new\n"
        proposal = build_edit_proposal(
            "patch", {"mode": "patch", "patch": patch_body}, task_id=task_id,
        )

        # Backend-canonical: the header resolves, in the backend's own
        # namespace, to "/backend/elsewhere/secret.txt" -- unrelated to
        # host_workspace under either namespace.
        assert proposal.resolved_target_paths == ("/backend/elsewhere/secret.txt",)

        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(host_workspace), task_id=task_id,
        ) is False
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_v4a_session_policy_auto_approves_non_host_tilde_target(monkeypatch):
    """``AUTO_APPROVE_SESSION`` ("Don't Ask") auto-allows every non-sensitive
    edit for the session regardless of where it lands -- it never inspects
    the workspace boundary at all (unlike ``AUTO_APPROVE_WORKSPACE``).

    A non-host V4A header that ``tools.file_tools._resolve_v4a_policy_target``
    cannot canonicalize (a tilde-prefixed path needs a live shell round-trip
    against the backend's own ``$HOME`` -- see that function's docstring) must
    still be auto-approved under this policy: failing the workspace-boundary
    check closed for an unresolvable target is right for
    ``AUTO_APPROVE_WORKSPACE``, but wrongly denying ``AUTO_APPROVE_SESSION``
    too would contradict the ACP "Don't Ask" contract and force an
    unnecessary prompt (or a timeout-denial) for an ordinary edit.
    """
    from tools.file_operations import ReadResult

    class FakeNonHostEnv:
        """Not a LocalEnvironment -- _file_ops_uses_host_paths() reads this."""

        cwd = "/remote/base"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    patch_body = "*** Update File: ~/notes.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal(
        "patch", {"mode": "patch", "patch": patch_body}, task_id="session-policy-task",
    )

    # Unresolvable (tilde-prefixed) on a non-host backend.
    assert proposal.resolved_target_paths == (None,)

    assert should_auto_approve_edit(proposal, "session") is True


def test_workspace_auto_approval_maps_boundary_into_docker_workspace_mount(tmp_path, monkeypatch):
    """The ``AUTO_APPROVE_WORKSPACE`` boundary must be translated into a
    Docker backend's OWN mounted namespace when its
    ``docker_mount_cwd_to_workspace`` feature bind-mounts EXACTLY the
    client-reported workspace to a fixed in-container path (normally
    ``/workspace``).

    ``_resolve_workspace_boundary`` previously ran ONLY the client-reported,
    host-style workspace (e.g. ``/Users/me/project``) through
    ``tools.file_tools._resolve_path_for_task`` -- for an absolute input
    that resolver just normalizes the string, it never maps a host path
    onto the backend's own mount point. Meanwhile a V4A target on that same
    backend resolves (via ``tools.file_tools._resolve_v4a_policy_target``)
    to the backend-canonical ``/workspace/x`` -- an entirely different,
    textually-unrelated namespace. Comparing the two directly made a
    legitimate, in-workspace V4A edit fail the boundary check and prompt
    (or, under a stricter policy, get denied) even though it targets the
    exact directory the session was configured with.

    ``tools.terminal_tool._resolve_task_host_cwd`` is the single source of
    truth for which host directory (if any) got mounted for this task; this
    test drives it (and ``_get_env_config``) through their real module
    attributes -- not a bespoke boundary-mapping mock -- so the fix is
    proven against the actual seam it reads from, and confirms the second
    boundary candidate is added ONLY when that mount source provably IS the
    registered workspace (an unconfigured/non-Docker task must still fail
    closed on the unmapped host-style boundary alone, per the untouched
    ``test_v4a_auto_approval_on_non_host_backend_uses_backend_cwd_not_host_resolve``
    above).
    """
    import tools.terminal_tool as terminal_tool
    from tools.file_operations import ReadResult

    task_id = "docker-mount-workspace-boundary-task"
    host_workspace = tmp_path / "host-workspace"
    host_workspace.mkdir()

    class FakeDockerEnv:
        """Not a LocalEnvironment -- _file_ops_uses_host_paths() reads this."""

        cwd = "/workspace"

    class FakeDockerBackend:
        env = FakeDockerEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        # The live symlink-safety probe (tools.file_tools
        # ._verify_realpath_within_any) needs a backend that can run
        # commands -- this one reports the target as its OWN real path
        # (no symlink in the way), confirming the mapped boundary above is
        # genuinely safe to auto-approve.
        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            return _fake_walkup_exec(command)

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeDockerBackend())
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "docker", "cwd": "/workspace", "host_cwd": str(host_workspace)},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._resolve_task_host_cwd",
        lambda config, task_id: config.get("host_cwd"),
    )

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(host_workspace)})
    try:
        proposal = build_edit_proposal(
            "patch", {"mode": "patch", "patch": "*** Update File: x.txt\n@@\n-old\n+new\n"},
            task_id=task_id,
        )
        # Backend-canonical: /workspace/x.txt, unrelated in string terms to
        # host_workspace -- exactly the mismatch the fix must bridge.
        assert proposal.resolved_target_paths == ("/workspace/x.txt",)

        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(host_workspace), task_id=task_id,
        ) is True
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_v4a_auto_approval_denies_a_non_host_symlink_escape(tmp_path, monkeypatch):
    """A V4A target that LOOKS workspace-local under the lexical
    ``tools.file_tools._resolve_v4a_policy_target`` join must still be
    denied ``AUTO_APPROVE_WORKSPACE`` auto-approval when the non-host
    backend's OWN filesystem resolves it, via a real symlink, to a location
    outside every workspace boundary.

    ``/workspace/link`` is a symlink to ``/outside`` that only exists on
    the backend -- invisible to both the purely lexical
    ``posixpath.normpath`` join that computes ``resolved_target_paths`` and
    to a host-side ``pathlib.Path.resolve()`` (which cannot see a backend-
    only symlink either). Before this fix, a V4A header like
    ``/workspace/link/file`` would pass the lexical containment check and
    auto-approve, even though the backend's own shell -- which is what
    actually applies the patch -- follows the real symlink and writes
    ``/outside/file`` instead. The live ``readlink -f`` verification this
    fix adds is the only vantage point that can see that symlink at all,
    and must deny auto-approval (falling through to the normal approval
    prompt, never silently allowing OR silently writing) whenever it
    reports a real location outside every boundary.
    """
    import tools.terminal_tool as terminal_tool
    from tools.file_operations import ReadResult

    task_id = "non-host-symlink-escape-task"
    host_workspace = tmp_path / "host-workspace"
    host_workspace.mkdir()

    class FakeDockerEnv:
        cwd = "/workspace"

    class FakeDockerBackend:
        env = FakeDockerEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            # Simulates the backend's real filesystem: /workspace/link is a
            # symlink to /outside, so the target's REAL path lands entirely
            # outside both /workspace and the host-mapped boundary.
            return _fake_walkup_exec(command, {"/workspace/link/file.txt": "/outside/file.txt"})

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeDockerBackend())
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "docker", "cwd": "/workspace", "host_cwd": str(host_workspace)},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._resolve_task_host_cwd",
        lambda config, task_id: config.get("host_cwd"),
    )

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(host_workspace)})
    try:
        proposal = build_edit_proposal(
            "patch", {"mode": "patch", "patch": "*** Update File: link/file.txt\n@@\n-old\n+new\n"},
            task_id=task_id,
        )
        # Lexically looks workspace-local -- this is exactly what makes the
        # escape dangerous: a naive containment check would approve it.
        assert proposal.resolved_target_paths == ("/workspace/link/file.txt",)

        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(host_workspace), task_id=task_id,
        ) is False
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_verify_realpath_within_any_keeps_the_targets_own_symlink_literal_when_dereference_final_is_false():
    """``dereference_final=False`` (set for a V4A ``Delete`` target or a
    ``Move``'s source -- see ``EditProposal.target_dereference_final``)
    must resolve only the target's PARENT chain and reattach its own
    final path component literally, never invoking ``readlink -f``/
    ``realpath`` on the target itself even when it is a real symlink.

    ``ShellFileOperations.delete_file()``/``move_file()`` act on the
    filesystem ENTRY (``Path.unlink()``/``mv``), not on whatever a
    symlink there points to, so a workspace-local symlink pointing
    outside every boundary must still verify as in-workspace for these
    two operations -- the opposite of the default ``dereference_final=
    True`` behaviour, which correctly denies such a target for a
    write-through op (see
    ``test_v4a_auto_approval_denies_a_non_host_symlink_escape``).
    """
    from tools.file_operations import ExecuteResult
    from tools.file_tools import _verify_realpath_within_any

    class FakeBackend:
        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            # A real backend filesystem where /workspace is an ordinary
            # directory and /workspace/link is a real symlink to
            # /outside. The dereference_final=False script never assigns
            # the raw target to `p` directly (it goes through `full=` /
            # `dirname` first) -- distinguishing on that is exactly the
            # code-path difference this fix introduces.
            if "full=" in command:
                target_line = "T\t/workspace/link"
            else:
                target_line = "T\t/outside"
            return ExecuteResult(stdout=f"{target_line}\nB0\t/workspace\n", exit_code=0)

    backend = FakeBackend()

    assert _verify_realpath_within_any(
        "/workspace/link", ("/workspace",), backend, dereference_final=False,
    ) is True
    assert _verify_realpath_within_any(
        "/workspace/link", ("/workspace",), backend, dereference_final=True,
    ) is False


def test_v4a_delete_auto_approval_does_not_dereference_a_workspace_symlink_target(tmp_path, monkeypatch):
    """A V4A ``Delete`` targeting a workspace-local symlink that points
    OUTSIDE every boundary must still auto-approve under
    ``AUTO_APPROVE_WORKSPACE``: ``ShellFileOperations.delete_file()``
    calls ``Path.unlink()`` on the entry itself, removing the symlink
    without ever touching whatever it points to. ``_extract_v4a_patch_paths``
    marks a Delete target's ``target_dereference_final`` as False, so the
    live verification must resolve only the target's parent chain and
    keep "link" as its own (in-workspace) name, instead of following it
    to /outside and wrongly denying auto-approval the way a write-through
    Update correctly does (see
    ``test_v4a_auto_approval_denies_a_non_host_symlink_escape``).
    """
    import re

    import tools.terminal_tool as terminal_tool
    from tools.file_operations import ExecuteResult, ReadResult

    task_id = "non-host-symlink-delete-task"
    host_workspace = tmp_path / "host-workspace"
    host_workspace.mkdir()

    class FakeDockerEnv:
        cwd = "/workspace"

    class FakeDockerBackend:
        env = FakeDockerEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            tags = re.findall(r"printf '(\w+)\\t", command)
            # /workspace/link is a REAL symlink to /outside on the
            # backend's filesystem. The correct dereference_final=False
            # script (used for a Delete target) resolves only its parent
            # ("/workspace", an ordinary directory) and reattaches "link"
            # literally, so it must never surface "/outside" here.
            target_line = "T\t/outside" if "p='/workspace/link';" in command else "T\t/workspace/link"
            lines = [target_line] + [f"{tag}\t/workspace" for tag in tags if tag != "T"]
            return ExecuteResult(stdout="\n".join(lines) + "\n", exit_code=0)

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeDockerBackend())
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "docker", "cwd": "/workspace", "host_cwd": str(host_workspace)},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._resolve_task_host_cwd",
        lambda config, task_id: config.get("host_cwd"),
    )

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(host_workspace)})
    try:
        proposal = build_edit_proposal(
            "patch", {"mode": "patch", "patch": "*** Delete File: link\n"},
            task_id=task_id,
        )
        assert proposal.resolved_target_paths == ("/workspace/link",)
        assert proposal.target_dereference_final == (False,)

        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(host_workspace), task_id=task_id,
        ) is True
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_write_file_preview_line_ending_uses_the_same_byte_window_as_the_real_write(monkeypatch):
    """The preview's line-ending detection must agree with the real write's
    own probe even for a UTF-8 file whose first newline sits after byte
    4096 but before character 4096 (e.g. ~3000 emoji ahead of a CRLF).

    ``ShellFileOperations._probe_write_target()``'s production path (no
    ``pre_content`` supplied -- how ``write_file_tool``/``patch_tool``
    actually call it) detects the line ending from a live ``head -c 4096``
    probe: a BYTE window over the on-disk file that, for this fixture,
    contains only emoji and therefore no newline at all. Before the fix,
    ``_normalize_new_text_for_preview`` fed the FULL ``old_text`` straight
    to ``_detect_line_ending``, whose own ``sample[:4096]`` is a CHARACTER
    slice -- since this fixture has well under 4096 characters total, that
    slice covers the WHOLE string, including the CRLF the real byte-based
    probe never sees, and wrongly normalized new_text to CRLF.
    """
    from tools.file_operations import ReadResult

    # ~3000 emoji (4 bytes each in UTF-8 == ~12000 bytes, so byte offset
    # 4096 falls well before them ending) then a CRLF -- character offset
    # 4096 falls comfortably past the whole (~3010-character) string.
    emoji_old_text = "\U0001F600" * 3000 + "\r\nafter\r\n"

    class FakeEmojiBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=emoji_old_text)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeEmojiBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "emoji.txt", "content": "plain new content\n"}, task_id="some-task",
    )

    # Must NOT be normalized to CRLF: the real write's byte-capped probe
    # never observes a line ending in the first 4096 bytes of this file.
    assert proposal.new_text == "plain new content\n"


def test_patch_replace_preview_line_ending_uses_the_full_character_window_not_the_byte_one(monkeypatch):
    """Unlike ``write_file``, ``patch_replace()``'s real write must NOT be
    matched by a byte-capped line-ending sample -- it already has the FULL
    file content in hand (its own ``_cat()``, for the fuzzy match) and
    passes that same content through to ``write_file()`` as ``pre_content``,
    so ``_probe_write_target()`` takes its ``pre_content`` branch and
    detects the line ending via ``_detect_line_ending(pre_content)`` on the
    FULL (character-sliced) text -- the live byte-based ``head -c 4096``
    probe never runs at all in that case.

    Using the SAME ~3000-emoji-then-CRLF fixture as the write_file byte-
    window test above (deliberately chosen so the two proposal kinds must
    disagree): for patch_replace, the character-based full-text detection
    DOES see the CRLF (unlike write_file's byte-capped one), so the
    preview's normalized replacement must be CRLF, not the model's bare LF.
    """
    from tools.file_operations import ReadResult

    # Same divergence fixture as the write_file byte-window test: byte
    # offset 4096 falls well before the emoji run ends, but character
    # offset 4096 falls past the whole (~3010-character) string.
    emoji_old_text = "\U0001F600" * 3000 + "\r\nafter\r\n"

    class FakeEmojiBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=emoji_old_text)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeEmojiBackend()
    )

    proposal = build_edit_proposal(
        "patch",
        {"mode": "replace", "path": "emoji.txt", "old_string": "after", "new_string": "AFTER"},
        task_id="some-task",
    )

    # patch_replace's real write DOES see the CRLF (full-text, character-
    # based detection) -- the preview must normalize to match it, the
    # opposite of the write_file case above.
    assert proposal.new_text == "\U0001F600" * 3000 + "\r\nAFTER\r\n"


def test_write_file_preview_uses_character_window_for_lint_or_lsp_covered_extensions(monkeypatch):
    """Unlike a plain-text extension, ``write_file()``'s real probe for a
    linted/LSP-covered extension (e.g. ``.py``) sets ``want_pre=True`` and
    reads FULL pre-content -- detecting the line ending via
    ``_detect_line_ending(pre_content)`` on that FULL (character-sliced)
    text, the SAME character-based path ``patch_replace`` always takes,
    never the byte-capped ``head -c 4096`` probe the plain-text write_file
    case (see
    ``test_write_file_preview_line_ending_uses_the_same_byte_window_as_the_real_write``)
    is byte-capped to match.

    Using the SAME ~3000-emoji-then-CRLF fixture, but with a ``.py`` path
    this time: the preview must normalize to CRLF, the OPPOSITE of the
    plain-text write_file outcome for the identical bytes.
    """
    from tools.file_operations import ReadResult

    emoji_old_text = "\U0001F600" * 3000 + "\r\nafter\r\n"

    class FakeEmojiBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=emoji_old_text)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeEmojiBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "emoji.py", "content": "plain new content\n"}, task_id="some-task",
    )

    assert proposal.new_text == "plain new content\r\n"


def test_v4a_auto_approval_denies_a_non_host_symlink_escape_via_the_tmp_exemption(monkeypatch):
    """The global-temp-dir ``AUTO_APPROVE_WORKSPACE`` exemption must ALSO go
    through the non-host symlink-safety verification, not bypass it via an
    unconditional early return.

    ``/tmp/link`` is a symlink to ``/outside`` that only exists on a
    non-host backend: lexically, ``/tmp/link/file.txt`` qualifies for the
    global-tmp exemption on the host too (``tempfile.gettempdir()`` is
    ``/tmp`` on Linux), but the backend's own shell -- which actually
    applies the patch -- follows the real symlink and writes
    ``/outside/file.txt`` instead. Before this fix, the tmp-dir check
    ``return``ed immediately, so the live ``verify_backend`` gate below it
    was never even reached for this branch.
    """
    from tools.file_operations import ReadResult

    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")

    task_id = "non-host-tmp-symlink-escape-task"

    class FakeNonHostEnv:
        cwd = "/remote/base"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            # Simulates the backend's real filesystem: /tmp/link is a
            # symlink to /outside.
            return _fake_walkup_exec(command, {"/tmp/link/file.txt": "/outside/file.txt"})

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend())

    patch_body = "*** Update File: /tmp/link/file.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id=task_id)
    # Absolute V4A header -- _resolve_v4a_policy_target normalizes it as-is,
    # still lexically under /tmp.
    assert proposal.resolved_target_paths == ("/tmp/link/file.txt",)

    assert should_auto_approve_edit(proposal, "workspace_session", cwd=None, task_id=task_id) is False


def test_verify_realpath_within_any_resolves_new_nested_paths_without_requiring_parents(tmp_path):
    """``readlink -f``/``realpath`` require every path component but the
    LAST to already exist, so a brand-new nested target (``write_file``/V4A
    ADD create missing parent directories on write, e.g.
    ``workspace/newdir/newdir2/file.py`` when neither ``newdir`` nor
    ``newdir2`` exist yet) must not be treated as unverifiable and force an
    unnecessary approval prompt.

    ``tools.file_tools._verify_realpath_within_any`` walks up to the
    nearest EXISTING ancestor, resolves THAT ancestor's real path, and
    appends the missing suffix back on literally. Exercised here against a
    REAL ``ShellFileOperations``-over-``LocalEnvironment`` backend (not a
    fake ``_exec`` reply), so the actual shell script is proven to work on
    a real filesystem, not just this module's handling of a canned result.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    from tools.file_tools import _verify_realpath_within_any

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = LocalEnvironment(cwd=str(workspace), timeout=15)
    real_ops = ShellFileOperations(env, cwd=str(workspace))

    # newdir/newdir2 do NOT exist yet.
    nested_new_target = str(workspace / "newdir" / "newdir2" / "file.py")

    assert _verify_realpath_within_any(nested_new_target, (str(workspace),), real_ops) is True

    # Sanity: the walk-up must stop at the correct existing ancestor
    # (workspace itself), not misreport containment under an unrelated
    # existing sibling directory.
    (workspace / "unrelated").mkdir()
    assert _verify_realpath_within_any(nested_new_target, (str(workspace / "unrelated"),), real_ops) is False

    # A DANGLING symlink (its target doesn't exist yet) must be resolved,
    # not treated as "missing" and walked past: `-e` alone follows symlinks
    # and would see /workspace/link as absent, walk up to /workspace, and
    # reconstruct the symlink's own lexical path -- never noticing it
    # actually points outside the workspace. The real _atomic_write()
    # explicitly checks `-L` and follows a dangling symlink the same as a
    # live one, so this verifier must too.
    outside = tmp_path / "outside"
    outside.mkdir()
    dangling_target = outside / "new.txt"  # does not exist yet
    danglink = workspace / "link"
    danglink.symlink_to(dangling_target)

    assert _verify_realpath_within_any(str(danglink), (str(workspace),), real_ops) is False
    assert _verify_realpath_within_any(str(danglink), (str(outside),), real_ops) is True


def test_write_file_preview_line_ending_accounts_for_a_stripped_bom(monkeypatch):
    """The preview's byte-capped line-ending sample must account for a
    leading UTF-8 BOM that ``read_file_raw()`` already stripped from
    ``old_text``: the real ``head -c 4096`` probe reads the RAW on-disk
    bytes, BOM included, so its 4096-byte window covers 3 FEWER bytes of
    actual content than a naive 4096-byte cap of the (already BOM-less)
    ``old_text`` would.

    Fixture: 4092 ASCII characters then CRLF then more text. With the BOM
    counted in (as the real probe does), the on-disk window (bytes 0-4095)
    is BOM(3) + 4092 A's + the bare '\\r' of the CRLF at byte 4095 -- it
    ends exactly one byte before the '\\n', so the real write's probe sees
    NEITHER "\\r\\n" nor a bare "\\n" and does not normalize. A byte cap
    that ignores the missing BOM bytes would instead cover 3 bytes further
    into old_text, landing past the full "\\r\\n" and wrongly normalizing.
    """
    from tools.file_operations import ReadResult

    old_text = "A" * 4092 + "\r\nrest\r\n"

    class FakeBomBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=old_text, _had_bom=True)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBomBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "bom.txt", "content": "plain new content\n"}, task_id="some-task",
    )

    # Must NOT be normalized to CRLF: accounting for the stripped BOM, the
    # real write's byte-capped probe window ends one byte before the '\n'.
    assert proposal.new_text == "plain new content\n"


def test_verify_backend_accepts_the_remote_temp_root_alongside_workspace_boundaries(monkeypatch):
    """The non-host symlink-safety verification must accept the backend's
    OWN temp root as a SECOND valid boundary, not just the workspace
    boundary candidates -- otherwise an ordinary, non-escaping remote
    ``/tmp/x`` edit (which the lexical check already exempts via
    ``tempfile.gettempdir()``) would be wrongly denied by the verify step
    for landing outside the (unrelated) workspace.
    """
    from tools.file_operations import ReadResult

    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")

    task_id = "non-host-tmp-legitimate-task"

    class FakeNonHostEnv:
        cwd = "/remote/base"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            # No symlink involved: the backend reports the target's own
            # real path unchanged.
            return _fake_walkup_exec(command)

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend())

    patch_body = "*** Update File: /tmp/x.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id=task_id)
    assert proposal.resolved_target_paths == ("/tmp/x.txt",)

    # cwd (workspace boundary) is unrelated to /tmp -- only the temp-root
    # candidate the verify step must also accept makes this legitimate.
    assert should_auto_approve_edit(
        proposal, "workspace_session", cwd="/remote/base/some-other-workspace", task_id=task_id,
    ) is True


def test_verify_backend_uses_the_selected_backends_own_temp_dir_not_the_controllers(monkeypatch):
    """The temp-dir exemption (lexical AND backend verification) must use
    the SELECTED non-host backend's OWN temp directory
    (``BaseEnvironment.get_temp_dir()``), not the ACP controller's
    ``tempfile.gettempdir()``.

    The ACP server can run on macOS/Windows (``tempfile.gettempdir()``
    returning e.g. ``/var/folders/...`` or ``C:\\Users\\...\\Temp``) while
    a selected SSH/container backend is POSIX -- in that case the
    controller's own temp root describes nothing on the backend at all, so
    a genuinely-in-backend-/tmp edit must still be recognized via the
    backend's OWN reported temp dir instead.
    """
    from tools.file_operations import ReadResult

    # The controller's "global" temp dir looks nothing like the backend's
    # own /tmp.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/var/folders/xy/controller-temp")

    task_id = "non-host-backend-temp-dir-task"

    class FakeNonHostEnv:
        cwd = "/remote/base"

        def get_temp_dir(self):
            return "/tmp"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            return _fake_walkup_exec(command)

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend())

    patch_body = "*** Update File: /tmp/x.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id=task_id)
    assert proposal.resolved_target_paths == ("/tmp/x.txt",)

    assert should_auto_approve_edit(
        proposal, "workspace_session", cwd="/remote/base/unrelated-workspace", task_id=task_id,
    ) is True


def test_workspace_auto_approval_drops_the_host_path_once_docker_mapping_is_confirmed(tmp_path, monkeypatch):
    """Once a Docker mount is confirmed, the mapped in-container path
    (``/workspace``) must REPLACE the host-style boundary candidate, not
    join it: inside the container, the client-reported host path string
    (e.g. ``/Users/me/project``) names no real location at all, so keeping
    it as a second accepted boundary would let an absolute target that
    happens to reuse that host string literally (e.g. a V4A header naming
    it directly) pass containment despite being unrelated to the
    container's actual mounted workspace.
    """
    import tools.terminal_tool as terminal_tool
    from tools.file_operations import ReadResult

    # Keep the global-tmp exemption out of the way: tmp_path itself lives
    # under the real system temp dir, which would otherwise let the target
    # qualify for the UNRELATED temp-dir boundary and mask what this test
    # is actually checking.
    fake_tmp_root = tmp_path / "unrelated-tmp-root"
    fake_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp_root))

    task_id = "docker-mount-host-path-foreign-task"
    host_workspace = tmp_path / "host-workspace"
    host_workspace.mkdir()

    class FakeDockerEnv:
        cwd = "/workspace"

    class FakeDockerBackend:
        env = FakeDockerEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

        def _escape_shell_arg(self, arg):
            return f"'{arg}'"

        def _exec(self, command, **kwargs):
            # No symlink involved -- the backend reports the (foreign,
            # un-mounted) literal path unchanged.
            return _fake_walkup_exec(command)

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": FakeDockerBackend())
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "docker", "cwd": "/workspace", "host_cwd": str(host_workspace)},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._resolve_task_host_cwd",
        lambda config, task_id: config.get("host_cwd"),
    )

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(host_workspace)})
    try:
        # An absolute V4A header naming the HOST workspace path literally:
        # _resolve_v4a_policy_target returns it unchanged (already
        # absolute), but that string is foreign inside the container --
        # only /workspace is actually mounted there.
        patch_body = f"*** Update File: {host_workspace}/out.txt\n@@\n-old\n+new\n"
        proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id=task_id)
        assert proposal.resolved_target_paths == (f"{host_workspace}/out.txt",)

        assert should_auto_approve_edit(
            proposal, "workspace_session", cwd=str(host_workspace), task_id=task_id,
        ) is False
    finally:
        terminal_tool.clear_task_env_overrides(task_id)


def test_write_file_preview_only_normalizes_the_crlf_case(monkeypatch):
    """``write_file()``'s real write ONLY normalizes incoming content when
    the EXISTING file is CRLF-terminated (``if original_ending == "\\r\\n":
    content = _normalize_line_endings(content, "\\r\\n")``) -- for an
    LF-terminated (or single-line) existing file, incoming content that
    happens to contain CRLF or lone-CR characters is written UNCHANGED, not
    force-normalized to bare LF. The preview must match: only the CRLF case
    triggers normalization.
    """
    from tools.file_operations import ReadResult

    class FakeLfBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="line one\nline two\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeLfBackend()
    )

    mixed_new_content = "line one\r\nline two changed\r\n"
    proposal = build_edit_proposal(
        "write_file", {"path": "lf_file.txt", "content": mixed_new_content}, task_id="some-task",
    )

    # Must stay exactly as the model proposed -- write_file() would write
    # these mixed endings unchanged against an LF-terminated file.
    assert proposal.new_text == mixed_new_content


def test_write_file_preview_character_window_accounts_for_a_stripped_bom(monkeypatch):
    """For a linted/LSP-covered extension (e.g. ``.py``), ``write_file()``'s
    real probe reads FULL pre-content via a bare ``cat`` that does NOT
    strip a leading BOM (unlike ``read_file_raw()``, which always does).
    Its character window therefore covers ONE FEWER actual-content
    character than a same-length slice of our (BOM-less) ``old_text``
    would, for a file where the first CRLF straddles the 4096-character
    boundary.
    """
    from tools.file_operations import ReadResult

    # Exactly 4095 ASCII chars then CRLF then more text. Real probe (BOM +
    # this content): char 4096 of the RAW text is the 4095th ASCII char;
    # its 4096-char window ends there, one character BEFORE the CRLF --
    # sees no line ending. Our old_text (BOM-stripped) capped to the SAME
    # 4095 real characters (4096 - 1 for the missing BOM slot) must match.
    old_text = "A" * 4095 + "\r\nrest\r\n"

    class FakeBomPyBackend:
        def read_file_raw(self, path, **kwargs):
            return ReadResult(content=old_text, _had_bom=True)

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeBomPyBackend()
    )

    proposal = build_edit_proposal(
        "write_file", {"path": "bom.py", "content": "plain new content\n"}, task_id="some-task",
    )

    # Must NOT be normalized to CRLF: accounting for the BOM the real cat
    # output has (and old_text doesn't), the real probe's window ends one
    # character before the CRLF.
    assert proposal.new_text == "plain new content\n"


def test_maybe_require_edit_approval_freezes_non_host_v4a_headers_after_approval(monkeypatch):
    """After a non-host (SSH/container/sandbox) V4A patch is approved,
    ``maybe_require_edit_approval`` must rewrite its headers, IN PLACE on
    the SAME ``arguments`` dict the real dispatch goes on to execute, to
    the exact backend-canonical paths approval was granted for.

    Without this, the relative header stays unresolved and
    ``ShellFileOperations._exec()`` interprets it against the backend's
    *live* ``env.cwd`` only at execution time -- when two ACP sessions
    share a persistent Docker environment, the other session's ``cd``
    between this approval and the actual dispatch could redirect an
    approved write to a completely different path. Freezing the header
    here removes that live-cwd dependency entirely.
    """
    from acp_adapter.edit_approval import maybe_require_edit_approval
    from tools.file_operations import ReadResult

    task_id = "freeze-non-host-v4a-task"

    class FakeNonHostEnv:
        cwd = "/workspace"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    arguments = {"mode": "patch", "patch": "*** Update File: relative.txt\n@@\n-old\n+new\n"}
    set_edit_approval_requester(lambda _proposal: True)
    try:
        result = maybe_require_edit_approval("patch", arguments, task_id=task_id)
    finally:
        set_edit_approval_requester(None)

    assert result is None  # approved, not blocked
    # The SAME arguments dict now carries the frozen, backend-canonical
    # header -- exactly what a subsequent file_ops.patch_v4a() dispatch
    # call would receive, regardless of what env.cwd becomes afterward.
    assert arguments["patch"] == "*** Update File: /workspace/relative.txt\n@@\n-old\n+new\n"


def test_write_file_preview_derives_extension_from_the_resolved_symlink_target(tmp_path, monkeypatch):
    """``write_file_tool()`` hands ``write_file()`` the RESOLVED path
    (``_resolve_path_for_task``'s host ``Path.resolve()`` follows a
    symlink to its real target), so ``write_file()``'s own ``ext =
    os.path.splitext(path)[1]`` sees the symlink TARGET's extension, not
    the symlink name's. The preview must derive its extension from the
    SAME resolved path.

    ``alias.txt -> script.py``: the real write treats this as a ``.py``
    write (lint-covered -- full pre-content, character-based line-ending
    detection); deriving the extension from the raw ``alias.txt`` name
    would instead select the byte-capped probe, disagreeing for the same
    multibyte-prefix-then-CRLF fixture the byte/character-window split
    exists to handle. Exercised against a REAL ``ShellFileOperations``-
    over-``LocalEnvironment`` backend so the real symlink-following
    ``resolve()`` is genuinely exercised, not mocked.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_target = workspace / "script.py"
    # ~3000 emoji (4 bytes each) then CRLF -- byte offset 4096 falls well
    # before the emoji run ends, but the full (character-window) text is
    # scanned in its entirety, so the two windows must disagree here.
    content = "\U0001F600" * 3000 + "\r\nafter\r\n"
    real_target.write_text(content, encoding="utf-8")
    alias = workspace / "alias.txt"
    alias.symlink_to(real_target)

    env = LocalEnvironment(cwd=str(workspace), timeout=15)
    real_ops = ShellFileOperations(env, cwd=str(workspace))
    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": real_ops)

    proposal = build_edit_proposal(
        "write_file", {"path": str(alias), "content": "plain new content\n"}, task_id="some-task",
    )

    # .py is lint-covered -> character-window (full pre-content) detection,
    # which DOES see the CRLF -- must normalize to CRLF, the opposite of
    # what byte-window (from the alias's own ".txt" extension) would give.
    assert proposal.new_text == "plain new content\r\n"


def test_v4a_preview_reads_the_same_snapshot_used_for_policy_resolution(monkeypatch):
    """A non-host V4A preview must read via the ALREADY-RESOLVED
    backend-canonical path (the SAME snapshot ``resolved_target_paths``
    uses), not the raw relative header re-resolved independently.

    If two ACP sessions share a persistent non-host environment, resolving
    the preview read and the policy/freeze target SEPARATELY (each its own
    live ``env.cwd`` lookup) lets the other session's ``cd`` land between
    them -- the preview would then show content read from ONE file while
    the (later-frozen) execution target is a DIFFERENT one. Resolving once
    and reading that SAME resolved path removes the gap entirely.
    """
    from tools.file_operations import ReadResult

    task_id = "single-snapshot-v4a-task"
    captured_paths = []

    class FakeNonHostEnv:
        cwd = "/workspace"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            captured_paths.append(path)
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    patch_body = "*** Update File: relative.txt\n@@\n-old\n+new\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch_body}, task_id=task_id)

    assert proposal.resolved_target_paths == ("/workspace/relative.txt",)
    # The preview read must have used the SAME resolved absolute path, not
    # the raw relative header re-resolved against a possibly-different
    # later env.cwd snapshot.
    assert captured_paths == ["/workspace/relative.txt"]


def test_freeze_rewrites_only_the_resolvable_targets_in_a_mixed_v4a_patch(monkeypatch):
    """A mixed multi-file non-host V4A patch -- one unresolvable target
    (e.g. a tilde header) alongside a resolvable relative one -- must still
    freeze the RESOLVABLE header, not abandon the whole patch's freeze just
    because one target couldn't be canonicalized.
    """
    from acp_adapter.edit_approval import maybe_require_edit_approval
    from tools.file_operations import ReadResult

    task_id = "mixed-freeze-v4a-task"

    class FakeNonHostEnv:
        cwd = "/workspace"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    arguments = {
        "mode": "patch",
        "patch": (
            "*** Update File: ~/notes\n@@\n-old\n+new\n"
            "*** Update File: src/x.py\n@@\n-old\n+new\n"
        ),
    }
    set_edit_approval_requester(lambda _proposal: True)
    try:
        result = maybe_require_edit_approval("patch", arguments, task_id=task_id)
    finally:
        set_edit_approval_requester(None)

    assert result is None
    # The resolvable header is frozen to its backend-canonical path...
    assert "*** Update File: /workspace/src/x.py" in arguments["patch"]
    # ...while the unresolvable tilde header is left exactly as written,
    # for the backend's own live resolution.
    assert "*** Update File: ~/notes" in arguments["patch"]


def test_verify_realpath_within_any_resolves_a_symlinked_boundary_too(tmp_path):
    """When the workspace BOUNDARY itself is a symlink on the backend
    (e.g. ``/workspace -> /srv/project``), a target the backend reports as
    living under the boundary's REAL path must still be recognized as
    contained -- comparing against the boundary's own unresolved spelling
    would fail every legitimate edit closed and defeat auto-approval
    entirely for this common setup. Exercised against a REAL
    ``ShellFileOperations``-over-``LocalEnvironment`` backend with a
    genuine symlinked directory.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    from tools.file_tools import _verify_realpath_within_any

    real_project = tmp_path / "srv-project"
    real_project.mkdir()
    workspace_link = tmp_path / "workspace"
    workspace_link.symlink_to(real_project)

    target = real_project / "file.txt"
    target.write_text("hi", encoding="utf-8")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    real_ops = ShellFileOperations(env, cwd=str(tmp_path))

    # The target is addressed THROUGH the symlinked boundary spelling.
    target_via_link = str(workspace_link / "file.txt")
    assert _verify_realpath_within_any(target_via_link, (str(workspace_link),), real_ops) is True


def test_freeze_does_not_rewrite_a_traversal_header(monkeypatch):
    """A V4A header containing a ``..`` traversal component must NEVER be
    frozen, regardless of what it resolves to: ``patch_tool()``'s own
    ``_collect_v4a_header_paths()`` unconditionally rejects any header
    containing ``..`` before execution. Rewriting it to its normalized
    (``..``-free) absolute form here would silently launder it past that
    check, letting ACP execute an input every other dispatch path refuses.
    """
    from acp_adapter.edit_approval import maybe_require_edit_approval
    from tools.file_operations import ReadResult

    task_id = "traversal-freeze-task"

    class FakeNonHostEnv:
        cwd = "/workspace"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            return ReadResult(content="old\n")

    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id="default": FakeNonHostBackend()
    )

    original_patch = "*** Update File: ../outside.txt\n@@\n-old\n+new\n"
    arguments = {"mode": "patch", "patch": original_patch}
    set_edit_approval_requester(lambda _proposal: True)
    try:
        result = maybe_require_edit_approval("patch", arguments, task_id=task_id)
    finally:
        set_edit_approval_requester(None)

    assert result is None
    # Left completely untouched -- the raw ".." header survives so
    # patch_tool()'s own traversal rejection still sees and rejects it.
    assert arguments["patch"] == original_patch
