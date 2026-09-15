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
    resolve them against its own live cwd. The preview must do the same --
    resolving via ``_resolve_edit_path`` (host-anchored) would preview a
    path the real V4A apply never even looks at.

    ``resolved_target_paths`` (used only by ``should_auto_approve_edit``'s
    policy check, never for the patch body itself) must still land on a
    BACKEND-canonical location -- ``tools.file_tools._resolve_v4a_policy_target``
    joining the raw header onto the backend's own live ``env.cwd`` with a
    plain ``posixpath`` join, no shell round-trip -- rather than the raw,
    un-anchored header string a bare host resolve was leaving behind.
    """
    from tools.file_operations import ReadResult

    class FakeNonHostEnv:
        """Not a LocalEnvironment -- _file_ops_uses_host_paths() reads this."""

        cwd = "/remote/base"

    class FakeNonHostBackend:
        env = FakeNonHostEnv()

        def read_file_raw(self, path, **kwargs):
            assert path == "relative.txt"  # raw, unresolved
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
    # NOT the raw un-anchored header -- but the patch body handed to the
    # backend (asserted inside read_file_raw above) stays untouched.
    assert proposal.resolved_target_paths == ("/remote/base/relative.txt",)


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
