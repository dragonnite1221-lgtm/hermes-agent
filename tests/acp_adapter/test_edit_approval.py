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
        def read_file_raw(self, path):
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
        def read_file_raw(self, path):
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
        def read_file_raw(self, path):
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
        def read_file_raw(self, path):
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
        def read_file_raw(self, path):
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
