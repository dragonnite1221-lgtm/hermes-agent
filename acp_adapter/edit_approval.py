"""Pre-execution ACP edit approval helpers.

Intentionally isolated from the generic tool registry: ACP binds an edit
approval requester in a ContextVar for the duration of one ACP agent run; CLI,
gateway, and other sessions leave it unset and therefore bypass this guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from contextvars import ContextVar, Token
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditProposal:
    """A proposed edit that can be shown to an ACP client.

    ``path`` is a display string only (for multi-file V4A patches it is a
    comma-joined summary and must never be parsed back into a filesystem
    path). ``target_paths`` holds the real, individual target paths so
    approval logic can check each one; it defaults to ``(path,)`` for the
    single-target proposal kinds. ``resolved_target_paths`` holds the SAME
    targets after running through ``_resolve_edit_path`` (task-live-cwd-aware,
    matching what the real write will touch) -- policy checks must prefer
    these over the raw ``target_paths``/``path`` so a relative target is
    judged against the same location the write actually lands on, not this
    ACP process's own cwd; see ``should_auto_approve_edit``.
    """

    tool_name: str
    path: str
    old_text: str | None
    new_text: str
    arguments: dict[str, Any]
    target_paths: tuple[str, ...] | None = None
    resolved_target_paths: tuple[str, ...] | None = None


EditApprovalRequester = Callable[[EditProposal], bool]

_EDIT_APPROVAL_REQUESTER: ContextVar[EditApprovalRequester | None] = ContextVar("ACP_EDIT_APPROVAL_REQUESTER", default=None)
_PERMISSION_REQUEST_IDS = count(1)

SENSITIVE_AUTO_APPROVE_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
AUTO_APPROVE_ASK = "ask"
AUTO_APPROVE_WORKSPACE = "workspace_session"
AUTO_APPROVE_SESSION = "session"


def set_edit_approval_requester(requester: EditApprovalRequester | None) -> Token:
    """Bind an ACP edit approval requester for the current context."""
    return _EDIT_APPROVAL_REQUESTER.set(requester)


def reset_edit_approval_requester(token: Token) -> None:
    """Restore a previous edit approval requester binding."""
    _EDIT_APPROVAL_REQUESTER.reset(token)


def _resolve_edit_path(path: str, task_id: str = "default") -> str:
    """Resolve ``path`` through the exact same resolver the tool backend
    calls when it actually performs the write.

    ``tools.file_tools._resolve_path_for_task`` is that resolver: it is the
    literal function ``write_file_tool``/``patch`` call, and its resolution
    order already covers everything the real write can be anchored to --
    the task's *live* terminal cwd (updated by every ``cd`` the agent runs
    mid-session, via ``tools.terminal_tool.record_session_cwd``), then a
    registered task/session cwd override (what ACP registers at session
    create/load/resume — see ``acp_adapter/session.py``'s
    ``_register_task_cwd``), then ``$TERMINAL_CWD``, then the process cwd.

    Returns a string rather than a local ``Path``: for a non-local backend
    (SSH, container, sandbox) the resolved value lives in THAT backend's
    namespace, not this process's filesystem, and must only ever be handed
    to that same backend (see ``_read_text_if_exists``) -- never opened
    directly here.

    A leading ``~`` is deliberately NOT expanded here with
    ``Path.expanduser()``/``os.path.expanduser`` (which use this ACP
    process's own ``$HOME``): ``_resolve_path_for_task`` expands it itself
    via ``tools.file_tools._expand_tilde``, which prefers Hermes'
    profile-specific subprocess home over the raw process ``$HOME`` when
    they differ (gateway/cron contexts in particular). Expanding it
    ourselves first would make every ``~/...`` path "look" absolute and
    skip ``_resolve_path_for_task`` entirely, previewing a file resolved
    against the wrong home while the real write resolves it correctly.

    An ABSOLUTE ``path`` is also routed through ``_resolve_path_for_task``
    rather than returned as-is: that resolver does more than anchor relative
    inputs onto a base directory -- ``_anchor()`` also runs a host-backend
    absolute path through ``Path.resolve()`` (following symlinks; e.g.
    macOS's ``/tmp`` -> ``/private/tmp``, ``/var`` -> ``/private/var``) or a
    container backend's path through ``posixpath.normpath`` with no host
    symlink following, AND applies ``_host_text``'s Git-Bash/MSYS drive-path
    translation on Windows -- all BEFORE deciding whether the input was
    already absolute in that normalized form. Returning an absolute input
    unresolved skipped every one of those, so the preview and the real write
    could resolve the identical-looking string to different underlying
    paths. ``_resolve_path_for_task`` itself already special-cases an
    absolute input internally (no base-dir anchoring), so calling it
    unconditionally here is correct for both absolute and relative paths.
    """
    from tools.file_tools import _resolve_path_for_task

    return str(_resolve_path_for_task(path, task_id))


def _read_text_if_exists(
    path: str, task_id: str = "default", *, strip_fence_leaks: bool = True, deny_binary: bool = True,
) -> str | None:
    """Read ``path``'s current content for the approval diff.

    Reads through ``tools.file_tools._get_file_ops(task_id)`` -- the same
    ``ShellFileOperations`` backend ``write_file_tool``/``patch_tool`` use
    for the actual mutation -- instead of a local ``pathlib.Path`` read.
    For a local session that backend still shells out on this machine, so
    behavior is unchanged; for an SSH/container/sandbox-backed task, a
    direct local read would either wrongly report "file does not exist"
    for a path that only exists in the remote/container namespace, or read
    an unrelated file that happens to exist at that same-looking path on
    the ACP server's own host.

    ``strip_fence_leaks=False`` (used by ``_proposal_for_write_file`` and
    ``_proposal_for_patch_replace``; V4A keeps the default) opts out of
    ``ShellFileOperations``'s ``_strip_terminal_fence_leaks`` cleanup. That
    cleanup is applied by ``read_file_raw()`` (used here, and by the real
    V4A apply path in ``tools/patch_parser.py``, so V4A's preview and write
    already agree on stripped content) but NOT by ``write_file()`` (never
    reads old content for the write at all -- old_text is pure display) or
    ``patch_replace()``'s own internal read (``tools/file_operations.py``
    reads via a bare ``self._cat()`` for the "replace" patch mode -- old_text
    there also feeds the actual fuzzy-match). If the file's real content
    happens to contain text the stripper's regexes treat as leaked terminal
    wrapper noise (an OSC sequence, or a ``__HERMES_FENCE_<id>__`` marker),
    the default (stripped) preview would hide real existing bytes from the
    approval diff -- and, for patch-replace specifically, fuzzy-match
    old_string against a laundered version of the file while the real
    ``patch_replace()`` matches against the untouched bytes, computing a
    different result than what the approval diff showed the user.

    ``deny_binary=False`` (used only by ``_proposal_for_patch_v4a`` for a
    delete-only operation) trades the binary/image raise below for a
    descriptive placeholder instead. That guard exists to stop an
    existing image/binary file from being previewed as an empty/new one
    ahead of a CONTENT-OVERWRITING write -- irrelevant to a pure delete,
    which never derives new content from old and whose real V4A executor
    (``tools/patch_parser.py``'s ``_apply_delete``) already deletes
    binary/image targets just fine (it only checks ``.error``, never
    ``.is_binary``/``.is_image``). Requiring a successful TEXT preview
    before a delete-only V4A patch on an image/binary file could run
    would deny a delete the real executor allows.
    """
    resolved = _resolve_edit_path(path, task_id)

    from tools.file_tools import _get_file_ops

    file_ops = _get_file_ops(task_id)
    # Only pass the kwarg when non-default so a minimal test double that
    # implements read_file_raw(self, path) (no extra kwarg) keeps working.
    result = file_ops.read_file_raw(resolved) if strip_fence_leaks else file_ops.read_file_raw(
        resolved, strip_fence_leaks=False)
    if result.is_binary or result.is_image:
        if not deny_binary:
            kind = "image" if result.is_image else "binary"
            return f"[existing {kind} file, {result.file_size} bytes -- content not shown]"
        # An existing image sets NO .error at all -- just is_image=True,
        # is_binary=True, and empty (default) .content -- so checking only
        # .error below would fall through to `return result.content` and
        # show old_text="" for it, as if the path were empty or brand new,
        # when it actually holds existing binary/image content the user
        # was never shown. Fail closed instead of masking it as "no old
        # text".
        raise OSError(f"Cannot preview binary/image content at {path!r} for edit approval")
    if result.error:
        # ShellFileOperations.read_file_raw() sets .error for several very
        # different situations: a genuinely missing path (always and only
        # "File not found: {path}", from _suggest_similar_files -- the sole
        # producer of that exact prefix), but ALSO permission failures,
        # non-regular files (directory/FIFO/socket/device), binary content,
        # and transport/backend failures for a path that DOES exist.
        # Collapsing all of those to "no old text" would make an existing
        # (but unreadable-for-preview) file look like a brand-new one, and
        # an approved write_file could then silently overwrite content the
        # user was never shown. Only the confirmed-missing case is a real
        # "new file" signal; anything else must fail the proposal instead
        # of masquerading as one, so maybe_require_edit_approval's
        # fail-closed default kicks in.
        if result.error.startswith("File not found: "):
            # This prefix IS a reliable "confirmed absent" signal, not an
            # ambiguous one: read_file_raw() reaches _suggest_similar_files()
            # (the sole producer of this exact prefix) only when
            # _probe_regular_file()'s existence probe echoes its MISSING
            # sentinel -- i.e. the shell ran and the `[ -e ... ]` check
            # itself said the path is absent. A probe that fails to run at
            # all (dropped SSH connection, a died container, ...) is a
            # DIFFERENT status ("env_unavailable"), reported as "Terminal
            # environment unavailable: ... Retry shortly." -- a different
            # message that does not start with "File not found: " and so
            # falls through to the `raise OSError` below, failing the
            # proposal closed exactly like every other unreadable-for-a-
            # different-reason case. See tools/file_operations.py's
            # _probe_regular_file/read_file_raw.
            return None
        raise OSError(f"Cannot read current content of {path!r}: {result.error}")
    return result.content


def _required_path(arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path") or "")
    if not path:
        raise ValueError("path required")
    return path


def _normalize_new_text_for_preview(old_text: str | None, new_text: str) -> str:
    """Match ``write_file()``/``patch_replace()``'s own line-ending
    normalization, so the preview's ``new_text`` uses the SAME ending the
    real write will produce.

    ``ShellFileOperations.write_file()`` (and ``patch_replace()``, which
    calls it) preserve an existing file's dominant line ending: when the
    on-disk content is CRLF, the incoming (usually bare-LF, since models
    send bare-LF) content/replacement is normalized to CRLF before writing
    -- see ``tools/file_operations.py``'s ``write_file()`` (``original_ending``
    from ``_probe_write_target()``) and ``patch_replace()`` (``file_ending
    = _detect_line_ending(content)``). Without this, the preview's diff
    compares the model's bare-LF ``new_text`` against CRLF ``old_text`` and
    shows every unchanged line as modified (or a mixed-ending result),
    misrepresenting a write that will not actually touch those bytes.

    ``_detect_line_ending`` is the exact same pure helper
    ``_probe_write_target()`` uses on its own pre-content/sample, so this
    reproduces the real write's decision without a second shell round-trip
    (the preview already has ``old_text`` from ``_read_text_if_exists``).
    V4A's real apply path (``tools/patch_parser.py``) does no such
    normalization at all, so its preview (which also skips this) already
    matches.
    """
    if old_text is None:
        return new_text
    from tools.file_operations_common import _detect_line_ending, _normalize_line_endings

    file_ending = _detect_line_ending(old_text)
    return _normalize_line_endings(new_text, file_ending) if file_ending else new_text


def _proposal_for_write_file(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    path = _required_path(arguments)
    content = arguments.get("content")
    if content is None:
        raise ValueError("content required")
    resolved = _resolve_edit_path(path, task_id)
    # strip_fence_leaks=False: write_file()'s real write never sanitizes the
    # EXISTING file's content -- it fully overwrites, with no dependency on
    # old content at all -- so this old_text is pure display. Showing a
    # "cleaned" version would hide real bytes the file actually holds from
    # the user's approval review; see _read_text_if_exists's docstring.
    old_text = _read_text_if_exists(path, task_id, strip_fence_leaks=False)
    new_text = _normalize_new_text_for_preview(old_text, str(content))
    return EditProposal(
        "write_file", path, old_text, new_text, dict(arguments),
        resolved_target_paths=(resolved,),
    )


def _proposal_for_patch_replace(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    path = _required_path(arguments)
    old_string, new_string = arguments.get("old_string"), arguments.get("new_string")
    if old_string is None or new_string is None:
        raise ValueError("old_string and new_string required")
    # strip_fence_leaks=False: the real patch_replace() fuzzy-matches
    # old_string against UNSTRIPPED content (tools/file_operations.py reads
    # it via a bare self._cat(), not read_file_raw()); matching here against
    # the default stripped read could compute a new_text that diverges from
    # what patch_replace() actually produces once approved. See
    # _read_text_if_exists's docstring.
    old_text = _read_text_if_exists(path, task_id, strip_fence_leaks=False)
    if old_text is None:
        raise ValueError(f"Failed to read file: {path}")

    from tools.fuzzy_match import fuzzy_find_and_replace

    new_text, match_count, _strategy, error = fuzzy_find_and_replace(
        old_text, str(old_string), str(new_string), bool(arguments.get("replace_all", False)))
    if error or match_count == 0:
        raise ValueError(error or f"Could not find match for old_string in {path}")
    # Same normalization patch_replace() itself applies after the fuzzy
    # replace (file_ending = _detect_line_ending(content); new_content =
    # _normalize_line_endings(new_content, file_ending)) -- old_text here IS
    # that same content (BOM-stripped, fence-leak-unstripped cat output).
    new_text = _normalize_new_text_for_preview(old_text, new_text)
    resolved = _resolve_edit_path(path, task_id)
    return EditProposal("patch", path, old_text, new_text, dict(arguments), resolved_target_paths=(resolved,))


def _extract_v4a_patch_paths(patch_body: str) -> tuple[list[str], bool]:
    # Reuse the same parser that actually executes the patch (tools/
    # patch_parser.py, via tools/file_operations.py) instead of a second,
    # independently-maintained regex: a prior version of this function had
    # its own `\s+`-after-`***` regex that was stricter than the parser's
    # `\s*`, so a no-space header (`***Update File:`) that the parser still
    # executed could slip past approval extraction entirely, letting an
    # out-of-workspace target hide behind an in-workspace one. Deriving the
    # paths from the real parser makes that class of drift impossible.
    from tools.patch_parser import OperationType, parse_v4a_patch

    operations, _error = parse_v4a_patch(patch_body)
    paths: list[str] = []
    for op in operations:
        if op.file_path:
            paths.append(op.file_path)
        if op.new_path:
            paths.append(op.new_path)
    # Whether this is a single DELETE operation -- the sole V4A op kind whose
    # real executor (tools/patch_parser.py's _apply_delete) never derives new
    # content from old and already deletes binary/image targets just fine.
    is_delete_only = len(operations) == 1 and operations[0].operation is OperationType.DELETE
    return paths, is_delete_only


def _proposal_for_patch_v4a(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    patch_body = arguments.get("patch")
    if not isinstance(patch_body, str) or not patch_body:
        raise ValueError("patch content required")
    paths, is_delete_only = _extract_v4a_patch_paths(patch_body)
    if not paths:
        raise ValueError("no file paths found in V4A patch")
    single = len(paths) == 1
    # ACP only supports a single diff payload: surface the exact V4A patch as new_text so
    # patch-mode calls are permissioned and denied patches cannot mutate.
    return EditProposal(
        tool_name="patch",
        path=paths[0] if single else ", ".join(paths),
        # deny_binary=False only for a delete-only operation (never derives
        # new content from old, so #5's overwrite-masking protection does
        # not apply) -- see _read_text_if_exists's docstring and
        # tools/patch_parser.py's _apply_delete.
        old_text=_read_text_if_exists(paths[0], task_id, deny_binary=not is_delete_only) if single else None,
        # ACP only supports a single diff payload here.  Surface the exact V4A
        # patch content before execution so patch-mode calls are permissioned
        # and denied patches cannot mutate.
        new_text=patch_body,
        arguments=dict(arguments),
        # Keep the real per-file targets alongside the joined display string
        # so approval decisions never parse `path` back into a filesystem path.
        target_paths=tuple(paths),
        # Same targets resolved through the task-live-cwd-aware resolver, so
        # should_auto_approve_edit judges each one against where the real
        # V4A apply will touch, not this process's own cwd.
        resolved_target_paths=tuple(_resolve_edit_path(p, task_id) for p in paths),
    )


# (tool_name, patch mode or None) -> proposal builder.
_PROPOSAL_BUILDERS = {
    ("write_file", None): _proposal_for_write_file, ("patch", "replace"): _proposal_for_patch_replace,
    ("patch", "patch"): _proposal_for_patch_v4a,
}


def build_edit_proposal(
    tool_name: str, arguments: dict[str, Any], task_id: str = "default"
) -> EditProposal | None:
    """Return an edit proposal for supported file mutation calls.

    ``task_id`` must match the id the tool call will actually execute under
    (the ACP session id) so the preview resolves relative paths through the
    exact same live-cwd-aware resolver as the real write -- see
    ``_resolve_edit_path``.
    """
    mode = arguments.get("mode", "replace") if tool_name == "patch" else None
    builder = _PROPOSAL_BUILDERS.get((tool_name, mode))
    return builder(arguments, task_id) if builder else None


def _is_sensitive_auto_approve_path(path: str) -> bool:
    lowered = {part.lower() for part in Path(path).expanduser().parts}
    return bool(lowered & {".git", ".ssh"}) or Path(path).name.lower() in SENSITIVE_AUTO_APPROVE_NAMES


def _is_single_path_auto_approvable(raw_path: str, policy: str, cwd: str | None) -> bool:
    if _is_sensitive_auto_approve_path(raw_path):
        return False
    path = Path(raw_path).expanduser().resolve(strict=False)
    if policy == AUTO_APPROVE_SESSION:
        return True
    if policy == AUTO_APPROVE_WORKSPACE:
        # tempfile.gettempdir() is the real temp root on every platform
        # (``/private/tmp`` on macOS since resolve() follows the symlink).
        return path.is_relative_to(Path(tempfile.gettempdir()).resolve(strict=False)) or (
            bool(cwd) and path.is_relative_to(Path(cwd).expanduser().resolve(strict=False)))
    return False


def _resolve_workspace_boundary(cwd: str | None, task_id: str | None) -> str | None:
    """Return the AUTO_APPROVE_WORKSPACE boundary in the SAME namespace as
    ``proposal.resolved_target_paths``.

    ``cwd`` (``state.cwd`` at the call sites) is the ACP client's own report
    of the session's workspace directory. For a local backend that IS the
    filesystem namespace the write happens in, so comparing it directly
    against a resolved target works. For an SSH/container/sandbox-backed
    task it need not be: ``tools.file_tools._resolve_base_dir(task_id)`` is
    the exact base directory ``_resolve_path_for_task`` anchors a relative
    target onto (and, via ``_anchor``, the same normalization an absolute
    target goes through) -- i.e. the boundary as the resolver itself sees
    it, not as the client separately reported it. When ``task_id`` is
    unavailable (e.g. an ``EditProposal`` built by hand in a test) or the
    lookup fails, fall back to the given ``cwd`` unchanged.
    """
    if not task_id:
        return cwd
    try:
        from tools.file_tools import _resolve_base_dir

        return str(_resolve_base_dir(task_id))
    except Exception:
        return cwd


def should_auto_approve_edit(
    proposal: EditProposal, policy: str, cwd: str | None = None, task_id: str | None = None,
) -> bool:
    """Return whether an ACP edit proposal may bypass the prompt for this session.

    This is intentionally session-scoped and conservative: sensitive paths still
    ask even under autonomous policies. For multi-file V4A patches, ``proposal.path``
    is only a comma-joined display string and must never be parsed back into a
    filesystem path — every real target in ``proposal.target_paths`` is checked
    individually, and the whole patch is denied auto-approval unless all of them
    qualify.

    Targets are read from ``proposal.resolved_target_paths`` when the builder
    populated it (every builder does) — these already ran through
    ``_resolve_edit_path`` (the task's live-cwd-aware resolver, same as the
    preview and the real write), so a RELATIVE target is judged against the
    location the write actually lands on. Falling back to the raw
    ``target_paths``/``path`` (kept only for callers constructing an
    ``EditProposal`` directly, e.g. tests) would instead resolve a relative
    path against this ACP process's own ``os.getcwd()`` — wrong whenever the
    task's live/registered cwd differs, which could misjudge an
    out-of-workspace write as workspace-local and skip the approval prompt.

    ``task_id`` (when given, the same id ``build_edit_proposal`` used) also
    resolves ``cwd`` itself into the same namespace via
    ``_resolve_workspace_boundary`` before comparing — see that function's
    docstring for why a raw client-reported ``cwd`` is not always
    comparable to a task-resolved target.
    """

    policy = str(policy or AUTO_APPROVE_ASK).strip()
    if policy == AUTO_APPROVE_ASK:
        return False
    cwd = _resolve_workspace_boundary(cwd, task_id)
    targets = proposal.resolved_target_paths or proposal.target_paths or (proposal.path,)
    return all(_is_single_path_auto_approvable(target, policy, cwd) for target in targets)


def _denied(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def maybe_require_edit_approval(
    tool_name: str, arguments: dict[str, Any], task_id: str | None = None
) -> str | None:
    """Run ACP edit approval if bound.

    Returns a JSON tool-error string when the edit must be blocked, otherwise
    ``None`` so dispatch can continue.  Requester exceptions deny by default.

    ``task_id`` should be the same id the tool call is about to execute
    under (model_tools.py's dispatch already has it) so the preview reads
    relative paths through the identical live-cwd-aware resolver the real
    write uses -- see ``build_edit_proposal``/``_resolve_edit_path``.
    """
    requester = _EDIT_APPROVAL_REQUESTER.get()
    if requester is None:
        return None
    try:
        proposal = build_edit_proposal(tool_name, arguments, task_id or "default")
    except Exception as exc:
        logger.warning("Could not build ACP edit approval proposal for %s: %s", tool_name, exc)
        return _denied(f"Edit approval denied: could not prepare diff ({exc})")
    if proposal is None:
        return None
    try:
        approved = bool(requester(proposal))
    except Exception as exc:
        logger.warning("ACP edit approval requester failed: %s", exc)
        approved = False
    return None if approved else _denied("Edit approval denied by ACP client; file was not modified.")


def build_acp_edit_tool_call(proposal: EditProposal):
    """Build the ToolCallUpdate payload for ACP request_permission."""
    import acp

    return acp.update_tool_call(
        f"edit-approval-{next(_PERMISSION_REQUEST_IDS)}", title=f"Approve edit: {proposal.path}", kind="edit",
        status="pending",
        content=[acp.tool_diff_content(path=proposal.path, old_text=proposal.old_text, new_text=proposal.new_text)],
        raw_input={"tool": proposal.tool_name, "arguments": proposal.arguments},
    )


def make_acp_edit_approval_requester(
    request_permission_fn: Callable, loop: asyncio.AbstractEventLoop, session_id: str,
    timeout: float = 60.0, auto_approve_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> EditApprovalRequester:
    """Return a sync requester that bridges edit proposals to ACP permissions."""

    def _requester(proposal: EditProposal) -> bool:
        from acp.schema import PermissionOption
        from acp_adapter.permissions import await_permission

        if auto_approve_getter is not None:
            try:
                policy, cwd = auto_approve_getter()
                # session_id doubles as the task id for the root call (see
                # build_edit_proposal's own docstring) -- passing it lets
                # should_auto_approve_edit resolve `cwd` into the same
                # namespace as the proposal's resolved_target_paths.
                if should_auto_approve_edit(proposal, policy, cwd, task_id=session_id):
                    logger.info("Auto-approved ACP edit under policy %s: %s", policy, proposal.path)
                    return True
            except Exception:
                logger.debug("ACP edit auto-approval policy check failed", exc_info=True)

        response, _timed_out = await_permission(
            request_permission_fn, loop, session_id, tool_call=build_acp_edit_tool_call(proposal),
            options=[PermissionOption(option_id="allow_once", kind="allow_once", name="Allow edit"),
                     PermissionOption(option_id="deny", kind="reject_once", name="Deny")],
            timeout=timeout, what="Edit approval request",
        )
        outcome = getattr(response, "outcome", None)
        return getattr(outcome, "outcome", None) == "selected" and getattr(outcome, "option_id", None) == "allow_once"

    return _requester


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from concurrent.futures import TimeoutError as FutureTimeout  # noqa: F401,E402

def clear_edit_approval_requester() -> None:
    """Clear the current requester; primarily used by tests."""

    _EDIT_APPROVAL_REQUESTER.set(None)

def get_edit_approval_requester() -> EditApprovalRequester | None:
    return _EDIT_APPROVAL_REQUESTER.get()
# ---- END PLUGIN-COMPAT ----
