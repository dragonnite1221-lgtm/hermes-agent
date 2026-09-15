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
    single-target proposal kinds.
    """

    tool_name: str
    path: str
    old_text: str | None
    new_text: str
    arguments: dict[str, Any]
    target_paths: tuple[str, ...] | None = None


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
    """
    if Path(path).is_absolute():
        return path

    from tools.file_tools import _resolve_path_for_task

    return str(_resolve_path_for_task(path, task_id))


def _read_text_if_exists(path: str, task_id: str = "default") -> str | None:
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
    """
    resolved = _resolve_edit_path(path, task_id)

    from tools.file_tools import _get_file_ops

    result = _get_file_ops(task_id).read_file_raw(resolved)
    if result.is_binary or result.is_image:
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
            # NOTE: this prefix is not a perfectly reliable "confirmed
            # absent" signal by itself. It comes from
            # ShellFileOperations._suggest_similar_files(), reached
            # whenever read_file_raw()'s existence probe command exits
            # non-zero -- which is also what happens if the probe itself
            # fails to run at all (a dropped SSH connection, a container
            # that died, ...), not only when the shell's own `[ -e ... ]`
            # check says the path is missing. That ambiguity is inherent
            # to read_file_raw()'s exit-code contract and pre-dates this
            # module: write_file_tool/patch_tool read through the exact
            # same function and have the same blind spot before writing.
            # Fixing it for real means giving read_file_raw() a way to
            # report "could not determine existence" distinctly from
            # "confirmed absent", which is a change to the shared
            # tools.file_operations backend, not something to smuggle into
            # this preview-only module.
            return None
        raise OSError(f"Cannot read current content of {path!r}: {result.error}")
    return result.content


def _required_path(arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path") or "")
    if not path:
        raise ValueError("path required")
    return path


def _proposal_for_write_file(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    path = _required_path(arguments)
    content = arguments.get("content")
    if content is None:
        raise ValueError("content required")
    return EditProposal("write_file", path, _read_text_if_exists(path, task_id), str(content), dict(arguments))


def _proposal_for_patch_replace(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    path = _required_path(arguments)
    old_string, new_string = arguments.get("old_string"), arguments.get("new_string")
    if old_string is None or new_string is None:
        raise ValueError("old_string and new_string required")
    old_text = _read_text_if_exists(path, task_id)
    if old_text is None:
        raise ValueError(f"Failed to read file: {path}")

    from tools.fuzzy_match import fuzzy_find_and_replace

    new_text, match_count, _strategy, error = fuzzy_find_and_replace(
        old_text, str(old_string), str(new_string), bool(arguments.get("replace_all", False)))
    if error or match_count == 0:
        raise ValueError(error or f"Could not find match for old_string in {path}")
    return EditProposal("patch", path, old_text, new_text, dict(arguments))


def _extract_v4a_patch_paths(patch_body: str) -> list[str]:
    # Reuse the same parser that actually executes the patch (tools/
    # patch_parser.py, via tools/file_operations.py) instead of a second,
    # independently-maintained regex: a prior version of this function had
    # its own `\s+`-after-`***` regex that was stricter than the parser's
    # `\s*`, so a no-space header (`***Update File:`) that the parser still
    # executed could slip past approval extraction entirely, letting an
    # out-of-workspace target hide behind an in-workspace one. Deriving the
    # paths from the real parser makes that class of drift impossible.
    from tools.patch_parser import parse_v4a_patch

    operations, _error = parse_v4a_patch(patch_body)
    paths: list[str] = []
    for op in operations:
        if op.file_path:
            paths.append(op.file_path)
        if op.new_path:
            paths.append(op.new_path)
    return paths


def _proposal_for_patch_v4a(arguments: dict[str, Any], task_id: str = "default") -> EditProposal:
    patch_body = arguments.get("patch")
    if not isinstance(patch_body, str) or not patch_body:
        raise ValueError("patch content required")
    paths = _extract_v4a_patch_paths(patch_body)
    if not paths:
        raise ValueError("no file paths found in V4A patch")
    single = len(paths) == 1
    # ACP only supports a single diff payload: surface the exact V4A patch as new_text so
    # patch-mode calls are permissioned and denied patches cannot mutate.
    return EditProposal(
        tool_name="patch",
        path=paths[0] if single else ", ".join(paths),
        old_text=_read_text_if_exists(paths[0], task_id) if single else None,
        # ACP only supports a single diff payload here.  Surface the exact V4A
        # patch content before execution so patch-mode calls are permissioned
        # and denied patches cannot mutate.
        new_text=patch_body,
        arguments=dict(arguments),
        # Keep the real per-file targets alongside the joined display string
        # so approval decisions never parse `path` back into a filesystem path.
        target_paths=tuple(paths),
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


def should_auto_approve_edit(proposal: EditProposal, policy: str, cwd: str | None = None) -> bool:
    """Return whether an ACP edit proposal may bypass the prompt for this session.

    This is intentionally session-scoped and conservative: sensitive paths still
    ask even under autonomous policies. For multi-file V4A patches, ``proposal.path``
    is only a comma-joined display string and must never be parsed back into a
    filesystem path — every real target in ``proposal.target_paths`` is checked
    individually, and the whole patch is denied auto-approval unless all of them
    qualify.
    """

    policy = str(policy or AUTO_APPROVE_ASK).strip()
    if policy == AUTO_APPROVE_ASK:
        return False
    targets = proposal.target_paths or (proposal.path,)
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
                if should_auto_approve_edit(proposal, policy, cwd):
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
