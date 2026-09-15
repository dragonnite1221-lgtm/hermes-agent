"""Pre-execution ACP edit approval helpers.

Intentionally isolated from the generic tool registry: ACP binds an edit
approval requester in a ContextVar for the duration of one ACP agent run; CLI,
gateway, and other sessions leave it unset and therefore bypass this guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

    For a V4A proposal on a non-host (SSH/container/sandbox) backend, an
    entry can be ``None`` instead of a string: ``tools.file_tools
    ._resolve_v4a_policy_target`` returns ``None`` when it cannot compute a
    backend-canonical location for that target (a tilde-prefixed header, or
    no backend cwd at all) -- ``should_auto_approve_edit`` must treat that as
    "cannot verify workspace membership" and deny auto-approval, never fall
    back to raw-string host resolution.
    """

    tool_name: str
    path: str
    old_text: str | None
    new_text: str
    arguments: dict[str, Any]
    target_paths: tuple[str, ...] | None = None
    resolved_target_paths: tuple[str | None, ...] | None = None


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
    resolve: bool = True,
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

    ``resolve=False`` (used only by ``_proposal_for_patch_v4a`` on a
    non-host-paths backend) skips ``_resolve_edit_path`` and hands ``path``
    to the backend exactly as written. This mirrors
    ``tools/file_tools.py``'s ``_rewrite_v4a_patch_paths_for_host``, which
    rewrites V4A patch headers to host-resolved paths ONLY when
    ``_file_ops_uses_host_paths(file_ops)`` is true; for an SSH/container/
    sandbox backend it leaves headers untouched and lets THAT backend's own
    shell resolve them against its own live cwd (``ShellFileOperations._exec``
    always runs with ``cwd=effective_cwd`` from ``self.env.cwd``/``self.cwd``).
    Resolving via ``_resolve_edit_path`` there instead would anchor onto
    ``_resolve_path_for_task``'s notion of the task's cwd (which, before any
    terminal command has run in the session, can still be the ACP client's
    raw HOST workspace path, not the container's actual filesystem
    namespace) -- previewing a path the real V4A apply never even looks at.
    """
    resolved = _resolve_edit_path(path, task_id) if resolve else path

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


def _normalize_new_text_for_preview(old_text: str | None, new_text: str, *, use_byte_window: bool) -> str:
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

    ``use_byte_window`` selects WHICH of the two different real-write code
    paths this proposal kind must match -- they are NOT the same:

    * ``write_file`` (``use_byte_window=True``): ``write_file_tool`` calls
      ``ShellFileOperations.write_file()`` with NO ``pre_content``, so
      ``_probe_write_target()`` detects the line ending from a live
      ``head -c 4096`` probe -- a BYTE window over the ON-DISK file.
      ``old_text`` is therefore run through ``_byte_capped_sample`` before
      ``_detect_line_ending`` sees it: that helper's own ``sample[:4096]``
      is a CHARACTER slice, and for an existing file with enough multibyte
      characters ahead of its first newline (e.g. ~3000 emoji before a
      CRLF) that the newline falls after byte 4096 but before character
      4096, feeding it the full ``old_text`` directly would see a newline
      the real byte-based probe does not.

    * ``patch_replace`` (``use_byte_window=False``): ``patch_replace()``
      already has the FULL file content in hand (its own ``_cat()``, for
      the fuzzy match) and calls ``write_file()`` WITH that content as
      ``pre_content`` -- so ``_probe_write_target()`` takes its
      ``pre_content`` branch instead, detecting the line ending via
      ``_detect_line_ending(pre_content)`` on the FULL (CHARACTER-sliced)
      text, never the byte-based probe. Byte-capping here would instead
      make the preview disagree with patch_replace's real (character-
      based, full-text) decision for the exact same multibyte-heavy fixture
      the write_file case above is byte-capped to match.
    """
    if old_text is None:
        return new_text
    from tools.file_operations_common import _byte_capped_sample, _detect_line_ending, _normalize_line_endings

    sample = _byte_capped_sample(old_text) if use_byte_window else old_text
    file_ending = _detect_line_ending(sample)
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
    new_text = _normalize_new_text_for_preview(old_text, str(content), use_byte_window=True)
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
    # use_byte_window=False: patch_replace()'s real write passes this SAME
    # full content through to write_file() as pre_content, so its probe
    # takes the character-based (not byte-capped) branch -- see
    # _normalize_new_text_for_preview's docstring.
    new_text = _normalize_new_text_for_preview(old_text, new_text, use_byte_window=False)
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

    # Mirror tools/file_tools.py's _rewrite_v4a_patch_paths_for_host: it
    # rewrites V4A headers to host-resolved paths ONLY for a host-paths
    # backend (_file_ops_uses_host_paths); a non-host (SSH/container/
    # sandbox) backend gets the ORIGINAL headers untouched and resolves
    # them itself against its own live cwd. Resolving via
    # _resolve_edit_path unconditionally here would preview (and report as
    # the auto-approval target) a path the real V4A apply never even looks
    # at -- see _read_text_if_exists's docstring on resolve=False.
    from tools.file_tools import _file_ops_uses_host_paths, _get_file_ops, _resolve_v4a_policy_target

    file_ops = _get_file_ops(task_id)
    uses_host_paths = _file_ops_uses_host_paths(file_ops)

    # ACP only supports a single diff payload: surface the exact V4A patch as new_text so
    # patch-mode calls are permissioned and denied patches cannot mutate.
    return EditProposal(
        tool_name="patch",
        path=paths[0] if single else ", ".join(paths),
        # deny_binary=False only for a delete-only operation (never derives
        # new content from old, so #5's overwrite-masking protection does
        # not apply) -- see _read_text_if_exists's docstring and
        # tools/patch_parser.py's _apply_delete.
        old_text=_read_text_if_exists(
            paths[0], task_id, deny_binary=not is_delete_only, resolve=uses_host_paths,
        ) if single else None,
        # ACP only supports a single diff payload here.  Surface the exact V4A
        # patch content before execution so patch-mode calls are permissioned
        # and denied patches cannot mutate.
        new_text=patch_body,
        arguments=dict(arguments),
        # Keep the real per-file targets alongside the joined display string
        # so approval decisions never parse `path` back into a filesystem path.
        target_paths=tuple(paths),
        # Same targets resolved through the task-live-cwd-aware resolver --
        # ONLY on a host-paths backend, matching where the real (rewritten)
        # V4A headers will touch. On a non-host backend the real apply
        # leaves headers untouched and lets THAT backend's shell resolve
        # them against its own live env.cwd, so `_resolve_edit_path` (a HOST
        # resolver) would diverge from reality here -- but the raw string
        # can't be handed to `should_auto_approve_edit` either: comparing a
        # backend-namespace path via host `Path.resolve()` can misclassify
        # an out-of-workspace write as workspace-local (see
        # `_resolve_v4a_policy_target`'s docstring). So a non-host target
        # gets resolved via that SAME no-shell backend-canonical join
        # instead, for the policy check only -- the patch body handed to
        # the backend above is untouched either way.
        resolved_target_paths=(
            tuple(_resolve_edit_path(p, task_id) for p in paths) if uses_host_paths
            else tuple(_resolve_v4a_policy_target(p, file_ops) for p in paths)
        ),
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


def _is_single_path_auto_approvable(
    raw_path: str, policy: str, cwd_candidates: tuple[str | None, ...], resolved_path: str | None = None,
    verify_backend: Callable[[str], bool] | None = None,
) -> bool:
    """``raw_path`` is the ORIGINAL (pre-canonicalization) target, always a
    real string -- used for the sensitive-path guard and, under
    ``AUTO_APPROVE_SESSION`` ("Don't Ask"), that guard is the ONLY check:
    a session-wide accept auto-allows every non-sensitive edit regardless of
    where it lands, so it must not depend on knowing a canonical location.

    ``resolved_path`` is the backend-canonical location used ONLY for the
    ``AUTO_APPROVE_WORKSPACE`` boundary check; it can be ``None`` (a non-host
    V4A target ``tools.file_tools._resolve_v4a_policy_target`` could not
    canonicalize -- a tilde header, or no backend cwd at all), in which case
    that check fails closed (denies auto-approval) rather than guessing --
    but this must NOT also deny ``AUTO_APPROVE_SESSION``, which never
    inspects the boundary in the first place.

    ``cwd_candidates`` (from ``_resolve_workspace_boundary``) holds one or
    more acceptable boundary namespaces: ``resolved_path`` need only fall
    under ANY of them, since a non-host backend's own mounted root (e.g.
    Docker's ``/workspace``) and the client-reported host workspace it maps
    from are the SAME logical workspace even though they are textually
    unrelated paths.

    ``verify_backend`` (only set for a non-host backend -- see
    ``should_auto_approve_edit``) is an ADDITIONAL confirmation gate run
    AFTER the lexical containment check above already passed: every
    resolver that can populate ``resolved_path`` for a non-host backend
    (``tools.file_tools._resolve_v4a_policy_target`` for V4A,
    ``_resolve_path_for_task``'s container-path branch for write_file/
    patch_replace) is a plain string join/normalize with NO knowledge of
    the backend's actual filesystem, so a workspace-internal symlink the
    host cannot see (e.g. ``/workspace/link -> /outside``) can make an
    apparently in-workspace target resolve, once the backend's own shell
    actually applies the write, to a location entirely outside every
    boundary -- silently escaping the workspace without ever prompting.
    Skipped entirely for a host-paths (local) backend, where
    ``Path.resolve()`` above already followed any real symlink on the SAME
    filesystem the write happens on.
    """
    if _is_sensitive_auto_approve_path(raw_path):
        return False
    if policy == AUTO_APPROVE_SESSION:
        return True
    if policy == AUTO_APPROVE_WORKSPACE:
        if resolved_path is None:
            return False
        path = Path(resolved_path).expanduser().resolve(strict=False)
        # tempfile.gettempdir() is the real temp root on every platform
        # (``/private/tmp`` on macOS since resolve() follows the symlink).
        if path.is_relative_to(Path(tempfile.gettempdir()).resolve(strict=False)):
            return True
        in_boundary = any(
            bool(cwd) and path.is_relative_to(Path(cwd).expanduser().resolve(strict=False))
            for cwd in cwd_candidates
        )
        if not in_boundary:
            return False
        if verify_backend is not None and not verify_backend(resolved_path):
            return False
        return True
    return False


def _resolve_workspace_boundary(cwd: str | None, task_id: str | None) -> tuple[str | None, ...]:
    """Return the AUTO_APPROVE_WORKSPACE boundary candidate(s), in the SAME
    namespace(s) ``proposal.resolved_target_paths`` might land in -- WITHOUT
    letting either candidate drift with the task's *live* cwd.

    ``cwd`` (``state.cwd`` at the call sites) is the ACP client's own report
    of the session's ORIGINAL workspace directory, set once at session
    create/load/resume. For a local backend that IS the filesystem
    namespace the write happens in, so comparing it directly against a
    resolved target works. For an SSH/container/sandbox-backed task it need
    not be, since the client-reported cwd can differ from the backend's own
    filesystem view.

    The fix is NOT ``tools.file_tools._resolve_base_dir(task_id)``: that
    function's ``_authoritative_workspace_root`` prefers the task's *live*
    terminal cwd (updated by every ``cd`` the agent runs) over the
    registered session cwd -- exactly right for resolving a relative EDIT
    TARGET (matching where the write actually lands), but wrong for the
    approval BOUNDARY itself, which must stay anchored to the workspace the
    session was configured with. Using the live cwd for both would let an
    agent that ``cd``s OUTSIDE the original workspace silently auto-approve
    every subsequent relative write there, since the target and the
    boundary would always be resolved against the identical (now
    out-of-workspace) live directory.

    The FIRST (always present) candidate reads ONLY the REGISTERED session
    cwd override (``tools.file_tools_paths._registered_task_cwd_override``
    -- what ``acp_adapter/session.py``'s ``_register_task_cwd`` sets at
    session create/load/resume, deliberately skipping the live-cwd tier)
    and runs it through ``_resolve_path_for_task`` for the SAME namespace
    normalization a host-paths-backend resolved target gets. When
    ``task_id`` is unavailable (e.g. an ``EditProposal`` built by hand in a
    test), no override is registered, or the lookup fails, this candidate
    falls back to the given ``cwd`` unchanged.

    A SECOND candidate is added only for a Docker task whose
    ``docker_mount_cwd_to_workspace`` feature bind-mounts EXACTLY this
    registered workspace to a fixed in-container path (``config["cwd"]``,
    normally ``/workspace``) -- e.g. a V4A target on that backend resolves
    (via ``tools.file_tools._resolve_v4a_policy_target``) to
    ``/workspace/x``, which the first (host-style) candidate above can never
    match, since a client-reported host directory like ``/Users/me/project``
    shares no path relationship with the container's own mount point.
    ``tools.terminal_tool._resolve_task_host_cwd`` is the single source of
    truth for which host directory (if any) was actually mounted for this
    task; re-deriving the mapped candidate from that STATIC config value
    (rather than trusting the backend's LIVE ``env.cwd``, which an agent's
    own ``cd`` can move away from the mount root) keeps the same
    live-cwd-drift protection the first candidate already has, and only
    adds this candidate when the mount source provably IS the registered
    workspace -- anything else (SSH, an unconfigured Docker task, a
    mismatched host dir) is left to fail closed on the first candidate
    alone, exactly like every other "cannot verify" case in this module.
    """
    if not task_id:
        return (cwd,)
    root: str | None = None
    candidates: list[str | None] = []
    try:
        from tools.file_tools import _resolve_path_for_task
        from tools.file_tools_paths import _registered_task_cwd_override

        root = _registered_task_cwd_override(task_id)
        candidates.append(str(_resolve_path_for_task(root, task_id)) if root else cwd)
    except Exception:
        candidates.append(cwd)
    try:
        from tools.terminal_tool import _get_env_config, _resolve_task_host_cwd

        config = _get_env_config()
        host_cwd = _resolve_task_host_cwd(config, task_id)
        if root and host_cwd and os.path.normpath(os.path.expanduser(root)) == os.path.normpath(host_cwd):
            candidates.append(str(config.get("cwd")))
    except Exception:
        pass
    return tuple(candidates)


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

    A ``None`` entry in ``resolved_target_paths`` (only ``tools.file_tools
    ._resolve_v4a_policy_target`` produces one, for a non-host V4A target it
    could not canonicalize against the backend's own cwd) fails ONLY the
    ``AUTO_APPROVE_WORKSPACE`` boundary check for that target closed, rather
    than falling back to the raw string — comparing an un-canonicalized
    backend-namespace path via host ``Path.resolve()`` is exactly the
    false-workspace-local misclassification this function exists to prevent.
    It does not affect ``AUTO_APPROVE_SESSION`` ("Don't Ask"), which never
    inspects the workspace boundary — a session-wide accept still
    auto-allows a non-sensitive edit whose canonical location is unknown;
    see ``_is_single_path_auto_approvable``.

    For a NON-HOST (SSH/container/sandbox) backend, a target that lexically
    looks workspace-local is additionally verified via a live backend-side
    ``readlink -f`` round-trip (``tools.file_tools._verify_realpath_within_any``)
    before ``AUTO_APPROVE_WORKSPACE`` skips the prompt: every resolver that
    can produce a ``resolved_target_paths`` entry for such a backend is a
    plain string join with no knowledge of the backend's actual filesystem,
    so a workspace-internal symlink the host cannot see could otherwise
    escape the workspace unprompted once the backend's own shell follows it
    for real. This is skipped for a host-paths (local) backend, where
    ``Path.resolve()`` already followed any real symlink on the SAME
    filesystem the write happens on.
    """

    policy = str(policy or AUTO_APPROVE_ASK).strip()
    if policy == AUTO_APPROVE_ASK:
        return False
    cwd_candidates = _resolve_workspace_boundary(cwd, task_id)
    verify_backend = None
    if task_id:
        try:
            from tools.file_tools import _file_ops_uses_host_paths, _get_file_ops, _verify_realpath_within_any

            file_ops = _get_file_ops(task_id)
            if not _file_ops_uses_host_paths(file_ops):
                verify_backend = lambda resolved, _fo=file_ops: _verify_realpath_within_any(
                    resolved, cwd_candidates, _fo)
        except Exception:
            # Could not even determine the backend type for this task --
            # fail closed rather than silently skipping the symlink check.
            verify_backend = lambda _resolved: False
    raw_targets = proposal.target_paths or (proposal.path,)
    resolved_targets = proposal.resolved_target_paths
    if resolved_targets is None:
        resolved_targets = raw_targets
    elif len(resolved_targets) != len(raw_targets):
        # Builder invariant violated (every real builder keeps these the
        # same length) -- don't guess an alignment, fail every target's
        # workspace check closed instead of pairing the wrong entries.
        resolved_targets = (None,) * len(raw_targets)
    return all(
        _is_single_path_auto_approvable(raw, policy, cwd_candidates, resolved, verify_backend)
        for raw, resolved in zip(raw_targets, resolved_targets)
    )


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
