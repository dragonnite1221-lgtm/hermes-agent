"""V4A mutations preserve link targets and refuse existing binary Add targets."""
import json

import pytest

from tools import file_tools, terminal_tool
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def local_patch(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    decoy = tmp_path / 'decoy'
    decoy.mkdir()
    monkeypatch.setattr(terminal_tool, '_task_env_overrides', {})
    terminal_tool.register_task_env_overrides('file-safety', {'cwd': str(workspace)})
    ops = ShellFileOperations(LocalEnvironment(cwd=str(decoy)))
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda task_id='default': ops)
    def apply(body):
        return json.loads(registry.dispatch('patch',
            {'mode': 'patch', 'patch': f'*** Begin Patch\n{body}\n*** End Patch'},
            task_id='file-safety'))
    return workspace, apply, ops


@pytest.mark.parametrize('operation', ['Delete', 'Move'])
@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('update', [None, 'valid', 'invalid'])
def test_entry_operations_preserve_symlink_target(local_patch, operation, reuse, update):
    workspace, apply, _ = local_patch
    parent = workspace / 'actual'
    parent.mkdir()
    (workspace / 'parent').symlink_to(parent, target_is_directory=True)
    target = parent / 'real.json'
    original = '{"value": 1}\n'
    target.write_text(original, encoding='utf-8')
    alias = parent / 'alias.txt'
    alias.symlink_to('real.json')
    body = f'*** {operation} File: parent/./alias.txt'
    if operation == 'Move':
        body += ' -> parent/moved.txt'
    if reuse:
        body += '\n*** Add File: parent/alias.txt\n+replacement'
    if update:
        new = '{"value": 2}' if update == 'valid' else 'invalid JSON'
        body = f'*** Update File: parent/alias.txt\n@@\n-{original.rstrip()}\n+{new}\n' + body
    result = apply(body)
    if update == 'invalid':
        assert result.get('error'), result
        # Syntax errors can occur during apply; the content write must never
        # bypass the referent's .json validation because the alias is .txt.
        assert target.read_text(encoding='utf-8') == original
        return
    assert not result.get('error'), result
    expected = '{"value": 2}\n' if update else original
    assert target.read_text(encoding='utf-8') == expected
    assert not alias.is_symlink()
    if reuse:
        assert alias.read_text(encoding='utf-8') == 'replacement'
    assert str(alias) in result['files_modified']
    if operation == 'Move':
        assert (parent / 'moved.txt').is_symlink()
        assert (parent / 'moved.txt').readlink() == target.relative_to(parent)


def test_host_approval_describes_the_deleted_entry(local_patch):
    from acp_adapter.edit_approval import build_edit_proposal

    workspace, _, _ = local_patch
    target = workspace / 'real.txt'
    target.write_text('original', encoding='utf-8')
    alias = workspace / 'alias.txt'
    alias.symlink_to(target)
    proposal = build_edit_proposal('patch', {
        'mode': 'patch', 'patch': '*** Delete File: alias.txt\n',
    }, 'file-safety')
    assert proposal.resolved_target_paths == (str(alias),)
    assert proposal.target_dereference_final == (False,)


@pytest.mark.parametrize('phase', ['validation', 'apply', 'read_error'])
def test_add_preserves_existing_binary_and_validation_is_atomic(local_patch, monkeypatch, phase):
    from tools.file_operations_common import ReadResult

    workspace, apply, ops = local_patch
    original = b'cache\x00\xff\xfe\x00bytes'
    target = workspace / 'cache.dat'
    target.write_bytes(original)
    read = ops.read_file_raw
    calls = 0
    def read_with_race(path, **kwargs):
        nonlocal calls
        calls += 1
        if phase == 'read_error':
            return ReadResult(error='Failed to read file: Permission denied')
        if calls == 1:
            return ReadResult(error=f'File not found: {path}')
        return read(path, **kwargs)
    if phase != 'validation':
        monkeypatch.setattr(ops, 'read_file_raw', read_with_race)
    body = '*** Add File: cache.dat\n+replacement'
    if phase == 'validation':
        body = '*** Add File: new.txt\n+new\n' + body
    result = apply(body)
    assert result.get('error'), result
    assert target.read_bytes() == original
    assert not (workspace / 'new.txt').exists()
