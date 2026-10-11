import io
import json
import pytest
from sigma.dev_gateway import Gateway


@pytest.fixture
def broker(tmp_path):
    workspace = tmp_path / 'sandbox'
    workspace.mkdir()
    (workspace / 'calculator.py').write_text('def add(a, b):\n    return a - b\n', encoding='utf-8')
    return Gateway(workspace, tmp_path / 'state')


@pytest.mark.parametrize('payload', [{'operation': 'apply'}, {'operation': 'approve', 'approved': True},
                                    {'operation': 'shell', 'command': 'whoami'}, {'operation': 'delete'},
                                    {'operation': 'push'}, {'operation': 'merge'},
                                    {'operation': 'read', 'file': '../secret'},
                                    {'operation': 'propose', 'content': '', 'approved': True}])
def test_model_cannot_authorize_or_escape(broker, payload):
    before = broker.read()
    with pytest.raises(PermissionError):
        broker.dispatch(payload)
    assert broker.read() == before


def test_diff_tests_do_not_write_and_noninteractive_approval_denied(broker):
    before = broker.read()
    proposal = broker.propose('def add(a, b):\n    return a + b\n')
    assert '-    return a - b' in proposal['diff']
    assert broker.test()['status'] == 'FAIL'
    assert broker.test(proposal['id'])['status'] == 'PASS'
    with pytest.raises(PermissionError):
        broker.human_apply(proposal['id'], io.StringIO('APPLY ' + proposal['id']), io.StringIO())
    assert broker.read() == before
    assert 'noninteractive' in (broker.state / 'audit.jsonl').read_text()


def test_no_execution_of_arbitrary_python(broker):
    proposal = broker.propose('import os\nos.remove("calculator.py")\n')
    assert broker.test(proposal['id'])['status'] == 'INCONCLUSIVE'
    assert broker.read()


def test_tampered_proposal_denied(broker):
    proposal = broker.propose('def add(a, b):\n    return a + b\n')
    path = broker.state / (proposal['id'] + '.json')
    proposal['replacement'] = 'other'
    path.write_text(json.dumps(proposal))
    with pytest.raises(PermissionError):
        broker.proposal(proposal['id'])


class TestTerminal(io.StringIO):
    __test__ = False

    def isatty(self):
        return True


def test_operator_gate_apply_and_backup_in_temporary_fixture(broker):
    original = broker.read()
    proposal = broker.propose('def add(a, b):\n    return a + b\n')
    # Simulated terminal is restricted to this software test, never a live approval.
    result = broker.human_apply(proposal['id'], TestTerminal('APPLY ' + proposal['id'] + '\n'), TestTerminal())
    assert result['status'] == 'APPLIED'
    assert broker.test()['status'] == 'PASS'
    assert (broker.state / (proposal['id'] + '.backup.txt')).read_text() == original
    with pytest.raises(PermissionError):
        broker.human_apply(proposal['id'], TestTerminal('APPLY ' + proposal['id']), TestTerminal())


def test_declined_approval_preserves_target(broker):
    original = broker.read()
    proposal = broker.propose('def add(a, b):\n    return a + b\n')
    assert broker.human_apply(proposal['id'], TestTerminal('no\n'), TestTerminal())['status'] == 'DENIED'
    assert broker.read() == original


def test_expired_approval_preserves_target(broker, monkeypatch):
    import sigma.dev_gateway as module
    original = broker.read()
    proposal = broker.propose('def add(a, b):\n    return a + b\n')
    monkeypatch.setattr(module.time, 'time', lambda: proposal['created'] + 601)
    with pytest.raises(PermissionError):
        broker.human_apply(proposal['id'], TestTerminal('APPLY ' + proposal['id']), TestTerminal())
    assert broker.read() == original


def test_hard_link_denied(broker, tmp_path):
    import os
    os.link(broker.workspace / 'calculator.py', tmp_path / 'linked.py')
    with pytest.raises(PermissionError):
        broker.read()
