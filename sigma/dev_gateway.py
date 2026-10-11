"""Small local broker: one file, no shell, no execution of proposed code.

The model-facing surface only supports read, propose and test. Applying a
proposal is available exclusively through the separate interactive CLI.
This is an application boundary, not isolation against hostile OS processes.
"""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import secrets
import time


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def redirected(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


class Gateway:
    allowed_file = 'calculator.py'

    def __init__(self, workspace, state):
        self.workspace = Path(workspace).absolute()
        self.state = Path(state).absolute()
        if not self.workspace.is_dir() or redirected(self.workspace):
            raise ValueError('Workspace must be an existing ordinary directory.')
        if self.state.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.state):
            raise ValueError('Audit state must be outside the workspace.')
        self.state.mkdir(parents=True, exist_ok=True)
        self._path()

    def audit(self, operation, decision, **details):
        record = {'time': time.time(), 'operation': operation, 'decision': decision, **details}
        with (self.state / 'audit.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _path(self, name='calculator.py'):
        if name != self.allowed_file:
            self.audit('path', 'deny', requested=str(name))
            raise PermissionError('Only calculator.py is available.')
        path = self.workspace / name
        if (self.workspace.resolve() != self.workspace or redirected(self.workspace)
                or not path.is_file() or redirected(path)
                or path.resolve().parent != self.workspace or path.stat().st_nlink != 1):
            raise PermissionError('Links, redirects and missing files are denied.')
        if path.stat().st_size > 8192:
            raise ValueError('File size limit exceeded.')
        return path

    def read(self, name='calculator.py'):
        content = self._path(name).read_text(encoding='utf-8')
        self.audit('read', 'allow', file=name, sha256=digest(content))
        return content

    def propose(self, content):
        if not isinstance(content, str) or len(content.encode('utf-8')) > 8192 or '\x00' in content:
            raise ValueError('Invalid replacement.')
        original = self.read()
        proposal_id = secrets.token_hex(16)
        proposal = {'id': proposal_id, 'file': self.allowed_file, 'created': time.time(),
                    'base': digest(original), 'replacement': content, 'replacement_hash': digest(content),
                    'diff': ''.join(difflib.unified_diff(original.splitlines(True), content.splitlines(True),
                                                      fromfile='a/calculator.py', tofile='b/calculator.py'))}
        (self.state / (proposal_id + '.json')).write_text(json.dumps(proposal, indent=2), encoding='utf-8')
        self.audit('propose', 'pending_human', id=proposal_id, base=proposal['base'], replacement_hash=proposal['replacement_hash'])
        return proposal

    def proposal(self, proposal_id):
        if len(proposal_id) != 32 or any(c not in '0123456789abcdef' for c in proposal_id):
            raise ValueError('Invalid proposal identifier.')
        proposal = json.loads((self.state / (proposal_id + '.json')).read_text(encoding='utf-8'))
        if proposal['id'] != proposal_id or proposal['file'] != self.allowed_file or digest(proposal['replacement']) != proposal['replacement_hash']:
            raise PermissionError('Invalid proposal integrity.')
        return proposal

    def test(self, proposal_id=None):
        """Fixed AST check. Never import, eval, exec, spawn or run model code."""
        content = self.proposal(proposal_id)['replacement'] if proposal_id else self.read()
        try:
            tree = ast.parse(content)
            contracts = {ast.dump(ast.parse('def add(a, b):\n    return a ' + op + ' b\n')): op for op in ('+', '-')}
            operation = contracts.get(ast.dump(tree))
            if operation is None:
                raise ValueError('Outside AST contract.')
            results = [{'inputs': [a, b], 'expected': a + b,
                        'actual': a + b if operation == '+' else a - b}
                       for a, b in [(2, 3), (-2, 5), (0, 0)]]
            result = {'status': 'PASS' if all(r['actual'] == r['expected'] for r in results) else 'FAIL', 'cases': results}
        except (SyntaxError, ValueError, RecursionError):
            result = {'status': 'INCONCLUSIVE', 'reason': 'Outside the predefined AST test contract; no code executed.'}
        self.audit('test', result['status'], id=proposal_id)
        return result

    def dispatch(self, request):
        """Only this method is exposed to model proposals; approval is absent."""
        if not isinstance(request, dict):
            self.audit('request', 'deny', reason='not_an_object')
            raise ValueError('Request must be an object.')
        self.audit('request', 'received', operation_requested=str(request.get('operation')))
        try:
            result = self._dispatch(request)
            self.audit('request', 'complete', operation_requested=str(request.get('operation')))
            return result
        except Exception as error:
            self.audit('request', 'deny', reason=type(error).__name__)
            raise

    def _dispatch(self, request):
        operation = request.get('operation')
        if operation == 'read' and set(request) <= {'operation', 'file'}:
            return {'content': self.read(request.get('file', self.allowed_file))}
        if operation == 'propose' and set(request) == {'operation', 'content'}:
            return self.propose(request['content'])
        if operation == 'test' and set(request) <= {'operation', 'proposal_id'}:
            return self.test(request.get('proposal_id'))
        self.audit('request', 'deny', operation_requested=str(operation))
        raise PermissionError('Capability does not grant authority. This operation is not exposed.')

    def human_apply(self, proposal_id, input_stream, output_stream):
        """Separate operator channel. A model request cannot supply approval."""
        if not input_stream.isatty() or not output_stream.isatty():
            self.audit('approve', 'deny', id=proposal_id, reason='noninteractive')
            raise PermissionError('Approval requires a human interactive terminal.')
        lock = self.state / 'human-approval.lock'
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self.audit('approve', 'deny', id=proposal_id, reason='another_review_or_unresolved_crash')
            raise PermissionError('Another review is active, or a previous review needs operator recovery.')
        os.close(descriptor)
        try:
            return self._review_and_apply(proposal_id, input_stream, output_stream)
        finally:
            lock.unlink()

    def _review_and_apply(self, proposal_id, input_stream, output_stream):
        proposal = self.proposal(proposal_id)
        receipt = self.state / (proposal_id + '.approved.json')
        if receipt.exists():
            raise PermissionError('Approval has already been consumed.')
        original = self.read()
        if digest(original) != proposal['base'] or not 0 <= time.time() - proposal['created'] <= 600:
            self.audit('approve', 'deny', id=proposal_id, reason='stale_or_expired')
            raise PermissionError('Proposal is stale or expired.')
        expected_diff = ''.join(difflib.unified_diff(original.splitlines(True), proposal['replacement'].splitlines(True),
                                                   fromfile='a/calculator.py', tofile='b/calculator.py'))
        if proposal['diff'] != expected_diff:
            raise PermissionError('Diff integrity mismatch.')
        output_stream.write(expected_diff + '\nType APPLY ' + proposal_id + ' to authorize exactly this change: ')
        output_stream.flush()
        if input_stream.readline().strip() != 'APPLY ' + proposal_id:
            self.audit('approve', 'deny', id=proposal_id, reason='human_declined')
            return {'status': 'DENIED'}
        if self.proposal(proposal_id) != proposal or digest(self.read()) != proposal['base'] or time.time() - proposal['created'] > 600:
            raise PermissionError('Content changed during review.')
        self.audit('approve', 'allow', id=proposal_id, replacement_hash=proposal['replacement_hash'])
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump({'id': proposal_id, 'replacement_hash': proposal['replacement_hash'], 'time': time.time()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        (self.state / (proposal_id + '.backup.txt')).write_text(original, encoding='utf-8')
        target = self._path()
        # The only target write occurs after the interactive approval above.
        pending = self.state / (proposal_id + '.pending.txt')
        with pending.open('x', encoding='utf-8', newline='') as stream:
            stream.write(proposal['replacement'])
            stream.flush()
            os.fsync(stream.fileno())
        self._path()
        os.replace(pending, target)
        self.audit('apply', 'complete', id=proposal_id, sha256=digest(target.read_text(encoding='utf-8')))
        return {'status': 'APPLIED', 'id': proposal_id}
