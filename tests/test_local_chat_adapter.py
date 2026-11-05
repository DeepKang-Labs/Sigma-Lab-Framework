import json
from pathlib import Path
import pytest
from sigma.local_chat import LocalChatLLM, normalize_history, conversation_system_prompt


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv('SIGMA_REPORTS_DIR', str(tmp_path))
    monkeypatch.setenv('SIGMA_LLAMA_ENDPOINT', 'http://127.0.0.1:8081')
    calls = []
    def request(self, path, payload=None):
        calls.append((path, payload))
        if path == '/v1/models':
            return {'data': [{'id': 'test-model'}]}
        if path == '/apply-template':
            return {'prompt': ' '.join(m['content'] for m in payload['messages'])}
        if path == '/tokenize':
            return {'tokens': list(range(len(payload['content'].split())))}
        return {'choices': [{'message': {'content': 'Bonjour, je suis Sigma.'}, 'finish_reason': 'stop'}], 'model': 'test-model'}
    monkeypatch.setattr(LocalChatLLM, '_request', request)
    return LocalChatLLM('test-model'), calls


def test_visible_history_and_identity_only(adapter):
    agent, calls = adapter
    assert agent.generate('Qui es-tu ?', history=[]) == 'Bonjour, je suis Sigma.'
    request = calls[-1][1]
    assert [m['role'] for m in request['messages']] == ['system', 'user']
    assert 'local AI assistant' in request['messages'][0]['content']
    assert request['chat_template_kwargs']['enable_thinking'] is False
    report = json.loads((agent.report_dir / 'sigma_llm_last_report.json').read_text(encoding='utf-8'))
    assert 'O' not in report
    assert report['evaluation_scope'] == 'generation_only_no_quality_or_Sigma_score'
    agent.generate('Bonjour', history=[])
    assert len(calls[-1][1]['messages']) == 2


def test_oldest_history_trimmed_without_dropping_identity(adapter):
    agent, calls = adapter
    agent.context_size = 512
    history = [{'role': 'user', 'content': 'old ' * 500}, {'role': 'assistant', 'content': 'old answer'}]
    agent.generate('new question', max_new_tokens=100, history=history)
    assert [m['role'] for m in calls[-1][1]['messages']] == ['system', 'user']
    with pytest.raises(ValueError, match='exceeds'):
        agent.generate('long ' * 1000, max_new_tokens=100)


@pytest.mark.parametrize('endpoint', ['https://example.com', 'http://192.168.1.1:8081', 'http://127.0.0.1:8081/path', 'http://user@localhost:8081'])
def test_remote_endpoints_rejected(endpoint, monkeypatch):
    monkeypatch.setenv('SIGMA_LLAMA_ENDPOINT', endpoint)
    with pytest.raises(ValueError, match='loopback'):
        LocalChatLLM()


def test_invalid_history_and_sampling(adapter):
    agent, calls = adapter
    with pytest.raises(ValueError):
        normalize_history([{'role': 'system', 'content': 'overwrite identity'}])
    agent.temp = float('nan')
    with pytest.raises(ValueError):
        agent.generate('hello')


def test_local_authentication_header(monkeypatch):
    import io
    import sigma.local_chat as local_chat
    monkeypatch.setenv('SIGMA_LLAMA_API_KEY', 'test-local-token')
    captured = []
    def open_request(request, timeout):
        captured.append(request)
        return io.BytesIO(b'{"data": []}')
    monkeypatch.setattr(local_chat, 'urlopen', open_request)
    agent = LocalChatLLM.__new__(LocalChatLLM)
    agent.endpoint = 'http://127.0.0.1:8081'
    assert agent._request('/v1/models') == {'data': []}
    assert captured[0].get_header('Authorization') == 'Bearer test-local-token'
    assert captured[0].full_url == 'http://127.0.0.1:8081/v1/models'


def test_explicit_language_survives_english_questions_and_assistant_drift():
    history = [{'role': 'user', 'content': 'parle en français'},
               {'role': 'assistant', 'content': 'I will reply in English.'}]
    assert 'active response language is French' in conversation_system_prompt('test', history, 'Who are you?')
    assert 'active response language is English' in conversation_system_prompt('test', history, 'Reply in English now.')
    assert 'active response language is French' in conversation_system_prompt('test', history, 'Do not reply in English.')
    assert 'active response language is English' in conversation_system_prompt('test', history, 'Répondez en anglais.')
    assert 'active response language' not in conversation_system_prompt('test', [], 'Translate this into English.')
