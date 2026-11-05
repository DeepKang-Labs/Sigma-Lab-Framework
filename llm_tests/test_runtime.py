import importlib
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    for name in ('CONFIGS', 'STATE', 'REPORTS', 'OUTPUTS'):
        monkeypatch.setenv(f'SIGMA_{name}_DIR', str(tmp_path / name.lower()))
    monkeypatch.setenv('SIGMA_ENABLE_JUDGE', '0')
    monkeypatch.setenv('SIGMA_ENABLE_RAG', '0')
    import sigma_llm_complete
    sl = importlib.reload(sigma_llm_complete)
    local = tmp_path / 'model'
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'hello': 2, 'user': 3,
                                 'assistant': 4, ':': 5, 'answer': 6}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}assistant: "
    tokenizer.save_pretrained(local)
    torch.manual_seed(5)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=32, n_embd=16,
                                     n_layer=1, n_head=2, eos_token_id=1, pad_token_id=1))
    model.save_pretrained(local)
    return sl, sl.SigmaLLM(str(local), local_files_only=True)


def test_real_local_model_generates_and_persists_roles(runtime):
    sl, agent = runtime
    output = agent.generate('hello', max_new_tokens=4)
    assert isinstance(output, str) and output
    assert agent.conv.as_messages() == [{'role': 'user', 'content': 'hello'},
                                        {'role': 'assistant', 'content': output}]
    report = json.loads(Path(sl.LAST_REPORT).read_text(encoding='utf-8'))
    assert report['output'] == output
    assert report['evaluation_scope']['judge_enabled'] is False
    assert not report['invariants']['errors']
    agent.generate('answer', max_new_tokens=4)
    assert [m['role'] for m in agent.conv.as_messages()] == ['user', 'assistant', 'user', 'assistant']


def test_prompt_once_and_context_reserves_output(runtime):
    sl, agent = runtime
    inputs = agent._build_inputs('hello', max_new_tokens=4)
    assert inputs['input_ids'][0].tolist().count(2) == 1
    assert agent._build_inputs('hello ' * 100, 4)['input_ids'].shape[-1] == 28
    with pytest.raises(ValueError):
        agent._build_inputs('hello', 32)
    assert agent._clean_output('An assistant can help.') == 'An assistant can help.'


def test_legacy_memory_roles_and_failed_generation(runtime, monkeypatch):
    sl, agent = runtime
    agent.conv.add('human', 'old prompt')
    agent.conv.add('ai', 'old answer')
    assert [m['role'] for m in agent.conv.as_messages()] == ['user', 'assistant']
    original = agent.conv.as_messages()
    def fail(**kwargs):
        raise RuntimeError('generation failed')
    monkeypatch.setattr(agent.model, 'generate', fail)
    with pytest.raises(RuntimeError, match='generation failed'):
        agent.generate('hello', 4)
    assert agent.conv.as_messages() == original


@pytest.mark.parametrize('prompt,tokens', [('', 4), ('hello', 0), ('hello', True)])
def test_invalid_generation_arguments(runtime, prompt, tokens):
    sl, agent = runtime
    with pytest.raises(ValueError):
        agent.generate(prompt, tokens)
    assert not agent.conv.as_messages()


def test_partial_parameters_and_judge_prompt(runtime):
    sl, agent = runtime
    Path(sl.PARAMS_FILE).write_text(json.dumps({'homeostasis': {'temp': {'max': 1.1}}}), encoding='utf-8')
    params = sl.load_sigma_params()
    assert params['homeostasis']['temp']['min'] == 0.6
    assert sl.DEFAULT_SIGMA_PARAMS['homeostasis']['temp']['max'] == 1.2
    from sigma.core.judge import JUDGE_PROMPT_FACT, JUDGE_PROMPT_COH
    assert '"score"' in JUDGE_PROMPT_FACT.format(prompt='hello', output='answer', evidence='none')
    assert '"score"' in JUDGE_PROMPT_COH.format(output='answer')


def test_corrupt_memory_is_preserved(runtime):
    sl, agent = runtime
    Path(sl.CONV_FILE).write_text('{broken', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        sl.ConversationMemory(10)
    assert Path(sl.CONV_FILE).read_text(encoding='utf-8') == '{broken'


def test_invalid_weights_and_nonfinite_metrics(runtime):
    sl, agent = runtime
    for weights in ([0, 0, 0, 0], [float('nan'), 1, 1, 1], [-1, 1, 1, 1]):
        with pytest.raises(ValueError):
            sl.ObjectivityEngine(weights)
    assert sum(sl.ObjectivityEngine([1, 1, 1, 1]).w) == 1
    metrics = sl.SigmaMetrics(0, float('nan'), 0, .5, 0, .5, .9, .9)
    assert sl.Invariants().check(metrics)['errors']


def test_output_decodes_only_new_tokens(runtime, monkeypatch):
    sl, agent = runtime
    def fixed_generation(**kwargs):
        prompt = kwargs['input_ids']
        return torch.cat((prompt, torch.tensor([[6]], device=prompt.device)), dim=1)
    monkeypatch.setattr(agent.model, 'generate', fixed_generation)
    assert agent.generate('hello', 4) == 'answer'


def test_legacy_prompt_keeps_current_message(runtime):
    sl, agent = runtime
    agent.tokenizer.chat_template = None
    inputs = agent._build_inputs('hello', 4)
    assert inputs['input_ids'][0].tolist().count(2) == 1
