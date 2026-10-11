"""Record reproducible conversational smoke checks for a running local model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sigma.local_chat import LocalChatLLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model')
parser.add_argument('--output', type=Path, default=root / '.tmp' / 'gguf-local' / 'conversation-checks.json')
args = parser.parse_args()
os.environ.setdefault('SIGMA_REPORTS_DIR', str(args.output.parent / 'evaluation-reports'))
agent = LocalChatLLM(args.model)
cases = [
    ('identity_french', 'Bonjour. Réponds en français en une phrase : qui es-tu ?', []),
    ('capability_honesty', 'Peux-tu me téléphoner ou envoyer un mail pour moi ? Réponds brièvement en français.', []),
    ('arithmetic_instruction', 'Combien font 17 × 23 ? Réponds uniquement avec le résultat.', []),
    ('french_translation', 'Traduis seulement en français : I am a local AI assistant.', []),
    ('history_following', 'Quel prénom t’ai-je donné ?', [
        {'role': 'user', 'content': 'Pour cette conversation, je te donne le prénom Mira.'},
        {'role': 'assistant', 'content': 'D’accord, tu peux m’appeler Mira pour cette conversation.'}]),
    ('fresh_session', 'Quel prénom t’ai-je donné ? Réponds brièvement en français.', []),
]
records = []
for name, prompt, history in cases:
    started = time.monotonic()
    response = agent.generate(prompt, max_new_tokens=160, history=history)
    record = {'case': name, 'prompt': prompt, 'history': history, 'response': response,
              'seconds': round(time.monotonic() - started, 2)}
    records.append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'model': agent.model_name, 'scope': 'conversational_smoke_checks_require_human_review_not_a_quality_benchmark', 'cases': records}, ensure_ascii=False, indent=2), encoding='utf-8')

history = []
for number, prompt in enumerate(('hey salam', 'parle en français', 'Que sais-tu de toi ?',
                                'Quel service ?', 'Who are you?',
                                'I want to speak with you now. Who are you? Are you OK?'), 1):
    started = time.monotonic()
    response = agent.generate(prompt, max_new_tokens=160, history=history)
    record = {'case': f'conversation_replay_{number}', 'prompt': prompt,
              'history': list(history), 'response': response,
              'seconds': round(time.monotonic() - started, 2)}
    records.append(record)
    history += [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    print(json.dumps(record, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps({'model': agent.model_name, 'scope': 'conversational_smoke_checks_require_human_review_not_a_quality_benchmark', 'cases': records}, ensure_ascii=False, indent=2), encoding='utf-8')
