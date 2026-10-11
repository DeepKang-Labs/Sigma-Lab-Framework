"""Loopback-only llama.cpp chat adapter; no invented Sigma quality scores."""
import json
import math
import os
import re
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SYSTEM_PROMPT = (
    'You are Sigma, a local AI assistant running on the user\'s computer. '
    'You are not a human or a customer service representative. '
    'Answer the actual question directly. Reply in the language requested by the user; '
    'Default to a brief, natural answer of one to three sentences unless detail is requested. '
    'Do not repeat your identity or limitations unless asked or relevant. '
    'A greeting needs a greeting; a language change needs only a short acknowledgment. '
    'Keep an explicit language preference across later turns until the user changes it. '
    'when the language is unclear, use French. Do not invent meetings, services, '
    'phone calls, email capabilities, personal experiences or facts about your configuration. '
    'If asked about yourself, explain that you are a local language model, without claiming '
    'consciousness or verified ethical judgment. Admit uncertainty. '
    'This application is an experimental local chat and diagnostic research interface. '
    'You have no tools for browsing, sending messages, making calls or inspecting files. '
    'You see only the instructions and current conversation supplied to you. '
    'Your configured model is {model}.'
)


def normalize_history(history):
    messages = []
    for entry in history or []:
        if not isinstance(entry, dict) or entry.get('role') not in ('user', 'assistant') or not isinstance(entry.get('content'), str):
            raise ValueError('History must contain text user/assistant messages.')
        messages.append({'role': entry['role'], 'content': entry['content']})
    return messages


def conversation_system_prompt(model, history, prompt):
    """Preserve explicit French/English requests across input language changes."""
    preference = None
    names = {'français': 'French', 'francais': 'French', 'french': 'French',
             'anglais': 'English', 'english': 'English'}
    pattern = re.compile(
        r'\b(?:parle[rz]?|r[ée]pond(?:s|ez|re)?|[ée]cris|[ée]crivez|continue[rz]?|'
        r'speak|reply|answer|respond|write|continue)\s+'
        r'(?:(?:uniquement|seulement|only|just|briefly|brièvement)\s+)?'
        r'(?:(?:en|in)\s+)?(français|francais|french|anglais|english)\b', re.I)
    for message in [*normalize_history(history), {'role': 'user', 'content': prompt}]:
        if message['role'] == 'user':
            for match in pattern.finditer(message['content']):
                prefix = message['content'][:match.start()]
                if re.search(r"\b(?:not|never|don't|don’t|ne)\s*$", prefix, re.I):
                    continue
                preference = names[match.group(1).lower()]
    instruction = SYSTEM_PROMPT.format(model=model)
    if preference:
        instruction += (f' The active response language is {preference}. Reply in {preference}. '
                        'A question written in another language does not change this explicit preference.')
    return instruction


class LocalChatLLM:
    def __init__(self, model_name=None):
        self.endpoint = os.getenv('SIGMA_LLAMA_ENDPOINT', 'http://127.0.0.1:8081').rstrip('/')
        parsed = urlparse(self.endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('The llama.cpp endpoint must be an HTTP loopback origin.')
        available = self._request('/v1/models')['data']
        requested = model_name or os.getenv('SIGMA_LLM_MODEL')
        identifiers = [item['id'] for item in available]
        self.model_name = requested or identifiers[0]
        if self.model_name not in identifiers:
            raise ValueError(f'Requested model is not loaded in the local server: {self.model_name}')
        self.temp, self.top_p = 0.6, 0.9
        self.context_size = int(os.getenv('SIGMA_LLAMA_CONTEXT', '4096'))
        self._lock = threading.RLock()
        self.report_dir = Path(os.getenv('SIGMA_REPORTS_DIR', 'reports'))
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, path, payload=None):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8') if payload is not None else None
        headers = {'Content-Type': 'application/json'}
        api_key = os.getenv('SIGMA_LLAMA_API_KEY')
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        request = Request(self.endpoint + path, data=data, headers=headers)
        with urlopen(request, timeout=600) as response:
            return json.load(response)

    def _messages(self, prompt, history, tokens):
        messages = [{'role': 'system', 'content': conversation_system_prompt(self.model_name, history, prompt)}]
        messages += normalize_history(history)
        messages.append({'role': 'user', 'content': prompt})
        while True:
            rendered = self._request('/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
            count = len(self._request('/tokenize', {'content': rendered})['tokens'])
            if count + tokens <= self.context_size - 16:
                return messages
            if len(messages) <= 2:
                raise ValueError('This message exceeds the configured context. Shorten it or increase the server context.')
            messages.pop(1)
            while len(messages) > 2 and messages[1]['role'] == 'assistant':
                messages.pop(1)

    def generate(self, user_msg, max_new_tokens=200, *, history=None, temperature=None, top_p=None):
        if not isinstance(user_msg, str) or not user_msg.strip():
            raise ValueError('A non-empty text prompt is required.')
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens < self.context_size:
            raise ValueError('Invalid generation token budget.')
        temperature = self.temp if temperature is None else float(temperature)
        top_p = self.top_p if top_p is None else float(top_p)
        if not math.isfinite(temperature) or not 0 < temperature <= 2 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError('Invalid sampling settings.')
        with self._lock:
            messages = self._messages(user_msg, history, max_new_tokens)
            result = self._request('/v1/chat/completions', {
                'model': self.model_name, 'messages': messages, 'temperature': temperature,
                'top_p': top_p, 'max_tokens': max_new_tokens, 'stream': False,
                'seed': int(os.getenv('SIGMA_LLM_SEED', '42')),
                'chat_template_kwargs': {'enable_thinking': False},
            })
            choice = result['choices'][0]
            text = choice['message'].get('content')
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError('The local model returned no visible answer.')
            report = {'timestamp': int(time.time()), 'model': result.get('model', self.model_name),
                      'backend': 'llama.cpp', 'prompt': user_msg, 'output': text,
                      'usage': result.get('usage'), 'finish_reason': choice.get('finish_reason'),
                      'messages': messages, 'temperature': temperature, 'top_p': top_p,
                      'seed': int(os.getenv('SIGMA_LLM_SEED', '42')),
                      'evaluation_scope': 'generation_only_no_quality_or_Sigma_score',
                      'history_source': 'current_browser_conversation'}
            target = self.report_dir / 'sigma_llm_last_report.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(target)
            return text
