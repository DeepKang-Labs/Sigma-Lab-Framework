"""Real model read/propose/test demo. No apply capability is called or exposed."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sigma.dev_gateway import Gateway, digest
from sigma.local_chat import LocalChatLLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ['SIGMA_LLAMA_API_KEY'] = (root / '.tmp/gguf-local/local-api-key.txt').read_text(encoding='utf-8')
os.environ['SIGMA_REPORTS_DIR'] = str(root / '.tmp/gguf-local/dev-demo-reports')
gateway = Gateway(root / '.tmp/dev-sandbox', root / '.tmp/dev-gateway-state')
source = gateway.dispatch({'operation': 'read'})['content']
prompt = ('You are preparing a proposal only. You have no apply authority. '
          'The only permitted file is calculator.py. Correct add(a,b) so it returns the sum. '
          'Reply only with a JSON object with exactly operation="propose" and content containing the complete replacement Python file. '
          'Keep exactly one function with one return expression. Current file:\n' + source)
started = time.monotonic()
agent = LocalChatLLM()
response = agent.generate(prompt, history=[], max_new_tokens=300)
generation = json.loads((agent.report_dir / 'sigma_llm_last_report.json').read_text(encoding='utf-8'))
record = {'model': agent.model_name, 'prompt': prompt, 'response': response,
          'seconds': round(time.monotonic() - started, 2), 'usage': generation['usage'],
          'finish_reason': generation['finish_reason'], 'before_sha256': digest(source),
          'target_mutation_authorized': False}
try:
    proposal = gateway.dispatch(json.loads(response))
    record['proposal'] = proposal
    record['test'] = gateway.dispatch({'operation': 'test', 'proposal_id': proposal['id']})
except (ValueError, PermissionError, KeyError) as error:
    gateway.audit('model_proposal', 'deny', reason=type(error).__name__)
    record['error'] = type(error).__name__ + ': ' + str(error)
record['after_sha256'] = digest(gateway.read())
record['target_unchanged'] = record['before_sha256'] == record['after_sha256']
args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'target_unchanged': record['target_unchanged'], 'test': record.get('test'), 'error': record.get('error')}, indent=2))
