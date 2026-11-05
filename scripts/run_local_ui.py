"""Start the local interface with isolated session files and verified system TLS."""
import argparse
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=root / '.tmp' / 'llm-local')
parser.add_argument('--cache-dir', type=Path)
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--backend', choices=('transformers', 'llama.cpp'), default='transformers')
parser.add_argument('--endpoint', default='http://127.0.0.1:8081')
parser.add_argument('--model')
args = parser.parse_args()
if not 1 <= args.port <= 65535:
    parser.error('port must be between 1 and 65535')
if sys.platform == 'win32':
    import truststore
    truststore.inject_into_ssl()
for name in ('CONFIGS', 'STATE', 'REPORTS', 'OUTPUTS'):
    os.environ[f'SIGMA_{name}_DIR'] = str(args.data_dir.resolve() / name.lower())
if args.cache_dir:
    os.environ['HF_HOME'] = str(args.cache_dir.resolve())
os.environ['PORT'] = str(args.port)
os.environ['SIGMA_LLM_BACKEND'] = args.backend
os.environ['SIGMA_LLAMA_ENDPOINT'] = args.endpoint
if args.model:
    os.environ['SIGMA_LLM_MODEL'] = args.model
os.environ['SIGMA_BIND_HOST'] = '127.0.0.1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
sys.path.insert(0, str(root))
runpy.run_path(str(root / 'app.py'), run_name='__main__')
