"""Run a local GGUF model and Sigma UI; shut down the owned server on exit."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine', type=Path, required=True)
parser.add_argument('--model-file', type=Path, required=True)
parser.add_argument('--alias')
parser.add_argument('--data-dir', type=Path, default=root / '.tmp' / 'gguf-local')
parser.add_argument('--context', type=int, default=4096)
parser.add_argument('--gpu-layers', default='auto')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--server-port', type=int, default=8081)
args = parser.parse_args()
args.alias = args.alias or args.model_file.stem
if not args.engine.is_file() or not args.model_file.is_file():
    parser.error('The engine executable and downloaded model must exist.')
with args.model_file.open('rb') as stream:
    if stream.read(4) != b'GGUF':
        parser.error('The model does not have a GGUF header.')
if args.context < 1024 or not all(1 <= value <= 65535 for value in (args.port, args.server_port)) or args.port == args.server_port:
    parser.error('Invalid context or ports.')
args.data_dir.mkdir(parents=True, exist_ok=True)
log_path = args.data_dir / 'llama-server.log'
endpoint = f'http://127.0.0.1:{args.server_port}'
command = [str(args.engine.resolve()), '-m', str(args.model_file.resolve()),
           '--alias', args.alias, '--host', '127.0.0.1', '--port', str(args.server_port),
           '-c', str(args.context), '-np', '1', '-ngl', args.gpu_layers,
           '--fit', 'on', '--fit-target', '1024', '-t', '4', '-tb', '4',
           '-b', '256', '-ub', '128', '--chat-template-kwargs', '{"enable_thinking":false}']
with log_path.open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 600
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Local model server exited. See {log_path}')
            try:
                with urlopen(endpoint + '/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f'Model server did not become ready. See {log_path}')
            time.sleep(0.5)
        print(f'Local model ready: {args.alias}; Sigma UI on port {args.port}', flush=True)
        os.environ['SIGMA_LLAMA_CONTEXT'] = str(args.context)
        sys.argv = ['run_local_ui.py', '--backend', 'llama.cpp', '--model', args.alias,
                    '--endpoint', endpoint, '--data-dir', str(args.data_dir), '--port', str(args.port)]
        runpy.run_path(str(root / 'scripts' / 'run_local_ui.py'), run_name='__main__')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
