"""Restricted broker CLI. Human approval cannot be passed as JSON or flags."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sigma.dev_gateway import Gateway

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['request', 'approve'])
parser.add_argument('proposal_id', nargs='?')
args = parser.parse_args()
gateway = Gateway(root / '.tmp' / 'dev-sandbox', root / '.tmp' / 'dev-gateway-state')
if args.action == 'approve':
    result = gateway.human_apply(args.proposal_id or '', sys.stdin, sys.stdout)
else:
    request = json.loads(sys.stdin.read(16384))
    result = gateway.dispatch(request)
print(json.dumps(result, ensure_ascii=False, indent=2))
