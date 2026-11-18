"""Read one JSON query from stdin; emit one JSON result. No API or answer writes."""
import argparse
import json
import sys
from .table_engine import execute

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace', required=True)
p.add_argument('--backend', default='python')
a = p.parse_args()
try:
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('Query exceeds 64 KiB')
    request = json.loads(raw)
except (ValueError, UnicodeError):
    result = {'ok': False, 'result': None, 'error': 'Invalid JSON or query exceeds 64 KiB'}
else:
    result = execute(request, a.workspace, backend=a.backend)
print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
raise SystemExit(0 if result['ok'] else 2)
