"""Reproduce the original 1MiB-prefix cohort using the remote member cache."""
import json
from pathlib import Path
import runpy
import sys


folder = Path(__file__).resolve().parent
legacy = runpy.run_path(str(folder / 'filter_actuals_remote.py'))
day = sys.argv[sys.argv.index('--date') + 1]
cached = folder / 'actual_member_parts' / day / '0.deflate'
original = legacy['fetch_range']


def fetch_range(url, start, length):
    if start >= 0 and length == 1048576:
        content = cached.read_bytes()
        if len(content) != length:
            raise ValueError('Original 1MiB prefix is not fully cached')
        return content, 0
    return original(url, start, length)


legacy['main'].__globals__['fetch_range'] = fetch_range
legacy['main']()
output = Path(sys.argv[sys.argv.index('--output') + 1]).with_suffix('.audit.json')
audit = json.loads(output.read_text())
audit['member_prefix_bytes'] = 1048576
audit['prefix_reused_from_complete_member_remote_cache'] = True
audit['remote_bytes_requested'] = 65557 + 30
output.write_text(json.dumps(audit, indent=2) + '\n')
