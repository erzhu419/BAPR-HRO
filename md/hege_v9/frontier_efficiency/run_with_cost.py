"""Single-shot process wrapper: count startup/imports and output serialization."""
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

out = Path(__file__).resolve().parent
index = int(sys.argv[1]) if len(sys.argv) > 1 else 55
log = out / f'{index:03d}_run.log'
cost = out / f'{index:03d}_process_cost.json'
if log.exists() or cost.exists() or (out / f'{index:03d}_efficient.json').exists():
    raise FileExistsError('Refusing to overwrite a retained natural-state run')
start = perf_counter()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
command = [sys.executable, str(out / 'efficient_priority.py'), str(index)]
with log.open('x') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
after = resource.getrusage(resource.RUSAGE_CHILDREN)
self_usage = resource.getrusage(resource.RUSAGE_SELF)
record = dict(command=command, returncode=result.returncode,
              child_user_cpu=after.ru_utime - before.ru_utime,
              child_system_cpu=after.ru_stime - before.ru_stime,
              child_total_cpu=after.ru_utime - before.ru_utime + after.ru_stime - before.ru_stime,
              wall=perf_counter() - start, max_resident_kib=after.ru_maxrss,
              wrapper_cpu=self_usage.ru_utime + self_usage.ru_stime,
              definition='Cold Python child including all module imports, archive loading, graph construction, query, interval extraction, serialization and logging')
cost.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(result.returncode)
