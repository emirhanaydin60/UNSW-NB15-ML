"""Monitor running benchmark and dump final probe output when it stops.

This script polls `scripts/check_benchmark_status.py` every 60 seconds and
writes the final JSON to `Temp/final_bench_status.json` once `running` is False.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "Temp" / "final_bench_status.json"
CHECK_SCRIPT = ROOT / "scripts" / "check_benchmark_status.py"

def probe():
    p = subprocess.run(["python", str(CHECK_SCRIPT)], capture_output=True, text=True)
    if p.returncode != 0:
        # return a best-effort object
        return {'error': 'check script failed', 'stderr': p.stderr}
    try:
        return json.loads(p.stdout)
    except Exception as e:
        return {'error': 'json_load_failed', 'exc': str(e), 'stdout': p.stdout}


def main(poll_seconds: int = 60):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        out = probe()
        now = time.time()
        # write a rolling temp so we can inspect progress if needed
        try:
            (OUT_PATH.parent / 'last_probe.json').write_text(json.dumps({'ts': now, 'probe': out}, indent=2))
        except Exception:
            pass

        running = bool(out.get('running')) if isinstance(out, dict) else False
        if not running:
            try:
                OUT_PATH.write_text(json.dumps({'ts': now, 'final_probe': out}, indent=2))
            except Exception as e:
                print('failed to write final output:', e)
            print('Benchmark no longer running; final status written to', str(OUT_PATH))
            return 0

        # still running
        print('Benchmark still running; sleeping', poll_seconds, 'seconds')
        time.sleep(poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
