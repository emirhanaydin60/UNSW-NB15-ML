"""Probe running benchmark process and Temp JSON updates.

Outputs a JSON object with keys:
- running (bool)
- processes (list of {pid, cmdline, cpu_percent, mem_mb, create_time})
- json_path, json_mtime, json_size, last_measurement
- recent_bench_log (path, mtime, tail_lines)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import time

try:
    import psutil
except Exception:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "Temp" / "unsw_combined_parallel_bench_results.json"

out = {"running": False, "processes": [], "json_path": str(JSON_PATH), "json_mtime": None, "json_size": None, "last_measurement": None, "recent_bench_log": None}

# find python processes running the target script
if psutil is None:
    print(json.dumps({"error": "psutil not available"}))
    sys.exit(0)

candidates = []
for proc in psutil.process_iter(['pid','name','cmdline','create_time']):
    try:
        cmd = proc.info.get('cmdline') or []
        if any('unsw_valid_large_sample_bench.py' in str(c) for c in cmd):
            candidates.append(proc)
    except Exception:
        continue

if candidates:
    out['running'] = True
    for p in candidates:
        try:
            cpu = p.cpu_percent(interval=0.5)
            mem = p.memory_info().rss / (1024*1024)
            out['processes'].append({
                'pid': int(p.pid),
                'name': p.name(),
                'cmdline': p.cmdline(),
                'cpu_percent': cpu,
                'mem_mb': round(mem,2),
                'create_time': p.create_time(),
            })
        except Exception as e:
            out['processes'].append({'pid': int(p.pid), 'error': str(e)})

# JSON file info
if JSON_PATH.exists():
    st = JSON_PATH.stat()
    out['json_mtime'] = st.st_mtime
    out['json_size'] = st.st_size
    try:
        with JSON_PATH.open('r', encoding='utf-8') as f:
            j = json.load(f)
            measurements = j.get('measurements', [])
            if measurements:
                last = measurements[-1]
                out['last_measurement'] = last
    except Exception as e:
        out['json_load_error'] = str(e)

# search recent bench.log files under runs/
runs_dir = ROOT / 'runs'
best_log = None
best_mtime = 0
if runs_dir.exists():
    for p in runs_dir.rglob('bench.log'):
        try:
            m = p.stat().st_mtime
            if m > best_mtime:
                best_mtime = m
                best_log = p
        except Exception:
            continue

if best_log is not None:
    tail = []
    try:
        with best_log.open('rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # read last up to 64KB
            to_read = min(size, 65536)
            f.seek(size - to_read)
            data = f.read().decode('utf-8', errors='replace')
            tail = data.splitlines()[-200:]
        out['recent_bench_log'] = {'path': str(best_log), 'mtime': best_mtime, 'tail_lines': tail}
    except Exception as e:
        out['recent_bench_log'] = {'path': str(best_log), 'error': str(e)}

print(json.dumps(out, indent=2, default=str))
