import json
from pathlib import Path
from collections import defaultdict
import datetime
p = Path("runs/microbench_xgb_full_1790860876/metadata/diagnostics/balancing_events.jsonl")
lines = p.read_text().splitlines()
min_ts = defaultdict(lambda: None)
max_ts = defaultdict(lambda: None)
exceptions = []
seen_wolves = set()
for l in lines:
    obj = json.loads(l)
    w = obj.get("wolf_index")
    seen_wolves.add(w)
    t = obj.get("timestamp")
    if t:
        dt = datetime.datetime.fromisoformat(t.replace('Z', '+00:00'))
        if min_ts[w] is None or dt < min_ts[w]:
            min_ts[w] = dt
        if max_ts[w] is None or dt > max_ts[w]:
            max_ts[w] = dt
    if obj.get("exception"):
        exceptions.append(obj)
# Print per-wolf balancing spans
print("per_wolf_balancing_spans:")
for w in sorted(seen_wolves):
    mn = min_ts[w]
    mx = max_ts[w]
    dur = (mx - mn).total_seconds() if mn and mx else None
    print(f"wolf={w} start={mn.isoformat() if mn else None} end={mx.isoformat() if mx else None} duration_s={dur}")
print()
print(f"num_wolves_seen={len(seen_wolves)}")
print(f"exceptions_count={len(exceptions)}")
if exceptions:
    print("exceptions_sample:")
    for e in exceptions[:5]:
        print(json.dumps(e))
# Check completeness: expect at least one balancing event per wolf
print()
missing = [w for w in range(10) if w not in seen_wolves]
print(f"missing_wolves_expected_0_9 = {missing}")
# report total entries
print(f"total_events={len(lines)}")
