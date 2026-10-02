import json
from pathlib import Path
import sys

runs = sys.argv[1:]
if not runs:
    print(json.dumps({"error": "no runs provided"}))
    sys.exit(1)

out = {}
for r in runs:
    run_dir = Path("runs") / r
    meta = run_dir / "metadata"
    evf = meta / "runtime_events.jsonl"
    stats = {}
    events = []
    if evf.exists():
        with evf.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    pass
    durations = [e.get("duration_seconds", 0.0) for e in events]
    stats["runtime_events_count"] = len(events)
    stats["avg_candidate_duration"] = sum(durations) / len(durations) if durations else None
    stats["min_candidate_duration"] = min(durations) if durations else None
    stats["max_candidate_duration"] = max(durations) if durations else None
    stats["duration_list"] = durations
    # balancing diagnostics
    balf = meta / "diagnostics" / "balancing_events.jsonl"
    bal_events = []
    if balf.exists():
        with balf.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    bal_events.append(json.loads(line))
                except:
                    pass
    stats["balancing_event_count"] = len(bal_events)
    stats["balancing_exceptions"] = sum(1 for e in bal_events if e.get("exception"))
    stats["smotenc_failures"] = sum(1 for e in bal_events if e.get("smotenc_failure"))
    stats["fallback_count"] = sum(1 for e in bal_events if e.get("fallback"))
    # stage totals
    # try to read pilot_summary.json for totals
    summaryf = run_dir / "pilot_summary.json"
    if summaryf.exists():
        try:
            with summaryf.open("r", encoding="utf-8") as f:
                js = json.load(f)
                stats["total_elapsed_seconds"] = js.get("total_elapsed_seconds")
                stats["outer_folds"] = js.get("outer_folds")
                stats["total_candidates"] = js.get("total_candidates")
                stats["total_success"] = js.get("total_success")
                stats["total_failed"] = js.get("total_failed")
                stats["stage_totals"] = js.get("stage_totals")
        except:
            pass
    out[r] = stats
print(json.dumps(out, indent=2))
