import json
from pathlib import Path

runs=['pilot_xgb_1790937472','pilot_dt_1790938778','pilot_rf_1790939198','pilot_lr_1790941266']
for r in runs:
    p=Path('runs')/r/'metadata'/'runtime_events.jsonl'
    if not p.exists():
        print(r,'MISSING')
        continue
    evs=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    pairs=[(e['outer_fold'], e['wolf_index']) for e in evs]
    print(r,'count',len(evs),'unique_pairs',len(set(pairs)))
    folds=sorted({e['outer_fold'] for e in evs})
    print(' folds',folds)
    keys=set().union(*[set(e.keys()) for e in evs])
    print(' keys',sorted(keys))
    print()
