import json
j=json.load(open('Temp/unsw_combined_parallel_bench_results.json'))
base=663.47
x=[m for m in j['measurements'] if m['model']=='XGBoost']
for m in x:
    sp=base/float(m['elapsed_seconds']) if m.get('elapsed_seconds') else None
    print(m['workers'], m['model_threads'], m['elapsed_seconds'], 'speedup=', round(sp,3), 'cpu%', m.get('cpu_percent'))
best=min(x, key=lambda m: float(m['elapsed_seconds']))
print('BEST', best['workers'], best['model_threads'], best['elapsed_seconds'])
