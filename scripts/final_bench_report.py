import json, os, math

p = "Temp/unsw_combined_parallel_bench_results.json"
with open(p, "r", encoding="utf-8") as f:
    j = json.load(f)
ms = j.get("measurements", [])
S = len(ms)
succ = sum(1 for m in ms if m.get("status") == "success")
fail = S - succ
keys = [f"{m['model']}|{m.get('workers')}|{m.get('model_threads')}" for m in ms]
dups = len(keys) != len(set(keys))
print("JSON path:", p)
print("Total measurements:", S)
print("Successful:", succ)
print("Failed/partial:", fail)
print("Duplicate keys:", dups)

# XGBoost stats
base = 663.47
x = [m for m in ms if m["model"] == "XGBoost" and m.get("elapsed_seconds")]
if x:
    for m in x:
        sp = base / float(m["elapsed_seconds"])
        print("XGBoost", m["workers"], m["model_threads"], "elapsed", round(m["elapsed_seconds"], 3), "speedup", round(sp, 3), "cpu%", m.get("cpu_percent"))
    best = min(x, key=lambda m: float(m["elapsed_seconds"]))
    best_sp = base / float(best["elapsed_seconds"])
    print("Best XGBoost:", best["workers"], best["model_threads"], "elapsed", round(best["elapsed_seconds"], 3), "speedup", round(best_sp, 3))

    # compute optimized runtime from baseline 27 days
    baseline_days = 27.0
    optimized_days = baseline_days / best_sp
    optimized_hours = optimized_days * 24
    print("\nEstimated runtime with best measured CPU config:")
    print("days:", round(optimized_days, 3), "hours:", round(optimized_hours, 2), "speedup:", round(best_sp, 3))

    # hypothetical methodological reductions
    def est_days(factor):
        return optimized_days * factor

    # B: 8 wolves x15 iters
    factor_B = (8 * 15) / (10 * 20)
    days_B = est_days(factor_B)
    # C: 8x10
    factor_C = (8 * 10) / (10 * 20)
    days_C = est_days(factor_C)
    # D: inner 5->3
    factor_D = 3 / 5
    days_D = est_days(factor_D)
    # E: both B and D
    days_E = est_days(factor_B * factor_D)
    print("\nHypothetical runtimes (based on best-measured CPU):")
    print("8w x15 iters:", round(days_B, 3), "days")
    print("8w x10 iters:", round(days_C, 3), "days")
    print("inner 5->3:", round(days_D, 3), "days")
    print("8x15 + inner->3:", round(days_E, 3), "days")
else:
    print("No XGBoost elapsed measurements available")

# GPU note
print("\nGPU note: GPU benchmarking unavailable with current XGBoost build.")

# final validation: try parse
try:
    json.dumps(j)
    print("\nFinal JSON parse: OK")
except Exception as e:
    print("\nFinal JSON parse: FAILED", e)

print("\nDone")
