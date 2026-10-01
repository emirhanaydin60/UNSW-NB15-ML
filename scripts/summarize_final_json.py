import json, os

p = "Temp/unsw_combined_parallel_bench_results.json"
with open(p, "r", encoding="utf-8") as f:
    j = json.load(f)
ms = j.get("measurements", [])
# counts
total = len(ms)
by_status = {}
for m in ms:
    s = m.get("status") or "unknown"
    by_status.setdefault(s, []).append(m)
print("JSON:", p)
print("file_mtime:", open(p, "rb"))
print("Total measurements:", total)
print("Statuses:")
for k in sorted(by_status.keys()):
    print(" -", k + ":", len(by_status[k]))

# List successes
print("\nSuccessful measurements:")
for m in by_status.get("success", []):
    info = ["model=" + str(m.get("model"))]
    if "workers" in m:
        info.append("workers=" + str(m.get("workers")))
    if "model_threads" in m:
        info.append("threads=" + str(m.get("model_threads")))
    if "elapsed_seconds" in m:
        info.append("elapsed=" + str(round(m["elapsed_seconds"], 3)))
    if "timestamp" in m:
        info.append("ts=" + m["timestamp"])
    print(" -", ", ".join(info))

# Partial/fail details with exact errors
print("\nPartial/Failed measurements details (verbatim wolf errors):")
for m in by_status.get("partial_failure", []) + by_status.get("failed", []):
    hdr = f"Model={m.get('model')} workers={m.get('workers')} threads={m.get('model_threads')} status={m.get('status')}"
    print("\n" + hdr)
    errs = m.get("errors") or []
    for e in errs:
        wolf = e.get("wolf")
        err_str = e.get("error")
        print(f" - wolf={wolf}: {err_str}")

# XGBoost runtime estimate using only status=='success' and number_of_successful_wolves==10
x_success = [m for m in ms if m.get("model") == "XGBoost" and m.get("status") == "success" and m.get("number_of_successful_wolves") == 10 and m.get("elapsed_seconds")]
print("\nXGBoost valid success measurements (number_of_successful_wolves==10):", len(x_success))
for m in x_success:
    print(" - workers", m.get("workers"), "threads", m.get("model_threads"), "elapsed", round(m.get("elapsed_seconds"), 3))
if x_success:
    base = 663.47
    baseline_days = 27.0
    best = min(x_success, key=lambda m: float(m["elapsed_seconds"]))
    best_elapsed = float(best["elapsed_seconds"])
    best_sp = base / best_elapsed
    optimized_days = baseline_days / best_sp
    optimized_hours = optimized_days * 24
    print("\nBest XGBoost (valid): workers", best.get("workers"), "threads", best.get("model_threads"), "elapsed", round(best_elapsed, 3))
    print("speedup", round(best_sp, 3))
    print("Estimated runtime with best measured CPU config: days:", round(optimized_days, 3), "hours:", round(optimized_hours, 2), "speedup:", round(best_sp, 3))
    # hypotheticals
    factor_B = (8 * 15) / (10 * 20)
    factor_C = (8 * 10) / (10 * 20)
    factor_D = 3 / 5
    days_B = optimized_days * factor_B
    days_C = optimized_days * factor_C
    days_D = optimized_days * factor_D
    days_E = optimized_days * factor_B * factor_D
    print("\nHypothetical runtimes (based on best-measured CPU):")
    print("8w x15 iters:", round(days_B, 3), "days")
    print("8w x10 iters:", round(days_C, 3), "days")
    print("inner 5->3:", round(days_D, 3), "days")
    print("8x15 + inner->3:", round(days_E, 3), "days")
else:
    print("\nNo valid XGBoost success measurements available")
