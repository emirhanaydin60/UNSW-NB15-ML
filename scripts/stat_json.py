import os, datetime

p = r"C:\Users\emirh\Desktop\Projects\UNSW-NB15-ML\Temp\unsw_combined_parallel_bench_results.json"
st = os.stat(p)
mtime = datetime.datetime.fromtimestamp(st.st_mtime).astimezone(datetime.timezone.utc).isoformat()
print(mtime)
print(st.st_size)
