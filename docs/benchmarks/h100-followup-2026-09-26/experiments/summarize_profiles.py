"""Summarize paired raw Nsight SQLite exports without exposing environment data.

Usage: python summarize_profiles.py bench_h100_artifacts/2026-09-26
"""
import json
import pathlib
import sqlite3
import statistics
import sys

root = pathlib.Path(sys.argv[1])
result = {}
for label in ("control", "delivered"):
    con = sqlite3.connect(root / f"nsys-mature-{label}.sqlite")
    kernels = con.execute("""
        SELECT s.value, count(*), sum(k.end-k.start)/1e6,
               sum(k.end-k.start)/count(*)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY k.shortName ORDER BY sum(k.end-k.start) DESC
    """).fetchall()
    ranks = {}
    for device, duration in con.execute("""
        SELECT deviceId, (end-start)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        WHERE s.value LIKE 'nccl%' AND gridX>1 ORDER BY start
    """):
        ranks.setdefault(device, []).append(duration)
    collectives = 3 if label == "control" else 1
    nccl = {}
    for device, values in ranks.items():
        steps = [sum(values[i:i+collectives]) for i in range(0, len(values), collectives)]
        nccl[device] = {
            "gradient_calls": len(values),
            "median_ms_per_step_excluding_first": statistics.median(steps[1:]),
            "max_ms_per_step": max(steps),
        }
    result[label] = {
        "kernels": [dict(zip(("name", "calls", "total_ms", "mean_ms"), row)) for row in kernels],
        "nccl": nccl,
    }
print(json.dumps(result, indent=2))
