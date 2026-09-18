import sys
import time

sys.path.insert(0, "source/python")

from astronomy import astronomy as ae

N = 10_000

base = ae.Time.Make(2020, 1, 1, 0, 0, 0)

times = [
    base.AddDays(i * 0.01)
    for i in range(N)
]

# Warm-up
for t in times[:10]:
    ae.GeoMoon(t)

start = time.perf_counter()

results = [ae.GeoMoon(t) for t in times]

elapsed = time.perf_counter() - start

print("========== Astronomy Engine CPU Batch ==========")
print(f"Samples       : {N:,}")
print(f"Total time    : {elapsed:.6f} s")
print(f"Per sample    : {elapsed / N * 1e6:.2f} µs")
print(f"Throughput    : {N / elapsed:,.0f} calculations/s")
print()
print("First result:")
print(f"X = {results[0].x:.15f}")
print(f"Y = {results[0].y:.15f}")
print(f"Z = {results[0].z:.15f}")
print()
print("Last result:")
print(f"X = {results[-1].x:.15f}")
print(f"Y = {results[-1].y:.15f}")
print(f"Z = {results[-1].z:.15f}")
print("================================================")
