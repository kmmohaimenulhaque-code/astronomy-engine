import sys
import time

sys.path.insert(0, "source/python")

from astronomy import astronomy as ae

# One known test time.
t = ae.Time.Make(2026, 1, 1, 0, 0, 0)

# Warm-up
for _ in range(10):
    ae.GeoMoon(t)

# Single calculation
start = time.perf_counter()
v = ae.GeoMoon(t)
elapsed = time.perf_counter() - start

print("========== Astronomy Engine CPU ==========")
print("Time       : 2026-01-01 00:00:00")
print(f"X (AU)     : {v.x:.15f}")
print(f"Y (AU)     : {v.y:.15f}")
print(f"Z (AU)     : {v.z:.15f}")
print(f"Elapsed    : {elapsed * 1e6:.2f} µs")
print("==========================================")
