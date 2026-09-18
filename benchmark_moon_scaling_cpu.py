import sys
import time

sys.path.insert(0, "source/python")

from astronomy import astronomy as ae


def benchmark(N: int):
    print()
    print("=" * 58)
    print(f"  Astronomy Engine CPU — {N:,} Moon calculations")
    print("=" * 58)

    base = ae.Time.Make(2020, 1, 1, 0, 0, 0)

    # Generate observation times.
    times = [
        base.AddDays(i * 0.01)
        for i in range(N)
    ]

    # Warm-up.
    for t in times[:10]:
        ae.GeoMoon(t)

    # Benchmark.
    start = time.perf_counter()

    results = [
        ae.GeoMoon(t)
        for t in times
    ]

    elapsed = time.perf_counter() - start

    first = results[0]
    last = results[-1]

    per_sample_us = elapsed / N * 1_000_000
    throughput = N / elapsed

    print(f"Samples       : {N:,}")
    print(f"Total time    : {elapsed:.6f} s")
    print(f"Per sample    : {per_sample_us:.2f} µs")
    print(f"Throughput    : {throughput:,.0f} calculations/s")

    print()
    print("First result:")
    print(f"X = {first.x:.15f}")
    print(f"Y = {first.y:.15f}")
    print(f"Z = {first.z:.15f}")

    print()
    print("Last result:")
    print(f"X = {last.x:.15f}")
    print(f"Y = {last.y:.15f}")
    print(f"Z = {last.z:.15f}")


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║   NEURONEXUS — ASTRONOMY ENGINE CPU BASELINE          ║")
    print("╚════════════════════════════════════════════════════════╝")

    # Automatically benchmark all three scales.
    for N in (10_000, 100_000, 1_000_000):
        benchmark(N)

    print()
    print("=" * 58)
    print("  CPU BASELINE COMPLETE")
    print("=" * 58)


if __name__ == "__main__":
    main()
