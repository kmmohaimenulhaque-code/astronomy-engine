import time
import torch

DEVICE = torch.device("cuda")
PI2 = 2.0 * torch.pi
ARC = 206264.80624709636


def lunar_core(T):
    T2 = T * T

    def Sine(phi):
        return torch.sin(PI2 * phi)

    S1 = Sine(0.19833 + 0.05611 * T)
    S2 = Sine(0.27869 + 0.04508 * T)
    S3 = Sine(0.16827 - 0.36903 * T)
    S4 = Sine(0.34734 - 5.37261 * T)
    S5 = Sine(0.10498 - 5.37899 * T)
    S6 = Sine(0.42681 - 0.41855 * T)
    S7 = Sine(0.14943 - 5.37511 * T)

    DL0 = (
        0.84*S1 + 0.31*S2 + 14.27*S3 +
        7.26*S4 + 0.28*S5 + 0.24*S6
    )

    DL = (
        2.94*S1 + 0.31*S2 + 14.27*S3 +
        9.34*S4 + 1.12*S5 + 0.83*S6
    )

    DLS = -6.40*S1 - 1.89*S6

    DF = (
        0.21*S1 + 0.31*S2 + 14.27*S3 -
        88.70*S4 - 15.30*S5 + 0.24*S6 - 1.86*S7
    )

    DD = DL0 - DLS

    DGAM = (
        -3332e-9 * Sine(0.59734 - 5.37261*T)
        -539e-9 * Sine(0.35498 - 5.37899*T)
        -64e-9 * Sine(0.39943 - 5.37511*T)
    )

    L0 = (
        PI2 * torch.frac(
            0.60643382 + 1336.85522467*T - 0.00000313*T2
        )
        + DL0 / ARC
    )

    L = (
        PI2 * torch.frac(
            0.37489701 + 1325.55240982*T + 0.00002565*T2
        )
        + DL / ARC
    )

    LS = (
        PI2 * torch.frac(
            0.99312619 + 99.99735956*T - 0.00000044*T2
        )
        + DLS / ARC
    )

    F = (
        PI2 * torch.frac(
            0.25909118 + 1342.22782980*T - 0.00000892*T2
        )
        + DF / ARC
    )

    D = (
        PI2 * torch.frac(
            0.82736186 + 1236.85308708*T - 0.00000397*T2
        )
        + DD / ARC
    )

    return L0, L, LS, F, D, DGAM


def benchmark(total_samples, chunk_size=10_000_000_000):
    print()
    print("=" * 70)
    print(f"  NEURONEXUS — {total_samples:,} SAMPLE GPU RUN")
    print("=" * 70)

    chunks = (total_samples + chunk_size - 1) // chunk_size
    processed = 0

    # Prevent compiler/runtime warm-up from contaminating the benchmark.
    warmup = torch.linspace(
        0.20, 0.21, min(chunk_size, 100_000),
        dtype=torch.float64,
        device=DEVICE,
    )
    for _ in range(3):
        lunar_core(warmup)

    torch.cuda.synchronize()

    start = time.perf_counter()

    first_result = None
    last_result = None

    for chunk in range(chunks):
        n = min(chunk_size, total_samples - processed)

        T = torch.linspace(
            0.20,
            0.21,
            n,
            dtype=torch.float64,
            device=DEVICE,
        )

        result = lunar_core(T)

        if first_result is None:
            first_result = [x[0].item() for x in result]

        last_result = [x[-1].item() for x in result]

        processed += n

        del T, result

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = total_samples / elapsed

    print(f"Device       : {torch.cuda.get_device_name(0)}")
    print(f"Samples      : {total_samples:,}")
    print(f"Chunk size   : {chunk_size:,}")
    print(f"Chunks       : {chunks:,}")
    print(f"Time         : {elapsed:.6f} s")
    print(f"Throughput   : {throughput:,.0f} samples/s")
    print()
    print("First sample:")
    print(f"L0   = {first_result[0]:.15f}")
    print(f"L    = {first_result[1]:.15f}")
    print(f"LS   = {first_result[2]:.15f}")
    print(f"F    = {first_result[3]:.15f}")
    print(f"D    = {first_result[4]:.15f}")
    print(f"DGAM = {first_result[5]:.15e}")
    print()
    print(f"Last L0     = {last_result[0]:.15f}")
    print("=" * 70)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/PyTorch GPU unavailable.")

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     NEURONEXUS — TRILLION SAMPLE MARCH                   ║")
    print("║     MI300X / ROCm / FLOAT64 / STREAMED                   ║")
    print("╚════════════════════════════════════════════════════════════╝")

    benchmark(10_000_000_000, chunk_size=10_000_000)


if __name__ == "__main__":
    main()
