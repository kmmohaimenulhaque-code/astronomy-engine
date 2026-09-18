import time
import torch


# ============================================================
# NEURONEXUS — GPU LUNAR KERNEL v0.1
#
# Vectorized form of the opening mathematical kernel of
# Astronomy Engine's _CalcMoon().
#
# IMPORTANT:
# astronomy.py remains the trusted reference implementation.
# ============================================================

DEVICE = torch.device("cuda")

PI2 = 2.0 * torch.pi


def lunar_core(T):
    """
    Vectorized lunar fundamental-angle kernel.

    Parameters
    ----------
    T : torch.Tensor
        Julian centuries of terrestrial time.

    Returns
    -------
    dict
        Batched intermediate lunar quantities.
    """

    T2 = T * T

    def Sine(phi):
        return torch.sin(PI2 * phi)

    # Exact equations from Astronomy Engine _CalcMoon().
    S1 = Sine(0.19833 + 0.05611 * T)
    S2 = Sine(0.27869 + 0.04508 * T)
    S3 = Sine(0.16827 - 0.36903 * T)
    S4 = Sine(0.34734 - 5.37261 * T)
    S5 = Sine(0.10498 - 5.37899 * T)
    S6 = Sine(0.42681 - 0.41855 * T)
    S7 = Sine(0.14943 - 5.37511 * T)

    DL0 = (
        0.84 * S1
        + 0.31 * S2
        + 14.27 * S3
        + 7.26 * S4
        + 0.28 * S5
        + 0.24 * S6
    )

    DL = (
        2.94 * S1
        + 0.31 * S2
        + 14.27 * S3
        + 9.34 * S4
        + 1.12 * S5
        + 0.83 * S6
    )

    DLS = (
        -6.40 * S1
        -1.89 * S6
    )

    DF = (
        0.21 * S1
        + 0.31 * S2
        + 14.27 * S3
        -88.70 * S4
        -15.30 * S5
        + 0.24 * S6
        -1.86 * S7
    )

    DD = DL0 - DLS

    DGAM = (
        -3332e-9 * Sine(0.59734 - 5.37261 * T)
        -539e-9 * Sine(0.35498 - 5.37899 * T)
        -64e-9 * Sine(0.39943 - 5.37511 * T)
    )

    L0 = (
        PI2 * torch.frac(
            0.60643382
            + 1336.85522467 * T
            - 0.00000313 * T2
        )
        + DL0 / 206264.80624709636
    )

    L = (
        PI2 * torch.frac(
            0.37489701
            + 1325.55240982 * T
            + 0.00002565 * T2
        )
        + DL / 206264.80624709636
    )

    LS = (
        PI2 * torch.frac(
            0.99312619
            + 99.99735956 * T
            - 0.00000044 * T2
        )
        + DLS / 206264.80624709636
    )

    F = (
        PI2 * torch.frac(
            0.25909118
            + 1342.22782980 * T
            - 0.00000892 * T2
        )
        + DF / 206264.80624709636
    )

    D = (
        PI2 * torch.frac(
            0.82736186
            + 1236.85308708 * T
            - 0.00000397 * T2
        )
        + DD / 206264.80624709636
    )

    return {
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "S4": S4,
        "S5": S5,
        "S6": S6,
        "S7": S7,
        "DL0": DL0,
        "DL": DL,
        "DLS": DLS,
        "DF": DF,
        "DD": DD,
        "DGAM": DGAM,
        "L0": L0,
        "L": L,
        "LS": LS,
        "F": F,
        "D": D,
    }


def benchmark(N):
    print()
    print("=" * 64)
    print(f"  NEURONEXUS GPU LUNAR CORE — {N:,} SAMPLES")
    print("=" * 64)

    # Same kind of time span used by the CPU benchmarks.
    # T = Julian centuries from J2000-style TT scale.
    #
    # We deliberately use float64 here.
    # Scientific agreement comes before raw speed.
    T = torch.linspace(
        0.20,
        0.21,
        N,
        dtype=torch.float64,
        device=DEVICE,
    )

    # Warm-up
    for _ in range(3):
        lunar_core(T)

    torch.cuda.synchronize()

    start = time.perf_counter()

    result = lunar_core(T)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    throughput = N / elapsed

    print(f"Device       : {torch.cuda.get_device_name(0)}")
    print(f"Samples      : {N:,}")
    print(f"Time         : {elapsed:.6f} s")
    print(f"Throughput   : {throughput:,.0f} samples/s")
    print(f"L0 first     : {result['L0'][0].item():.15f}")
    print(f"L0 last      : {result['L0'][-1].item():.15f}")
    print(f"F first      : {result['F'][0].item():.15f}")
    print(f"DGAM first   : {result['DGAM'][0].item():.15e}")
    print("=" * 64)

    return elapsed


def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       NEURONEXUS — LUNAR GPU ENGINE v0.1                 ║")
    print("║       MI300X / ROCm vectorized mathematics               ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/PyTorch GPU is not available.")

    for N in (
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ):
        benchmark(N)


if __name__ == "__main__":
    main()
