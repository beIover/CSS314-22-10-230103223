import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT_ID = "230103223"
N = 30_000_000


def get_cpu_model():
    """Get CPU model on macOS with fallback."""
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return platform.processor() or platform.machine()


def worker_kernel_a(chunk):
    """Kernel A (Memory-Bound): Vector copy B[i] = A[i]."""
    return np.copy(chunk)


def worker_kernel_b(chunk):
    """Kernel B (Compute-Bound): Math formula B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])."""
    return np.sin(chunk) * np.cos(chunk) + np.tan(chunk)


def run_kernel(kernel_fn, data, num_workers, executor):
    """Executes a kernel function partitioned across num_workers processes."""
    chunk_size = len(data) // num_workers
    chunks = [
        (
            data[i * chunk_size : (i + 1) * chunk_size]
            if i < num_workers - 1
            else data[i * chunk_size :]
        )
        for i in range(num_workers)
    ]

    start = time.perf_counter()
    results = list(executor.map(kernel_fn, chunks))
    _out = np.concatenate(results)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    max_cores = os.cpu_count() or 4
    worker_counts = sorted(list(set([1, 2, max_cores])))

    rng = np.random.default_rng(int(STUDENT_ID))
    data_A = rng.uniform(0.1, 1.0, size=N).astype(np.float64)

    print("=== TASK 7: Arithmetic Intensity vs. Memory Bottlenecks ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {max_cores}")
    print(f"Array Size: {N:,} elements (float64)\n")

    # --- Benchmark Kernel A ---
    print("--- Kernel A: Memory-Bound (Vector Copy B[i] = A[i]) ---")
    print(
        f"{'Cores':<8} | {'Wall Time':<12} | {'Speedup':<10} | {'Efficiency':<12}"
    )
    print("-" * 50)

    times_a = {}
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for p in worker_counts:
            t_p = run_kernel(worker_kernel_a, data_A, p, executor)
            times_a[p] = t_p

        t1_a = times_a[1]
        for p in worker_counts:
            t_p = times_a[p]
            speedup = t1_a / t_p
            eff = speedup / p
            print(
                f"{p:<8} | {t_p:>8.2f} ms | {speedup:>8.2f}x | {eff:>10.2%}"
            )

    # --- Benchmark Kernel B ---
    print(
        "\n--- Kernel B: Compute-Bound (Trigonometric B[i] = sin*cos + tan) ---"
    )
    print(
        f"{'Cores':<8} | {'Wall Time':<12} | {'Speedup':<10} | {'Efficiency':<12}"
    )
    print("-" * 50)

    times_b = {}
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for p in worker_counts:
            t_p = run_kernel(worker_kernel_b, data_A, p, executor)
            times_b[p] = t_p

        t1_b = times_b[1]
        for p in worker_counts:
            t_p = times_b[p]
            speedup = t1_b / t_p
            eff = speedup / p
            print(
                f"{p:<8} | {t_p:>8.2f} ms | {speedup:>8.2f}x | {eff:>10.2%}"
            )

    # --- Bottleneck Analysis Output ---
    print("\n=== BOTTLENECK ANALYSIS ===")
    eff_a_max = (times_a[1] / times_a[max_cores]) / max_cores
    eff_b_max = (times_b[1] / times_b[max_cores]) / max_cores

    if eff_a_max < eff_b_max:
        print(
            "Kernel A (Memory-Bound Vector Copy) stalled in scaling due to memory channel saturation."
        )
        print(
            f"Reason: At {max_cores} cores, Kernel A achieved only {eff_a_max:.1%} parallel efficiency, "
            f"whereas Kernel B maintained {eff_b_max:.1%} efficiency."
        )
    else:
        print(
            "Kernel B (Compute-Bound Math) stalled in scaling due to CPU computational saturation."
        )


if __name__ == "__main__":
    main()