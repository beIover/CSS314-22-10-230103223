import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

STUDENT_ID = "230103223"
N_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]


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


def worker_task(args):
    """Worker function for vectorized SAXPY on a array chunk: y = a * x + b."""
    x, y, a, b, start_idx, end_idx = args
    y[start_idx:end_idx] = a * x[start_idx:end_idx] + b


def run_serial(x, a, b):
    """Single-threaded serial vector operation."""
    y = np.empty_like(x)
    start = time.perf_counter()
    y[:] = a * x + b
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


def run_parallel(x, a, b, num_workers, executor):
    """Multi-threaded parallel vector operation across workers."""
    n = len(x)
    y = np.empty_like(x)
    chunk_size = n // num_workers

    tasks = [
        (
            x,
            y,
            a,
            b,
            i * chunk_size,
            (i + 1) * chunk_size if i < num_workers - 1 else n,
        )
        for i in range(num_workers)
    ]

    start = time.perf_counter()
    list(executor.map(worker_task, tasks))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    num_workers = os.cpu_count() or 4
    rng = np.random.default_rng(int(STUDENT_ID))

    a, b = 2.5, 1.5

    print("=== TASK 5: The Overhead Crossover Point ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {num_workers}\n")

    print(
        f"{'Array Size (N)':<15} | {'Serial (ms)':<12} | {'Parallel (ms)':<12} | {'Parallel Faster?':<16}"
    )
    print("-" * 65)

    crossover_range = None
    prev_faster = False

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, N in enumerate(N_SIZES):
            x = rng.random(N, dtype=np.float64)

            t_serial = run_serial(x, a, b)
            t_parallel = run_parallel(x, a, b, num_workers, executor)

            faster = t_parallel < t_serial
            faster_str = "YES" if faster else "NO"

            print(
                f"{N:<15,} | {t_serial:>12.4f} | {t_parallel:>12.4f} | {faster_str:^16}"
            )

            # Detect the exact transition point where parallel becomes faster
            if faster and not prev_faster and crossover_range is None:
                if idx > 0:
                    crossover_range = (N_SIZES[idx - 1], N)
                else:
                    crossover_range = (0, N)
            prev_faster = faster

    print("-" * 65)
    if crossover_range:
        print(
            f"Exact Crossover Point: Parallel execution shifts from SLOWER to FASTER between N = {crossover_range[0]:,} and N = {crossover_range[1]:,}"
        )
    else:
        print(
            "Crossover Point: Serial execution remained faster for all tested N up to 10,000,000 due to thread dispatch overhead relative to vector SIMD instructions."
        )


if __name__ == "__main__":
    main()