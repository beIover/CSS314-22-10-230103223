import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT_ID = "230103223"
N = 20_000_000
P_PARALLEL = 0.95  # 95% параллелизуемая часть по закону Амдала


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


def compute_chunk(args):
    """Worker function to generate and sum a chunk of the array."""
    chunk_size, seed_offset = args
    rng = np.random.default_rng(int(STUDENT_ID) + seed_offset)
    chunk_data = rng.random(chunk_size, dtype=np.float64)
    return np.sum(chunk_data, dtype=np.float64)


def run_parallel_benchmark(num_workers):
    """Run parallel summation across specified number of workers."""
    chunk_size = N // num_workers
    remainder = N % num_workers

    # Подготовка параметров для каждого воркера
    tasks = []
    for i in range(num_workers):
        current_chunk = chunk_size + (remainder if i == 0 else 0)
        tasks.append((current_chunk, i))

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_chunk, tasks))
        _total_sum = sum(results)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    max_cores = os.cpu_count() or 4
    # Формируем список уникальных количеств воркеров: [1, 2, 4, max_cores, max_cores * 2]
    worker_counts = sorted(
        list(set([1, 2, 4, max_cores, max_cores * 2]))
    )

    print("=== TASK 2: Multi-Core Scaling vs. Amdahl's Law ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {max_cores}")
    print(f"Array Size: {N:,}\n")

    # Вычисляем T(1) — baseline время для 1 воркера
    t1_time = run_parallel_benchmark(1)

    print(
        f"{'Cores':<8} | {'Wall Time':<12} | {'Speedup S(p)':<14} | {'Amdahl Upper Bound':<18}"
    )
    print("-" * 60)

    for p in worker_counts:
        wall_time = (
            t1_time if p == 1 else run_parallel_benchmark(p)
        )
        speedup = t1_time / wall_time
        amdahl_bound = 1.0 / ((1.0 - P_PARALLEL) + (P_PARALLEL / p))

        print(
            f"{p:<8} | {wall_time:>8.2f} ms | {speedup:>12.2f}x | {amdahl_bound:>16.2f}x"
        )


if __name__ == "__main__":
    main()