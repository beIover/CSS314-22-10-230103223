# Author: Leilima Akat (2N 07P)
# Student ID: 230103223

import math
import os
import platform
import subprocess
import threading
import time
import numpy as np

STUDENT_ID = "230103223"
TOTAL_SAMPLES = 40_000_000
NUM_THREADS = 4
SAMPLES_PER_THREAD = TOTAL_SAMPLES // NUM_THREADS
CHUNK_SIZE = 5_000


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


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    seed_base = int(STUDENT_ID)
    num_cores = os.cpu_count() or 4

    print("=== TASK 2: Pseudo-Random Generator State Bottleneck ===")
    print("Author: Leilima Akat (2N 07P)")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {num_cores}")
    print(f"Total Sample Points: {TOTAL_SAMPLES:,}")
    print(f"Threads: {NUM_THREADS}\n")

    # -------------------------------------------------------------
    # 1. Shared PRNG Instance Guarded by Synchronization / Mutex Lock
    # -------------------------------------------------------------
    shared_rng = np.random.default_rng(seed_base)
    shared_lock = threading.Lock()
    results_shared = [0] * NUM_THREADS

    def worker_shared(idx):
        inside = 0
        num_chunks = SAMPLES_PER_THREAD // CHUNK_SIZE
        for _ in range(num_chunks):
            with shared_lock:
                x = shared_rng.random(CHUNK_SIZE, dtype=np.float64)
                y = shared_rng.random(CHUNK_SIZE, dtype=np.float64)
            inside += np.sum(x * x + y * y <= 1.0)
        results_shared[idx] = inside

    start_shared = time.perf_counter()
    threads_shared = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker_shared, args=(i,))
        t.start()
        threads_shared.append(t)

    for t in threads_shared:
        t.join()

    elapsed_shared_ms = (time.perf_counter() - start_shared) * 1000
    pi_shared = 4.0 * sum(results_shared) / TOTAL_SAMPLES
    error_shared = abs(pi_shared - math.pi)

    # -------------------------------------------------------------
    # 2. Independent PRNG per Thread (Thread-Isolated State)
    # -------------------------------------------------------------
    results_indep = [0] * NUM_THREADS

    def worker_indep(idx):
        thread_seed = (seed_base * 10) + idx
        local_rng = np.random.default_rng(thread_seed)
        x = local_rng.random(SAMPLES_PER_THREAD, dtype=np.float64)
        y = local_rng.random(SAMPLES_PER_THREAD, dtype=np.float64)
        inside = np.sum(x * x + y * y <= 1.0)
        results_indep[idx] = inside

    start_indep = time.perf_counter()
    threads_indep = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker_indep, args=(i,))
        t.start()
        threads_indep.append(t)

    for t in threads_indep:
        t.join()

    elapsed_indep_ms = (time.perf_counter() - start_indep) * 1000
    pi_indep = 4.0 * sum(results_indep) / TOTAL_SAMPLES
    error_indep = abs(pi_indep - math.pi)

    # -------------------------------------------------------------
    # Results & Speedup Calculation
    # -------------------------------------------------------------
    speedup = (
        elapsed_shared_ms / elapsed_indep_ms if elapsed_indep_ms > 0 else 1.0
    )

    print(
        f"{'PRNG Mode':<30} | {'Computed pi':<12} | {'Abs Error':<12} | {'Wall Time':<12}"
    )
    print("-" * 72)
    print(
        f"{'1. Shared PRNG (Mutex Guarded)':<30} | {pi_shared:>12.6f} | {error_shared:>12.6e} | {elapsed_shared_ms:>8.2f} ms"
    )
    print(
        f"{'2. Independent PRNG (Isolated)':<30} | {pi_indep:>12.6f} | {error_indep:>12.6e} | {elapsed_indep_ms:>8.2f} ms"
    )
    print("-" * 72)
    print(f"Speedup Factor (Shared / Independent): {speedup:.2f}x")


if __name__ == "__main__":
    main()