# Author: Leilima Akat (2N 07P)
# Student ID: 230103223

import hashlib
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = "230103223"
NUM_TASKS = 10_000


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


def cpu_bound_task(task_id):
    """CPU-bound task: Hashing 1KB data 500 times using PBKDF2 (releases GIL in CPython)."""
    salt = f"salt_{STUDENT_ID}_{task_id}".encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", b"password_data_padding_123", salt, 500)


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    num_cores = os.cpu_count() or 4

    # Target pool sizes: [1, Cores/2, Cores, Cores*2, Cores*4, Cores*16, 256]
    raw_pools = [
        1,
        max(1, num_cores // 2),
        num_cores,
        num_cores * 2,
        num_cores * 4,
        num_cores * 16,
        256,
    ]

    # Deduplicate while preserving order
    pool_sizes = []
    for p in raw_pools:
        if p not in pool_sizes:
            pool_sizes.append(p)

    print("=== TASK 5: Thread Oversubscription Threshold ===")
    print("Author: Leilima Akat (2N 07P)")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {num_cores}")
    print(f"Total CPU Tasks: {NUM_TASKS:,}\n")

    results = []
    t_baseline = None

    for p in pool_sizes:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            list(executor.map(cpu_bound_task, range(NUM_TASKS)))
        t_ms = (time.perf_counter() - t0) * 1000

        if p == 1:
            t_baseline = t_ms

        speedup = t_baseline / t_ms
        results.append((p, t_ms, speedup))

    # Identify optimal pool size (lowest execution time)
    min_time_idx = min(range(len(results)), key=lambda i: results[i][1])
    best_pool = results[min_time_idx][0]

    print(
        f"{'Pool Size':<12} | {'Execution Time (ms)':<22} | {'Speedup vs 1 Thread':<22} | {'Status'}"
    )
    print("-" * 88)

    first_degradation_pool = None

    for idx, (p, t_ms, speedup) in enumerate(results):
        if p == 1:
            status = "BASELINE"
        elif p == best_pool:
            status = "OPTIMAL (Peak Efficiency)"
        elif idx > min_time_idx:
            if first_degradation_pool is None:
                first_degradation_pool = p
                status = "OVERSUBSCRIBED (Performance Degradation Starts)"
            else:
                status = "OVERSUBSCRIBED (Context Switch Overhead)"
        elif p < best_pool:
            status = "UNDERUTILIZED"
        else:
            status = "NORMAL"

        print(f"{p:<12} | {t_ms:>22.2f} | {speedup:>20.2f}x | {status}")

    print("-" * 88)
    if first_degradation_pool:
        print(
            f"Oversubscription Saturation Point: Execution time begins worsening at Pool Size = {first_degradation_pool} "
            f"due to OS thread context switching overhead."
        )


if __name__ == "__main__":
    main()