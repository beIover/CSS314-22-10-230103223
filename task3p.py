# Author: Leilima Akat (2N 07P)
# Student ID: 230103223

import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

STUDENT_ID = "230103223"
N = 2_000_000


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


def pass1_worker(args):
    """Pass 1: Compute local inclusive prefix sum and chunk total."""
    arr, start_idx, end_idx = args
    chunk = arr[start_idx:end_idx]
    local_cumsum = np.cumsum(chunk, dtype=np.int64)
    return local_cumsum, local_cumsum[-1]


def pass2_worker(args):
    """Pass 2: Apply chunk offset to local prefix sum."""
    local_cumsum, offset = args
    return local_cumsum + offset


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    num_threads = os.cpu_count() or 4
    seed = int(STUDENT_ID)

    print("=== TASK 3: Parallel Prefix Scan (Inclusive) ===")
    print("Author: Leilima Akat (2N 07P)")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {num_threads}")
    print(f"Array Size: {N:,} integers (values 1 to 10)\n")

    # Generate 2,000,000 random integers (1 to 10) using student ID seed
    rng = np.random.default_rng(seed)
    input_arr = rng.integers(1, 11, size=N, dtype=np.int64)

    # 1. Serial inclusive prefix sum
    start_serial = time.perf_counter()
    serial_output = np.cumsum(input_arr, dtype=np.int64)
    time_serial_ms = (time.perf_counter() - start_serial) * 1000

    # 2. Parallel 2-pass chunked prefix scan
    start_parallel = time.perf_counter()
    chunk_size = N // num_threads
    chunks_args = [
        (
            input_arr,
            i * chunk_size,
            (i + 1) * chunk_size if i < num_threads - 1 else N,
        )
        for i in range(num_threads)
    ]

    # Pass 1: Local prefix sums & chunk totals
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pass1_results = list(executor.map(pass1_worker, chunks_args))

    local_cumsums, local_totals = zip(*pass1_results)

    # Offset reduction (Pass 1.5): Prefix sum of chunk totals
    offsets = np.zeros(num_threads, dtype=np.int64)
    offsets[1:] = np.cumsum(local_totals[:-1], dtype=np.int64)

    # Pass 2: Apply offsets
    pass2_args = list(zip(local_cumsums, offsets))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pass2_results = list(executor.map(pass2_worker, pass2_args))

    parallel_output = np.concatenate(pass2_results)
    time_parallel_ms = (time.perf_counter() - start_parallel) * 1000

    # 3. Verification constraint
    try:
        assert (
            parallel_output[-1] == serial_output[-1]
        ), "Mismatch in final prefix sum value!"
        assert np.array_equal(
            parallel_output, serial_output
        ), "Mismatch in array contents!"
        verification_status = "PASS"
    except AssertionError as e:
        verification_status = "FAIL"
        print(f"VERIFICATION ERROR: {e}")

    speedup = (
        time_serial_ms / time_parallel_ms if time_parallel_ms > 0 else 1.0
    )

    print(
        f"{'Serial Time':<15} | {'Parallel Time':<15} | {'Speedup':<10} | {'Verification'}"
    )
    print("-" * 62)
    print(
        f"{time_serial_ms:>12.2f} ms | {time_parallel_ms:>12.2f} ms | {speedup:>8.2f}x | {verification_status}"
    )

    if verification_status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()