
import os
import platform
import subprocess
import time

import numpy as np

STUDENT_ID = '230103223'
N = 20_000_000


def get_cpu_model():
    """Get the CPU model on macOS, with a portable fallback."""
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
    if not STUDENT_ID.isdigit():
        raise ValueError(
            "Replace STUDENT_ID with your numeric student ID."
        )

    seed = int(STUDENT_ID)

    print("=== TASK 1: Hardware Probe & Serial Baseline ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {os.cpu_count()}")
    print(f"Array Size: {N:,}")
    print("Data Type: float64")
    print("Execution Mode: Serial (single thread)")

    # Measure the complete workload, including array generation.
    start = time.perf_counter()

    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)

    # NumPy reductions; no multiprocessing or explicit parallel workers.
    total_sum = np.sum(data, dtype=np.float64)
    std_dev = np.std(data, dtype=np.float64, ddof=0)

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Sum: {total_sum:.10f}")
    print(f"Standard Deviation (population): {std_dev:.10f}")
    print(f"Execution Time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
