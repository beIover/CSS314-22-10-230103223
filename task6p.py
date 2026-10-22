
# Author: Leilima Akat (2N 07P)
# Student ID: 230103223

import os
import platform
import subprocess

STUDENT_ID = "230103223"
NUM_THREADS = 8
INCREMENTS_PER_THREAD = 1_000_000
TOTAL_OPS = NUM_THREADS * INCREMENTS_PER_THREAD


def get_cpu_model():
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

    compiler = "clang" if platform.system() == "Darwin" else "gcc"

    compile_cmd = [compiler, "-O2", "-pthread", "task6.c", "-o", "task6_bench"]
    comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)

    if comp_res.returncode != 0:
        print("Compilation failed:\n", comp_res.stderr)
        return

    run_res = subprocess.run(["./task6_bench"], capture_output=True, text=True)
    time_mutex_ms, time_cas_ms = map(float, run_res.stdout.strip().split(","))

    mutex_ops_sec = TOTAL_OPS / (time_mutex_ms / 1000.0)
    cas_ops_sec = TOTAL_OPS / (time_cas_ms / 1000.0)
    ratio = cas_ops_sec / mutex_ops_sec if mutex_ops_sec > 0 else 1.0

    print("=== TASK 6: Atomic CAS vs. OS Mutex Under Heavy Contention ===")
    print("Author: Leilima Akat (2N 07P)")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per Thread: {INCREMENTS_PER_THREAD:,}")
    print(f"Total Operations: {TOTAL_OPS:,}\n")

    print(
        f"{'Synchronization Primitive':<32} | {'Execution Time (ms)':<20} | {'Throughput (ops/sec)':<25}"
    )
    print("-" * 82)
    print(
        f"{'1. OS Mutex / Heavyweight Lock':<32} | {time_mutex_ms:>18.2f} ms | {mutex_ops_sec:>23,.2f} ops/sec"
    )
    print(
        f"{'2. Hardware Atomic CAS / Fetch-Add':<32} | {time_cas_ms:>18.2f} ms | {cas_ops_sec:>23,.2f} ops/sec"
    )
    print("-" * 82)
    print(f"Throughput Ratio (CAS / Mutex): {ratio:.2f}x speedup")


if __name__ == "__main__":
    main()
