import os
import platform
import subprocess

STUDENT_ID = "230103223"
NUM_THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED_TOTAL = NUM_THREADS * INCREMENTS_PER_THREAD


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

    compiler = "clang" if platform.system() == "Darwin" else "gcc"

    # Compile task6.c into native binary
    compile_cmd = [compiler, "-O1", "-pthread", "task6.c", "-o", "task6_bench"]
    comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)

    if comp_res.returncode != 0:
        print("Compilation failed:\n", comp_res.stderr)
        return

    # Run native benchmark binary
    run_res = subprocess.run(["./task6_bench"], capture_output=True, text=True)
    actual_unsafe, actual_sync, time_unsafe, time_sync = map(
        float, run_res.stdout.strip().split(",")
    )

    actual_unsafe = int(actual_unsafe)
    actual_sync = int(actual_sync)

    lost_updates = EXPECTED_TOTAL - actual_unsafe
    loss_pct = (lost_updates / EXPECTED_TOTAL) * 100.0
    time_penalty = time_sync / time_unsafe if time_unsafe > 0 else 1.0

    print("=== TASK 6: Synchronization Cost & Data Loss ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per Thread: {INCREMENTS_PER_THREAD:,}\n")

    print(f"{'Metric':<35} | {'Value':<20}")
    print("-" * 58)
    print(f"{'Expected Final Value':<35} | {EXPECTED_TOTAL:>20,}")
    print(f"{'Unsynchronized Final Value':<35} | {actual_unsafe:>20,}")
    print(f"{'Synchronized Final Value':<35} | {actual_sync:>20,}")
    print(f"{'Lost Updates':<35} | {lost_updates:>20,}")
    print(f"{'Data Loss Percentage':<35} | {loss_pct:>19.2f}%")
    print("-" * 58)
    print(f"{'Unsynchronized Execution Time':<35} | {time_unsafe:>17.2f} ms")
    print(f"{'Synchronized Execution Time':<35} | {time_sync:>17.2f} ms")
    print(f"{'Synchronization Time Penalty':<35} | {time_penalty:>19.2f}x slowdown")


if __name__ == "__main__":
    main()