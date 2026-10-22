import os
import platform
import subprocess

STUDENT_ID = "230103223"
ITERATIONS = 50_000_000


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

    num_cores = os.cpu_count() or 4
    compiler = "clang" if platform.system() == "Darwin" else "gcc"

    # Compile C code automatically
    compile_cmd = [
        compiler,
        "-O2",
        "-pthread",
        "task3.c",
        "-o",
        "task3_bench",
    ]
    comp_res = subprocess.run(
        compile_cmd, capture_output=True, text=True
    )

    if comp_res.returncode != 0:
        print("Compilation failed:\n", comp_res.stderr)
        return

    # Execute benchmark binary
    run_res = subprocess.run(
        ["./task3_bench", str(num_cores)],
        capture_output=True,
        text=True,
    )
    unpadded_ms, local_ms, ratio = map(
        float, run_res.stdout.strip().split(",")
    )

    print(
        "=== TASK 3: Cache Line Contention (False Sharing) ==="
    )
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Worker Threads (Cores): {num_cores}")
    print(f"Iterations per Thread: {ITERATIONS:,}\n")

    print(f"{'Benchmark Mode':<32} | {'Wall Time':<12}")
    print("-" * 48)
    print(
        f"{'Unpadded Shared (False Sharing)':<32} | {unpadded_ms:>8.2f} ms"
    )
    print(
        f"{'Local Accumulation / Padded':<32} | {local_ms:>8.2f} ms"
    )
    print("-" * 48)
    print(
        f"Execution Ratio (Unpadded / Local): {ratio:.2f}x"
    )


if __name__ == "__main__":
    main()