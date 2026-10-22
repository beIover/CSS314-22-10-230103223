import os
import platform
import subprocess
import threading
import time
from queue import Queue

STUDENT_ID = "230103223"
ITEMS = 1000
NUM_WORKERS = 4
CHUNK_SIZE = 5


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


def run_static():
    """Strategy A: Static Chunk Distribution (equal blocks per core)."""
    chunk_len = ITEMS // NUM_WORKERS
    worker_chunks = [
        list(range(w * chunk_len, (w + 1) * chunk_len))
        for w in range(NUM_WORKERS)
    ]

    finish_times = [0.0] * NUM_WORKERS

    def static_worker(w_id, items):
        for i in items:
            delay = 0.0001 * (i % 20)
            time.sleep(delay)
        finish_times[w_id] = time.perf_counter()

    start_time = time.perf_counter()
    threads = []
    for w in range(NUM_WORKERS):
        t = threading.Thread(
            target=static_worker, args=(w, worker_chunks[w])
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_time = (time.perf_counter() - start_time) * 1000
    delta = (max(finish_times) - min(finish_times)) * 1000
    return total_time, delta


def run_dynamic():
    """Strategy B: Dynamic Scheduling / Work-Stealing Queue (chunk size = 5)."""
    work_queue = Queue()
    for c_start in range(0, ITEMS, CHUNK_SIZE):
        work_queue.put(
            list(range(c_start, min(c_start + CHUNK_SIZE, ITEMS)))
        )

    finish_times = [0.0] * NUM_WORKERS

    def dynamic_worker(w_id):
        while not work_queue.empty():
            try:
                chunk = work_queue.get_nowait()
            except Exception:
                break
            for i in chunk:
                delay = 0.0001 * (i % 20)
                time.sleep(delay)
            work_queue.task_done()
        finish_times[w_id] = time.perf_counter()

    start_time = time.perf_counter()
    threads = []
    for w in range(NUM_WORKERS):
        t = threading.Thread(target=dynamic_worker, args=(w,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_time = (time.perf_counter() - start_time) * 1000
    delta = (max(finish_times) - min(finish_times)) * 1000
    return total_time, delta


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    print("=== TASK 4: Work Stealing & Scheduling Under Skew ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Total Work Items: {ITEMS}")
    print(f"Dynamic Chunk Size: {CHUNK_SIZE}\n")

    static_time, static_delta = run_static()
    dynamic_time, dynamic_delta = run_dynamic()

    print(
        f"{'Scheduling Strategy':<30} | {'Total Time (ms)':<15} | {'Finish Delta (ms)':<18}"
    )
    print("-" * 68)
    print(
        f"{'Strategy A (Static Chunking)':<30} | {static_time:>15.2f} | {static_delta:>18.2f}"
    )
    print(
        f"{'Strategy B (Dynamic / Queue)':<30} | {dynamic_time:>15.2f} | {dynamic_delta:>18.2f}"
    )
    print("-" * 68)
    print(
        f"Thread Delta Reduction: {static_delta - dynamic_delta:.2f} ms"
    )


if __name__ == "__main__":
    main()