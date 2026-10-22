# Author: Leilima Akat (2N 07P)
# Student ID: 230103223

import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import subprocess
import time
import numpy as np

STUDENT_ID = "230103223"
N_SIZES = [100_000, 1_000_000, 10_000_000]


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


def ipc_worker_pipe(conn):
    """Path A Worker: Receives serialized array over Pipe."""
    data = conn.recv()
    array_sum = np.sum(data, dtype=np.float64)
    conn.send(array_sum)


def ipc_worker_shm(shm_name, shape, dtype, conn):
    """Path B Worker: Zero-copy access via SharedMemory buffer."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    array_sum = np.sum(arr, dtype=np.float64)
    existing_shm.close()
    conn.send(array_sum)


def main():
    if not str(STUDENT_ID).isdigit():
        raise ValueError("Replace STUDENT_ID with your numeric student ID.")

    num_cores = os.cpu_count() or 4
    seed = int(STUDENT_ID)
    rng = np.random.default_rng(seed)

    print("=== TASK 4: Inter-Process Memory Transfer Overhead ===")
    print("Author: Leilima Akat (2N 07P)")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {num_cores}\n")

    print(
        f"{'Array Size (N)':<15} | {'Path A - IPC Pipe (ms)':<22} | {'Path B - Zero-Copy SHM (ms)':<25} | {'Speedup':<10}"
    )
    print("-" * 80)

    for N in N_SIZES:
        data = rng.random(N, dtype=np.float64)

        # -------------------------------------------------------------
        # Path A: IPC Pipe / Queue Serialization (Pickle Copy)
        # -------------------------------------------------------------
        parent_conn_a, child_conn_a = mp.Pipe()
        p_a = mp.Process(target=ipc_worker_pipe, args=(child_conn_a,))

        t0 = time.perf_counter()
        p_a.start()
        parent_conn_a.send(data)
        _res_a = parent_conn_a.recv()
        p_a.join()
        time_a_ms = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------------------
        # Path B: Zero-Copy Shared Memory (multiprocessing.shared_memory)
        # -------------------------------------------------------------
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shm_arr[:] = data[:]  # Zero-copy buffer write

        parent_conn_b, child_conn_b = mp.Pipe()
        p_b = mp.Process(
            target=ipc_worker_shm,
            args=(shm.name, data.shape, data.dtype, child_conn_b),
        )

        t0 = time.perf_counter()
        p_b.start()
        _res_b = parent_conn_b.recv()
        p_b.join()
        time_b_ms = (time.perf_counter() - t0) * 1000

        shm.close()
        shm.unlink()

        speedup = time_a_ms / time_b_ms if time_b_ms > 0 else 1.0
        print(
            f"{N:<15,} | {time_a_ms:>22.2f} | {time_b_ms:>25.2f} | {speedup:>8.2f}x"
        )


if __name__ == "__main__":
    main()