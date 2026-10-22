import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 12345678  # Replace with your actual Student ID
N = 8000
NUM_THREADS = os.cpu_count() or 4

def row_major_worker(start_row, end_row, matrix):
    return np.sum(matrix[start_row:end_row, :])

def col_major_worker(start_col, end_col, matrix):
    total = 0.0
    for j in range(start_col, end_col):
        total += np.sum(matrix[:, j])
    return total

def main():
    np.random.seed(STUDENT_ID)
    matrix = np.random.rand(N, N).astype(np.float64)
    chunk = N // NUM_THREADS

    # Mode A: Row-Major Traversal
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures_a = [
            executor.submit(row_major_worker, t * chunk, (t + 1) * chunk if t < NUM_THREADS - 1 else N, matrix)
            for t in range(NUM_THREADS)
        ]
        _ = sum(f.result() for f in futures_a)
    time_a_ms = (time.perf_counter() - t0) * 1000

    # Mode B: Column-Major Traversal
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures_b = [
            executor.submit(col_major_worker, t * chunk, (t + 1) * chunk if t < NUM_THREADS - 1 else N, matrix)
            for t in range(NUM_THREADS)
        ]
        _ = sum(f.result() for f in futures_b)
    time_b_ms = (time.perf_counter() - t0) * 1000

    penalty = time_b_ms / time_a_ms

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical Core Count: {NUM_THREADS}")
    print(f"Mode A (Row-Major) Runtime: {time_a_ms:.2f} ms")
    print(f"Mode B (Column-Major) Runtime: {time_b_ms:.2f} ms")
    print(f"Cache Penalty Factor: {penalty:.2f}x")

if __name__ == "__main__":
    main()