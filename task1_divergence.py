
import numpy as np
import time
from numba import cuda


# Kernel A: Uniform Path
@cuda.jit
def kernel_a_uniform(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        x = d_arr[idx]

        for _ in range(1000):
            x = x * 1.000001 + 0.000001

        d_arr[idx] = x


# Kernel B: Full Divergence
@cuda.jit
def kernel_b_divergence(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        x = d_arr[idx]

        if idx % 2 == 0:
            for _ in range(1000):
                x = x * 1.000001 + 0.000001
        else:
            for _ in range(1000):
                x = x - 0.000001
                x = x / 1.000001

        d_arr[idx] = x


# Kernel C: Warp-Aligned Branching
@cuda.jit
def kernel_c_warp_aligned(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        x = d_arr[idx]

        warp_id = idx // 32

        if warp_id % 2 == 0:
            for _ in range(1000):
                x = x * 1.000001 + 0.000001
        else:
            for _ in range(1000):
                x = x - 0.000001
                x = x / 1.000001

        d_arr[idx] = x


def benchmark_kernel(kernel, d_arr, threads_per_block=256, trials=10):

    blocks_per_grid = (d_arr.size + threads_per_block - 1) // threads_per_block

    # Warm-up
    kernel[blocks_per_grid, threads_per_block](d_arr)
    cuda.synchronize()

    times = []

    for _ in range(trials):
        start = time.perf_counter()

        kernel[blocks_per_grid, threads_per_block](d_arr)

        cuda.synchronize()

        end = time.perf_counter()

        times.append(end - start)

    return np.mean(times)


def main():

    N = 100000

    h_arr = np.ones(N, dtype=np.float32)

    d_arr = cuda.to_device(h_arr)

    time_a = benchmark_kernel(kernel_a_uniform, d_arr)

    d_arr = cuda.to_device(h_arr)
    time_b = benchmark_kernel(kernel_b_divergence, d_arr)

    d_arr = cuda.to_device(h_arr)
    time_c = benchmark_kernel(kernel_c_warp_aligned, d_arr)

    print("=" * 60)
    print("TASK 1: WARP DIVERGENCE MICROBENCHMARK")
    print("=" * 60)

    print(f"Kernel A - Uniform Path:          {time_a:.6f} seconds")
    print(f"Kernel B - Full Divergence:       {time_b:.6f} seconds")
    print(f"Kernel C - Warp-Aligned Branching:{time_c:.6f} seconds")

    print("=" * 60)


if __name__ == "__main__":
    main()
