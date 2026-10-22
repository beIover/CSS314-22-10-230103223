#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

#define ITERATIONS 50000000ULL

typedef struct {
    int thread_id;
    volatile uint64_t *shared_array;
} ThreadArgs;

// 1. Unpadded Shared Access (Triggers False Sharing)
// All threads write to adjacent 8-byte integers in the same 64-byte cache line.
void* run_unpadded(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int id = args->thread_id;
    for (uint64_t i = 0; i < ITERATIONS; i++) {
        args->shared_array[id]++;
    }
    return NULL;
}

// 2. Local Accumulation / Padded (No Cache Line Contention)
// Each thread updates a register/stack variable and writes to shared memory once.
void* run_local(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int id = args->thread_id;
    volatile uint64_t local_count = 0;
    for (uint64_t i = 0; i < ITERATIONS; i++) {
        local_count++;
    }
    args->shared_array[id] = local_count;
    return NULL;
}

double get_time_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main(int argc, char** argv) {
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    volatile uint64_t *unpadded_array = calloc(num_threads, sizeof(uint64_t));
    volatile uint64_t *local_array = calloc(num_threads, sizeof(uint64_t));

    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];

    struct timespec start, end;

    // Benchmark 1: Unpadded Shared Array
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].shared_array = unpadded_array;
        pthread_create(&threads[i], NULL, run_unpadded, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double unpadded_time = get_time_ms(start, end);

    // Benchmark 2: Local Accumulation
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].shared_array = local_array;
        pthread_create(&threads[i], NULL, run_local, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double local_time = get_time_ms(start, end);

    printf("%.2f,%.2f,%.2f\n", unpadded_time, local_time, unpadded_time / local_time);

    free((void*)unpadded_array);
    free((void*)local_array);
    return 0;
}