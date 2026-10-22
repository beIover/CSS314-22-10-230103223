#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4
#define INCREMENTS 2000000ULL

volatile long long unsynchronized_counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
long long synchronized_counter = 0;

// 1. Unsynchronized Worker (Exposes Race Conditions)
void* worker_unsafe(void* arg) {
    for (long long i = 0; i < INCREMENTS; i++) {
        unsynchronized_counter++;
    }
    return NULL;
}

// 2. Synchronized Worker (Uses Mutex Lock)
void* worker_sync(void* arg) {
    for (long long i = 0; i < INCREMENTS; i++) {
        pthread_mutex_lock(&lock);
        synchronized_counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

double get_time_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct timespec start, end;

    // Unsynchronized Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_unsafe, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_unsafe = get_time_ms(start, end);

    // Synchronized Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_sync, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_sync = get_time_ms(start, end);

    printf("%lld,%lld,%.2f,%.2f\n", unsynchronized_counter, synchronized_counter, time_unsafe, time_sync);
    return 0;
}