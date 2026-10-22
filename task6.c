#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

#define NUM_THREADS 8
#define INCREMENTS_PER_THREAD 1000000ULL

long long mutex_counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
_Atomic long long atomic_counter = 0;

void* worker_mutex(void* arg) {
    for (long long i = 0; i < INCREMENTS_PER_THREAD; i++) {
        pthread_mutex_lock(&lock);
        mutex_counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void* worker_cas(void* arg) {
    for (long long i = 0; i < INCREMENTS_PER_THREAD; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    return NULL;
}

double get_time_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++) pthread_create(&threads[i], NULL, worker_mutex, NULL);
    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_mutex_ms = get_time_ms(start, end);

    atomic_counter = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < NUM_THREADS; i++) pthread_create(&threads[i], NULL, worker_cas, NULL);
    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_cas_ms = get_time_ms(start, end);

    printf("%.2f,%.2f\n", time_mutex_ms, time_cas_ms);
    return 0;
}
