#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static unsigned long period_ms;

static uint64_t ns(struct timespec stamp) {
    return (uint64_t)stamp.tv_sec * UINT64_C(1000000000) + (uint64_t)stamp.tv_nsec;
}

static void *reclaim_periodically(void *unused) {
    (void)unused;
    int out = open("/tmp/riley-w5-periodic-trim.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
    if (out < 0) {
        perror("W5 periodic trim telemetry");
        return NULL;
    }
    struct timespec completed;
    if (clock_gettime(CLOCK_MONOTONIC, &completed) != 0) {
        close(out);
        return NULL;
    }
    for (unsigned long index = 1;; ++index) {
        struct timespec deadline = completed;
        deadline.tv_sec += (time_t)(period_ms / 1000);
        deadline.tv_nsec += (long)(period_ms % 1000) * 1000000;
        if (deadline.tv_nsec >= 1000000000) {
            ++deadline.tv_sec;
            deadline.tv_nsec -= 1000000000;
        }
        int error;
        do {
            error = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
        } while (error == EINTR);
        struct timespec started;
        if (error != 0 || clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
            break;
        }
        const int returned = malloc_trim(0);
        if (clock_gettime(CLOCK_MONOTONIC, &completed) != 0) {
            break;
        }
        char line[256];
        int count = snprintf(line, sizeof(line),
                             "{\"pid\":%ld,\"period_ms\":%lu,\"index\":%lu,\"started_ns\":%llu,"
                             "\"completed_ns\":%llu,\"returned\":%d}\n",
                             (long)getpid(), period_ms, index, (unsigned long long)ns(started),
                             (unsigned long long)ns(completed), returned);
        if (count < 0 || (size_t)count >= sizeof(line) || write(out, line, (size_t)count) != count) {
            break;
        }
    }
    close(out);
    return NULL;
}

__attribute__((constructor)) static void start_periodic_trim(void) {
    const char *expected = getenv("RILEY_W5_PERIODIC_TRIM_EXE");
    char actual[4096];
    if (expected == NULL || *expected == '\0') {
        return;
    }
    const ssize_t count = readlink("/proc/self/exe", actual, sizeof(actual));
    if (count < 0 || (size_t)count >= sizeof(actual)) {
        return;
    }
    actual[count] = '\0';
    if (strcmp(actual, expected) != 0) {
        return;
    }
    const char *interval = getenv("RILEY_W5_PERIODIC_TRIM_MS");
    char *end;
    if (interval == NULL || *interval < '0' || *interval > '9') {
        fprintf(stderr, "W5 invalid periodic trim interval\n");
        return;
    }
    errno = 0;
    period_ms = strtoul(interval, &end, 10);
    if (errno != 0 || *end != '\0' || period_ms == 0 || period_ms > 600000) {
        fprintf(stderr, "W5 invalid periodic trim interval\n");
        return;
    }
    pthread_t thread;
    int error = pthread_create(&thread, NULL, reclaim_periodically, NULL);
    if (error == 0) {
        (void)pthread_detach(thread);
    } else {
        fprintf(stderr, "W5 periodic trim thread error %d\n", error);
    }
}
