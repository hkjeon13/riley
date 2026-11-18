#define _GNU_SOURCE

#include <errno.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct resident_pages {
    unsigned long rss;
    unsigned long anonymous;
    unsigned long file;
    unsigned long shared;
};

static int resident_pages(struct resident_pages *pages) {
    FILE *status = fopen("/proc/self/status", "r");
    if (status == NULL) {
        return -1;
    }
    unsigned int found = 0;
    char line[256];
    while (fgets(line, sizeof(line), status) != NULL) {
        found += (sscanf(line, "VmRSS: %lu kB", &pages->rss) == 1);
        found += (sscanf(line, "RssAnon: %lu kB", &pages->anonymous) == 1);
        found += (sscanf(line, "RssFile: %lu kB", &pages->file) == 1);
        found += (sscanf(line, "RssShmem: %lu kB", &pages->shared) == 1);
    }
    fclose(status);
    return found == 4 ? 0 : -1;
}

static void *sample_loop(void *arg) {
    char *path = arg;
    FILE *out = fopen(path, "wx");
    free(path);
    struct timespec deadline;
    if (out == NULL || clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
        perror("Riley allocator sampler setup");
        if (out != NULL) {
            fclose(out);
        }
        return NULL;
    }
    for (unsigned int i = 0; i < 160; i++) {
        struct timespec timestamp;
        struct resident_pages pages = {0};
        if (clock_gettime(CLOCK_REALTIME, &timestamp) != 0 || resident_pages(&pages) != 0) {
            fputs("Riley allocator sampler could not read process metrics\n", stderr);
            break;
        }
        const struct mallinfo2 allocation = mallinfo2();
        if (fprintf(out,
                    "{\"pid\":%ld,\"unix_seconds\":%ld.%09ld,"
                    "\"rss_kib\":%lu,\"anonymous_kib\":%lu,\"file_kib\":%lu,\"shared_kib\":%lu,"
                    "\"arena_bytes\":%zu,\"arena_in_use_bytes\":%zu,\"arena_free_bytes\":%zu,"
                    "\"malloc_mmap_bytes\":%zu}\n",
                    (long)getpid(), (long)timestamp.tv_sec, timestamp.tv_nsec, pages.rss,
                    pages.anonymous, pages.file, pages.shared, allocation.arena, allocation.uordblks,
                    allocation.fordblks, allocation.hblkhd) < 0 || fflush(out) != 0) {
            perror("Riley allocator sampler output");
            break;
        }
        deadline.tv_sec += 15;
        int error;
        do {
            error = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
        } while (error == EINTR);
        if (error != 0) {
            fprintf(stderr, "Riley allocator sampler sleep error %d\n", error);
            break;
        }
    }
    fclose(out);
    return NULL;
}

__attribute__((constructor)) static void start_allocator_sampler(void) {
    const char *path = getenv("RILEY_W5_ALLOCATOR_OUTPUT");
    if (path == NULL || *path == '\0') {
        return;
    }
    char *output = strdup(path);
    if (output == NULL) {
        perror("Riley allocator sampler path");
        return;
    }
    pthread_t thread;
    const int error = pthread_create(&thread, NULL, sample_loop, output);
    if (error == 0) {
        (void)pthread_detach(thread);
    } else {
        free(output);
        fprintf(stderr, "Riley allocator sampler thread error %d\n", error);
    }
}
