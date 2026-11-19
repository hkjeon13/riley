#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void *wait_for_probe(void *arg) {
    (void)arg;
    for (;;) {
        int requested = open("/tmp/riley-w5-trim.request", O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        if (requested >= 0) {
            close(requested);
            break;
        }
        if (errno != ENOENT) {
            perror("W5 trim trigger");
            return NULL;
        }
        const struct timespec interval = {.tv_nsec = 250000000};
        nanosleep(&interval, NULL);
    }

    const struct mallinfo2 before = mallinfo2();
    const int released = malloc_trim(0);
    const struct mallinfo2 after = mallinfo2();
    const char *pending = "/tmp/riley-w5-trim.receipt.tmp";
    FILE *output = fopen(pending, "wx");
    if (output == NULL) {
        perror("W5 trim receipt");
        return NULL;
    }
    const int written = fprintf(
        output, "{\"pid\":%ld,\"malloc_trim_return\":%d,"
        "\"before\":{\"arena\":%zu,\"in_use\":%zu,\"free\":%zu,\"mmap\":%zu},"
        "\"after\":{\"arena\":%zu,\"in_use\":%zu,\"free\":%zu,\"mmap\":%zu}}\n",
        (long)getpid(), released, before.arena, before.uordblks, before.fordblks, before.hblkhd,
        after.arena, after.uordblks, after.fordblks, after.hblkhd);
    const int closed = fclose(output);
    if (written < 0 || closed != 0 || rename(pending, "/tmp/riley-w5-trim.receipt.json") != 0) {
        perror("W5 trim receipt");
    }
    return NULL;
}

__attribute__((constructor)) static void bind_probe_to_executable(void) {
    const char *expected = getenv("RILEY_W5_TRIM_EXE");
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
    pthread_t thread;
    int error = pthread_create(&thread, NULL, wait_for_probe, NULL);
    if (error == 0) {
        (void)pthread_detach(thread);
    } else {
        fprintf(stderr, "W5 trim thread error %d\n", error);
    }
}
