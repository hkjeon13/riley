#define _GNU_SOURCE
#include <fcntl.h>
#include <malloc.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    if (!mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024) || !mallopt(M_TRIM_THRESHOLD, -1)) {
        return 2;
    }
    const size_t length = 8 * 1024 * 1024;
    void *buffer = malloc(length);
    if (buffer == NULL) {
        return 2;
    }
    volatile char *pages = buffer;
    for (size_t offset = 0; offset < length; offset += 4096) {
        pages[offset] = 42;
    }
    free(buffer);
    if (close(creat("/tmp/w5-trim-smoke.ready", 0600)) != 0) {
        return 2;
    }
    for (int attempt = 0; attempt < 200; ++attempt) {
        if (access("/tmp/w5-trim-smoke.end", F_OK) == 0) {
            return 0;
        }
        usleep(100000);
    }
    return 2;
}
