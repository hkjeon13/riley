#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef int (*mallctl_fn)(const char *, void *, size_t *, void *, size_t);
enum { chunk_size = 16 * 1024, chunk_count = 2048, hold_stride = 128 };

static long rss_kib(void) {
    FILE *file = fopen("/proc/self/smaps_rollup", "r");
    if (file == NULL) return -1;
    char line[256];
    long value = -1;
    while (fgets(line, sizeof line, file) != NULL) {
        if (sscanf(line, "Rss: %ld kB", &value) == 1) break;
    }
    fclose(file);
    return value;
}

int main(void) {
    Dl_info provider;
    if (!dladdr((void *) malloc, &provider)) return 1;
    mallctl_fn ctl = (mallctl_fn) dlsym(RTLD_DEFAULT, "mallctl");
    const char *version = "";
    ssize_t dirty = -1, muzzy = -1;
    size_t threshold = 1;
    if (ctl != NULL) {
        size_t length = sizeof version;
        if (ctl("version", &version, &length, NULL, 0) != 0) return 2;
        length = sizeof dirty;
        if (ctl("opt.dirty_decay_ms", &dirty, &length, NULL, 0) != 0) return 2;
        length = sizeof muzzy;
        if (ctl("opt.muzzy_decay_ms", &muzzy, &length, NULL, 0) != 0) return 2;
        length = sizeof threshold;
        if (ctl("opt.oversize_threshold", &threshold, &length, NULL, 0) != 0) return 2;
    }
    long ready = rss_kib();
    unsigned char *chunks[chunk_count];
    for (int i = 0; i < chunk_count; ++i) {
        chunks[i] = malloc(chunk_size);
        if (chunks[i] == NULL) return 3;
        for (int offset = 0; offset < chunk_size; offset += 4096) {
            ((volatile unsigned char *) chunks[i])[offset] = 1;
        }
    }
    long allocated = rss_kib();
    for (int i = 0; i < chunk_count; ++i) {
        if (i % hold_stride != 0) free(chunks[i]);
    }
    long partially_freed = rss_kib();
    for (int i = 0; i < chunk_count; i += hold_stride) free(chunks[i]);
    if (ready < 0 || allocated < 0 || partially_freed < 0) return 4;
    printf("{\"malloc_provider\":\"%s\",\"version\":\"%s\","
           "\"dirty_decay_ms\":%zd,\"muzzy_decay_ms\":%zd,\"oversize_threshold\":%zu,"
           "\"rss_ready_kib\":%ld,\"rss_allocated_kib\":%ld,\"rss_partially_freed_kib\":%ld}\n",
           provider.dli_fname, version, dirty, muzzy, threshold, ready, allocated, partially_freed);
}
