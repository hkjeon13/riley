#define _GNU_SOURCE
#include <gnu/libc-version.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const size_t request = 800;
    unsigned char *target = malloc(request);
    unsigned char *guard = malloc(32);
    if (target == NULL || guard == NULL) {
        free(target);
        free(guard);
        return 1;
    }

    *(volatile unsigned char *)target = 17;
    *(volatile unsigned char *)guard = 23;
    const size_t usable = malloc_usable_size(target);
    const size_t before = mallinfo2().uordblks;
    free(target);
    const size_t after = mallinfo2().uordblks;

    const char *setting = getenv("GLIBC_TUNABLES");
    const int requested_disable =
        setting != NULL && strcmp(setting, "glibc.malloc.tcache_count=0") == 0;
    printf("{\"uid\": %lu,\"glibc\": \"%s\",\"request_bytes\": %zu,\"usable_bytes\": %zu,"
           "\"uordblks_before\": %zu,\"uordblks_after\": %zu,\"explicitly_disabled\": %s,"
           "\"other_tunable_present\": %s}\n",
           (unsigned long)getuid(), gnu_get_libc_version(), request, usable, before, after,
           requested_disable ? "true" : "false", setting != NULL && !requested_disable ? "true" : "false");
    free(guard);
    return 0;
}
