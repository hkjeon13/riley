#include "gate_e_root_bundle_held_v1.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    struct gate_e_root_bundle_held_v1 held;

    gate_e_root_bundle_held_v1_init(&held);
    for (size_t index = 0; index < GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1; ++index) {
        assert(held.directories[index].descriptor == -1);
    }
    assert(held.manifest.descriptor == -1);
    assert(held.bootstrap.descriptor == -1);
    assert(held.core.descriptor == -1);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == GATE_E_ROOT_BUNDLE_INVALID_ARGUMENT_V1);
    assert(gate_e_root_bundle_held_v1_close(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    assert(gate_e_root_bundle_held_v1_close(NULL) == GATE_E_ROOT_BUNDLE_INVALID_ARGUMENT_V1);
    (void)puts("gate-e root bundle held API link test passed");
    return 0;
}
