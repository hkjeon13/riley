#include "gate_e_root_bundle_sealed_leaves_v1.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    struct gate_e_root_bundle_sealed_leaves_v1 pair;

    gate_e_root_bundle_sealed_leaves_v1_init(&pair);
    assert(pair.bootstrap.descriptor == -1);
    assert(pair.core.descriptor == -1);
    assert(gate_e_root_bundle_sealed_leaves_v1_close(&pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1);
    assert(gate_e_root_bundle_sealed_leaves_v1_close(NULL) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1);
    (void)puts("gate-e root bundle sealed leaves API link test passed");
    return 0;
}
