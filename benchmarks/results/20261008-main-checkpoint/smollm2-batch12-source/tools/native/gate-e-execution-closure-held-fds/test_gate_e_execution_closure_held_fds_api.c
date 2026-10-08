#include "gate_e_execution_closure_held_fds_v1.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    struct gate_e_execution_closure_held_fds_v1 output;

    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(output.dynamic_loader.descriptor == -1);
    assert(output.interpreter.descriptor == -1);
    assert(output.runtime_leaf_count == 0U);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert(gate_e_execution_closure_held_fds_v1_close(NULL) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1);
    (void)puts("gate-e execution closure held-fds API link test passed");
    return 0;
}
