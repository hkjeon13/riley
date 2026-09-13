#define RILEY_FA3_METADATA_IMPLEMENTATION
#include "fa3_model_metadata.cuh"
#include "fa3_api.h"
extern "C" uint64_t riley_fa3_model_workspace_bytes() noexcept {
    return sizeof(riley_fa3_model::Workspace);
}
extern "C" int riley_fa3_model_metadata_prepare(void *stream, const void *packet,
    uint64_t packet_bytes, void *workspace, uint64_t bytes, uint32_t physical,
    uint32_t capacity, uint32_t context, void *status) noexcept {
    if(!packet || !workspace || !status || packet_bytes<riley_fa3_model::PacketWords*4 ||
       bytes!=sizeof(riley_fa3_model::Workspace) || reinterpret_cast<uintptr_t>(workspace)%256 ||
       !physical || physical>4096 || !capacity || capacity>1024 || !context || context>4096)
        return cudaErrorInvalidValue;
    riley_fa3_model::prepare<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned*>(packet),static_cast<riley_fa3_model::Workspace*>(workspace),
        physical,capacity,context,static_cast<unsigned*>(status));
    return cudaGetLastError();
}
