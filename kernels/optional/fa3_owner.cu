#include "../src/ffi_internal.hpp"
#include "fa3_api.h"
#include <new>

struct RileyCudaFa3Owner {
    RileyFa3Plan *plan=nullptr;
    RileyCudaStream *stream=nullptr;
    RileyCudaDeviceBuffer *buffers[4]{};
    bool close_failed=false;
};
namespace {
using namespace riley_cuda_internal;
constexpr const char *op="FA3 native owner";
RileyCudaStatus status(int code, RileyCudaErrorInfo *error) {
    if (!code) return RILEY_CUDA_STATUS_SUCCESS;
    return validation_error(error, code==RILEY_FA3_UNSUPPORTED?RILEY_CUDA_STATUS_NOT_SUPPORTED:
        code==RILEY_FA3_INVALID?RILEY_CUDA_STATUS_INVALID_ARGUMENT:RILEY_CUDA_STATUS_RUNTIME_ERROR,
        RILEY_CUDA_ERROR_STAGE_PREPARE,op,"experimental FA3 operation failed (not numerical qualification)");
}
void release(RileyCudaFa3Owner *p) {
    for(auto *b:p->buffers) b->active_uses.store(0,std::memory_order_release);
    p->stream->active_uses.store(0,std::memory_order_release);
}
}
extern "C" RileyCudaStatus riley_cuda_fa3_owner_create(
    RileyCudaStream *stream, RileyCudaDeviceBuffer *q, RileyCudaDeviceBuffer *k,
    RileyCudaDeviceBuffer *v, RileyCudaDeviceBuffer *o, const RileyFa3Spec *spec,
    RileyCudaFa3Owner **out, RileyCudaErrorInfo *error) noexcept {
    clear_error(error);
    if(!out) return status(RILEY_FA3_INVALID,error);
    *out=nullptr;
    if(!stream || !q || !k || !v || !o || !spec) return status(RILEY_FA3_INVALID,error);
    RileyCudaDeviceBuffer *buffers[]={q,k,v,o};
    for(int i=0;i<4;++i) {
        if(!same_context(stream->owner,buffers[i]->owner)) return status(RILEY_FA3_INVALID,error);
        for(int j=0;j<i;++j) if(buffers[i]==buffers[j]) return status(RILEY_FA3_INVALID,error);
    }
    if(command_batch_is_active(stream)) return status(RILEY_FA3_INVALID,error);
    CurrentContext current(stream->owner);
    auto result=current.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,op);
    if(result!=RILEY_CUDA_STATUS_SUCCESS) return result;
    auto *owner=new(std::nothrow) RileyCudaFa3Owner;
    if(!owner) return current.leave(status(RILEY_FA3_INTERNAL,error),error,RILEY_CUDA_ERROR_STAGE_PREPARE,op);
    owner->stream=stream;
    for(int i=0;i<4;++i) owner->buffers[i]=buffers[i];
    std::atomic<uint32_t> *uses[]={&stream->active_uses,&q->active_uses,&k->active_uses,&v->active_uses,&o->active_uses};
    int acquired=0;
    for(;acquired<5;++acquired) {
        uint32_t expected=0;
        if(!uses[acquired]->compare_exchange_strong(expected,1,std::memory_order_acq_rel)) break;
    }
    if(acquired!=5) {
        for(int i=0;i<acquired;++i) uses[i]->store(0,std::memory_order_release);
        delete owner;
        return current.leave(status(RILEY_FA3_INVALID,error),error,RILEY_CUDA_ERROR_STAGE_PREPARE,op);
    }
    RileyFa3Buffers raw{q->device_data,k->device_data,v->device_data,o->device_data,
                        q->byte_len,k->byte_len,v->byte_len,o->byte_len};
    result=status(riley_fa3_create(spec,&raw,stream->stream,&owner->plan),error);
    auto restored=current.leave(result,error,RILEY_CUDA_ERROR_STAGE_PREPARE,op);
    if(result!=RILEY_CUDA_STATUS_SUCCESS) { release(owner); delete owner; return restored; }
    if(restored!=RILEY_CUDA_STATUS_SUCCESS) {
        // The plan exists but context restoration is ambiguous: keep all leases.
        return restored;
    }
    *out=owner;
    return RILEY_CUDA_STATUS_SUCCESS;
}
extern "C" RileyCudaStatus riley_cuda_fa3_owner_operation(
    RileyCudaFa3Owner *owner, uint32_t operation, RileyCudaErrorInfo *error) noexcept {
    clear_error(error);
    if(!owner || operation>2) return status(RILEY_FA3_INVALID,error);
    if(owner->close_failed) return status(RILEY_FA3_CUDA,error);
    CurrentContext current(owner->stream->owner);
    auto result=current.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,op);
    if(result!=RILEY_CUDA_STATUS_SUCCESS) return result;
    if(operation==0) result=status(riley_fa3_enqueue(owner->plan),error);
    if(operation==1) result=cudaStreamSynchronize(owner->stream->stream)==cudaSuccess?
        RILEY_CUDA_STATUS_SUCCESS:status(RILEY_FA3_CUDA,error);
    if(operation==2) result=status(riley_fa3_destroy(&owner->plan),error);
    result=current.leave(result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,op);
    if(operation==2 && result!=RILEY_CUDA_STATUS_SUCCESS) owner->close_failed=true;
    if(operation==2 && result==RILEY_CUDA_STATUS_SUCCESS) { release(owner); delete owner; }
    // Failed close retains leases even when native destruction succeeded but
    // context restoration failed. Reuse is forbidden; the poisoned context stays alive.
    return result;
}
