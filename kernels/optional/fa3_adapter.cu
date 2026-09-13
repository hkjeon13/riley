#include "fa3_contract.hpp"
#include "flash_fwd_launch_template.h"
#include <cuda.h>
#include <cstdint>
#include <memory>

struct RileyFa3Plan {
    Flash_fwd_params params{};
    cudaStream_t stream{};
    CUcontext context{};
    void *workspace = nullptr;
    bool decode = false;
    bool close_failed = false;
};
namespace {
struct ApiError { int code; };
void runtime(cudaError_t e) { if (e != cudaSuccess) throw ApiError{RILEY_FA3_CUDA}; }
void driver(CUresult e) { if (e != CUDA_SUCCESS) throw ApiError{RILEY_FA3_CUDA}; }
void require(bool b, int code = RILEY_FA3_INVALID) { if (!b) throw ApiError{code}; }
template<class F> int boundary(F f) noexcept {
    try { f(); return RILEY_FA3_OK; }
    catch (const ApiError &e) { return e.code; }
    catch (const RileyFa3CudaError &) { return RILEY_FA3_CUDA; }
    catch (const RileyFa3CutlassError &) { return RILEY_FA3_CUDA; }
    catch (...) { return RILEY_FA3_INTERNAL; }
}
void context_check(RileyFa3Plan &p, bool cold) {
    CUcontext current = nullptr, stream_context = nullptr;
    driver(cuCtxGetCurrent(&current));
    driver(cuStreamGetCtx(reinterpret_cast<CUstream>(p.stream), &stream_context));
    require(current && current == p.context && stream_context == current, RILEY_FA3_CONTEXT);
    if (cold) {
        cudaStreamCaptureStatus state;
        runtime(cudaStreamIsCapturing(p.stream, &state));
        require(state == cudaStreamCaptureStatusNone, RILEY_FA3_CAPTURE);
    }
}
void dispatch(RileyFa3Plan &p, bool prepare_only) {
    if (p.decode)
        run_flash_fwd<90,64,64,1,cutlass::bfloat16_t,cutlass::bfloat16_t,
                      false,false,false,true,true,false,false,true,false,false>(p.params,p.stream,prepare_only);
    else
        run_flash_fwd<90,64,64,1,cutlass::bfloat16_t,cutlass::bfloat16_t,
                      true,false,false,true,true,false,false,true,false,false>(p.params,p.stream,prepare_only);
}
void check_buffer(const void *ptr, uint64_t bytes, uint64_t needed, int device) {
    require(ptr && bytes >= needed && reinterpret_cast<uintptr_t>(ptr)%16 == 0);
    cudaPointerAttributes attributes{};
    runtime(cudaPointerGetAttributes(&attributes, ptr));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device);
    CUdeviceptr base = 0; size_t extent = 0;
    driver(cuMemGetAddressRange(&base, &extent, reinterpret_cast<CUdeviceptr>(ptr)));
    auto offset = reinterpret_cast<CUdeviceptr>(ptr)-base;
    require(offset <= extent && bytes <= extent-offset);
}
}
extern "C" int riley_fa3_validate(const RileyFa3Spec *s, uint64_t *bytes) noexcept {
    if (bytes) *bytes = 0;
    return boundary([&] {
        require(bytes); riley_fa3::Layout l;
        require(riley_fa3::make_layout(s,l) == RILEY_FA3_OK);
        *bytes = l.bytes;
    });
}
extern "C" int riley_fa3_create(const RileyFa3Spec *s, const RileyFa3Buffers *buffers,
                                void *stream, RileyFa3Plan **out) noexcept {
    if (!out) return RILEY_FA3_INVALID;
    *out = nullptr;
    return boundary([&] {
        require(buffers && stream && stream != reinterpret_cast<void*>(1) && stream != reinterpret_cast<void*>(2));
        riley_fa3::Layout l;
        require(riley_fa3::make_layout(s,l) == RILEY_FA3_OK);
        auto p = std::make_unique<RileyFa3Plan>();
        p->stream = reinterpret_cast<cudaStream_t>(stream);
        driver(cuCtxGetCurrent(&p->context));
        context_check(*p,true);
        int device = 0; runtime(cudaGetDevice(&device));
        cudaDeviceProp properties{}; runtime(cudaGetDeviceProperties(&properties,device));
        int tensor_map = 0, cluster = 0;
        driver(cuDeviceGetAttribute(&tensor_map,CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,device));
        runtime(cudaDeviceGetAttribute(&cluster,cudaDevAttrClusterLaunch,device));
        require(properties.major == 9 && properties.minor == 0 && tensor_map && cluster,
                RILEY_FA3_UNSUPPORTED);
        const void *ptrs[] = {buffers->q,buffers->k,buffers->v,buffers->o};
        uint64_t sizes[] = {buffers->q_bytes,buffers->k_bytes,buffers->v_bytes,buffers->o_bytes};
        uint64_t qbytes = uint64_t(s->total_q)*9*64*2;
        uint64_t kvbytes = uint64_t(s->physical_pages)*3*16*64*2;
        for (int i=0;i<4;++i) check_buffer(ptrs[i],sizes[i],i==0||i==3?qbytes:kvbytes,device);
        for (int i=0;i<4;++i) for (int j=i+1;j<4;++j) {
            uintptr_t a=reinterpret_cast<uintptr_t>(ptrs[i]), b=reinterpret_cast<uintptr_t>(ptrs[j]);
            require(a<b ? sizes[i]<=b-a : sizes[j]<=a-b);
        }
        runtime(cudaMalloc(&p->workspace,l.bytes));
        try {
            runtime(cudaMemcpyAsync(p->workspace,l.metadata.data(),l.scheduler,cudaMemcpyHostToDevice,p->stream));
            runtime(cudaStreamSynchronize(p->stream)); // Cold copy completes before host metadata dies.
            auto &f=p->params;
            f.q_ptr=const_cast<void*>(buffers->q); f.k_ptr=const_cast<void*>(buffers->k);
            f.v_ptr=const_cast<void*>(buffers->v); f.o_ptr=buffers->o;
            f.q_row_stride=f.o_row_stride=9*64; f.q_head_stride=f.o_head_stride=64;
            f.k_row_stride=f.v_row_stride=64; f.k_head_stride=f.v_head_stride=16*64;
            f.k_batch_stride=f.v_batch_stride=3*16*64; f.v_dim_stride=1;
            f.h=9; f.h_k=3; f.b=s->batch; f.b_k=s->batch;
            f.seqlen_q=l.max_q; f.seqlen_k=l.max_k; f.total_q=s->total_q;
            f.d=f.dv=f.d_rounded=f.dv_rounded=64;
            f.seqlen_q_rounded=(l.max_q+127)/128*128; f.seqlen_k_rounded=(l.max_k+127)/128*128;
            f.scale_softmax=0.125f; f.num_splits=1; f.is_bf16=true;
            f.pack_gqa=true; p->decode=s->mode==1; f.is_causal=!p->decode;
            f.window_size_left=-1; f.window_size_right=p->decode?-1:0;
            f.arch=90; f.num_sm=properties.multiProcessorCount;
            f.page_size=16; f.num_pages=s->physical_pages; f.page_table_batch_stride=l.columns;
            auto at=[&](size_t n){return reinterpret_cast<int*>(static_cast<char*>(p->workspace)+n);};
            f.page_table=at(l.table); f.cu_seqlens_q=at(l.qptr); f.seqused_k=at(l.lengths);
            f.num_splits_dynamic_ptr=at(l.scheduler); f.num_m_blocks_ptr=at(l.scheduler)+l.rounded_b;
            f.varlen_batch_idx_ptr=at(l.scheduler)+2*l.rounded_b;
            f.num_nheads_in_l2_ptr=p->decode?nullptr:at(l.scheduler)+3*l.rounded_b;
            f.tile_count_semaphore=at(l.scheduler)+4*l.rounded_b;
            f.varlen_sort_batches=true; f.head_swizzle=!p->decode;
            f.softmax_lse_ptr=at(l.lse);
            dispatch(*p,true); // Attributes/TMA descriptor validation, no GPU launch.
        } catch (...) {
            // No attention was enqueued. A failed cold copy is drained before release.
            cudaStreamSynchronize(p->stream); cudaFree(p->workspace); throw;
        }
        *out=p.release();
    });
}
extern "C" int riley_fa3_enqueue(RileyFa3Plan *p) noexcept {
    return boundary([&] { require(p); require(!p->close_failed,RILEY_FA3_CUDA); context_check(*p,false); dispatch(*p,false); });
}
extern "C" int riley_fa3_destroy(RileyFa3Plan **p) noexcept {
    return boundary([&] {
        require(p && *p); require(!(*p)->close_failed,RILEY_FA3_CUDA); context_check(**p,true);
        // CUDA free can report a deferred error after an uncertain side effect.
        // Never issue a second destructive free after any close-stage failure.
        (*p)->close_failed=true;
        runtime(cudaStreamSynchronize((*p)->stream));
        runtime(cudaFree((*p)->workspace)); delete *p; *p=nullptr;
    });
}
