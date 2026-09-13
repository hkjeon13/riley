// Optional FlashInfer CUDA-core backend. The caller retains all device parents
// and establishes stream/context ownership. This file is built separately until
// the model recorder has an explicit non-exact numerical-profile selection.
#include "flashinfer_api.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>
#include <cstdint>
#include <cstddef>

namespace riley_flashinfer {
constexpr unsigned kRows = 32, kMaxPages = 256;
struct alignas(16) Metadata {
  int indices[kRows * kMaxPages];
  int indptr[1025], last[1024], requests[kRows], tiles[kRows];
  int output_indptr[kRows + 1], chunk_size;
  bool valid[kRows];
  alignas(16) __nv_bfloat16 mixed_output[kRows * 576];
};

static_assert(offsetof(Metadata, mixed_output) % 16 == 0, "FlashInfer vector store alignment");

// Only indices and lengths are translated; HND K/V remain in their original
// allocations. This preparation is per iteration, reusable across all layers.
template<bool Mixed>
__global__ void prepare(const uint32_t* shape, const uint32_t* active,
                        Metadata* metadata, uint32_t physical, uint32_t context,
                        uint32_t* status, const uint32_t* total, uint32_t capacity) {
  const unsigned row = threadIdx.x;
  __shared__ unsigned pages[kRows], offsets[kRows], bases[kRows];
  const unsigned count = *active;
  const bool live = count > 0 && count <= kRows && row < count &&
                    (!Mixed || shape[row * 416 + 18] == 1);
  const unsigned position = live ? shape[row * 416 + 1] : 0;
  const unsigned mapped = live && Mixed ? shape[row * 416 + 16] : row;
  bool valid = live && position < context;
  if (live && !valid) atomicOr(status, 4u);
  if constexpr(Mixed) {
    if (live && (!*total || *total > capacity || mapped >= *total ||
                 mapped >= 1024 || shape[row * 416 + 2] != 1)) {
      valid = false; atomicOr(status, 16u);
    }
    for(unsigned other=0;valid && other<count;++other) {
      if(other!=row && shape[other*416+18]==1 && shape[other*416+16]==mapped) {
        valid=false;atomicOr(status,16u);
      }
    }
  }
  pages[row] = valid ? position / 16 + 1 : 0;
  offsets[row] = mapped;
  metadata->valid[row] = valid;
  metadata->requests[row] = valid ? mapped : 0;
  metadata->tiles[row] = 0;
  metadata->output_indptr[row] = row;
  __syncthreads();
  if(row==0) {
    metadata->chunk_size=4096; metadata->output_indptr[kRows]=kRows;
    if(!count || count>kRows)atomicOr(status,2u);
    unsigned offset=0;
    for(unsigned i=0;i<kRows;++i) {
      bases[i]=offset;
      if(pages[i]) {
        // Serialize adjacent indptr endpoints; no same-value write race.
        metadata->indptr[offsets[i]]=offset;
        metadata->indptr[offsets[i]+1]=offset+pages[i];
        metadata->last[offsets[i]]=shape[i*416+1]%16+1;
      }
      offset+=pages[i];
    }
    // FlashInfer's protective loads read the final batch indptr even when only
    // 32 request blocks are launched; sparse mapped rows still need this bound.
    metadata->indptr[1024]=offset;
  }
  __syncthreads();
  for(unsigned page=0;page<pages[row];++page) {
    const auto index=shape[row*416+32+page];
    metadata->indices[bases[row]+page]=index;
    if(index>=physical){metadata->valid[row]=false;atomicOr(status,8u);}
  }
}

__global__ void scatter_mixed(const Metadata* metadata,__nv_bfloat16* output) {
  const unsigned row=blockIdx.x;
  if(!metadata->valid[row])return;
  for(unsigned i=threadIdx.x;i<576;i+=blockDim.x)
    output[metadata->requests[row]*576+i]=metadata->mixed_output[row*576+i];
}

struct Params {
  using DTypeQ = __nv_bfloat16;
  using DTypeKV = __nv_bfloat16;
  using DTypeO = __nv_bfloat16;
  using IdType = int;
  DTypeQ* q;
  flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv;
  DTypeO* o;
  float* lse;
  float sm_scale;
  uint32_t padded_batch_size, num_qo_heads;
  IdType q_stride_n, q_stride_h;
  int32_t window_left;
  bool enable_pdl;
  IdType *request_indices, *kv_tile_indices, *o_indptr, *kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;
  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t) const { return 1; }
  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch) const { return paged_kv.get_length(batch); }
};
}  // namespace riley_flashinfer

extern "C" uint64_t riley_flashinfer_decode_workspace_bytes() noexcept {
  return sizeof(riley_flashinfer::Metadata);
}
extern "C" int riley_flashinfer_decode_prepare(
    void* stream, const void* shape, const void* active, void* workspace,
    uint64_t workspace_bytes, uint32_t physical, uint32_t context, void* status) noexcept {
  if (!shape || !active || !workspace || !status ||
      workspace_bytes < sizeof(riley_flashinfer::Metadata) ||
      !physical || physical > 4096 || !context || context > 4096)
    return cudaErrorInvalidValue;
  riley_flashinfer::prepare<false><<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(shape), static_cast<const uint32_t*>(active),
      static_cast<riley_flashinfer::Metadata*>(workspace), physical, context,
      static_cast<uint32_t*>(status), nullptr, 32);
  return cudaGetLastError();
}
extern "C" int riley_flashinfer_decode_run(
    void* stream, const void* q, const void* k, const void* v, void* output,
    void* workspace, uint64_t workspace_bytes) noexcept {
  if (!q || !k || !v || !output || !workspace ||
      workspace_bytes < sizeof(riley_flashinfer::Metadata)) return cudaErrorInvalidValue;
  try {
    using namespace riley_flashinfer;
    auto* m = static_cast<Metadata*>(workspace);
    Params params{};
    params.q = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q));
    params.o = static_cast<__nv_bfloat16*>(output);
    params.paged_kv = flashinfer::paged_kv_t<__nv_bfloat16, int>(
        3, 16, 64, 1024, flashinfer::QKVLayout::kHND,
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k)),
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(v)),
        m->indices, m->indptr, m->last);
    params.sm_scale = .125f;
    params.padded_batch_size = 32; params.num_qo_heads = 9;
    params.q_stride_n = 576; params.q_stride_h = 64; params.window_left = -1;
    params.request_indices = m->requests; params.kv_tile_indices = m->tiles;
    params.o_indptr = m->output_indptr; params.kv_chunk_size_ptr = &m->chunk_size;
    params.block_valid_mask = m->valid;
    return flashinfer::BatchDecodeWithPagedKVCacheDispatched<64,
        flashinfer::PosEncodingMode::kNone,
        flashinfer::DefaultAttention<false, false, false, false>, Params>(
            params, nullptr, nullptr, false, static_cast<cudaStream_t>(stream));
  } catch (...) { return cudaErrorUnknown; }
}

extern "C" int riley_flashinfer_mixed_prepare(void* stream,const void* metadata,
    void* workspace,uint64_t bytes,uint32_t physical,uint32_t context,
    uint32_t capacity,void* status) noexcept {
  if(!metadata||!workspace||!status||bytes<sizeof(riley_flashinfer::Metadata)||
     !physical||physical>4096||!context||context>4096||!capacity||capacity>1024)
    return cudaErrorInvalidValue;
  const auto* m=static_cast<const uint32_t*>(metadata);
  riley_flashinfer::prepare<true><<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
      m+32,m+5,static_cast<riley_flashinfer::Metadata*>(workspace),physical,context,
      static_cast<uint32_t*>(status),m+9,capacity);
  return cudaGetLastError();
}
extern "C" int riley_flashinfer_mixed_run(void* stream,const void* q,const void* k,
    const void* v,void* output,void* workspace,uint64_t bytes) noexcept {
  if(!workspace||!output||bytes<sizeof(riley_flashinfer::Metadata))return cudaErrorInvalidValue;
  auto* m=static_cast<riley_flashinfer::Metadata*>(workspace);
  auto result=riley_flashinfer_decode_run(stream,q,k,v,m->mixed_output,workspace,bytes);
  if(result!=cudaSuccess)return result;
  riley_flashinfer::scatter_mixed<<<32,128,0,static_cast<cudaStream_t>(stream)>>>(m,static_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}
