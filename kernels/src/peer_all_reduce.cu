#include "ffi_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

// Creation and close of the tensor-parallel all-reduce group. The execute
// path, which shares the primitive span validation, lives in primitives.cu.

namespace {

using riley_cuda_internal::CaptureDomainControlLease;
using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::driver_error;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::release_child;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::set_error;
using riley_cuda_internal::validation_error;

using Group = RileyCudaPeerAllReduceGroup;

constexpr const char* kCreate = "create peer all-reduce group";
constexpr const char* kClose = "close peer all-reduce group";

// Every rank's primary-context capture domain is held for the whole create or
// close: these are broad control operations (peer access, allocation, free).
class ControlLeases final {
 public:
  bool acquire(RileyCudaContext* const* contexts, uint32_t count) noexcept {
    for (uint32_t rank = 0; rank < count; ++rank) {
      leases_[rank].reset(new (std::nothrow) CaptureDomainControlLease(
          contexts[rank]->capture_domain));
      if (!leases_[rank] || !leases_[rank]->active()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::unique_ptr<CaptureDomainControlLease> leases_[Group::kMaximumRanks];
};

// Frees whatever the group holds in each rank's context. Every pointer is
// attempted once and cleared, so a retry can never double-free.
RileyCudaStatus free_rank_memory(Group* group, RileyCudaErrorInfo* error,
                                 uint32_t stage, const char* operation) noexcept {
  RileyCudaStatus first = RILEY_CUDA_STATUS_SUCCESS;
  for (uint32_t rank = 0; rank < group->rank_count; ++rank) {
    if (group->staging[rank] == nullptr && group->signals[rank] == nullptr) {
      continue;
    }
    CurrentContext scope(group->contexts[rank]);
    RileyCudaStatus status = scope.enter(error, stage, operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      if (group->staging[rank] != nullptr) {
        status = runtime_error(cudaFree(group->staging[rank]), error, stage,
                               "free peer all-reduce staging buffer");
        group->staging[rank] = nullptr;
      }
      if (group->signals[rank] != nullptr) {
        const RileyCudaStatus signal_status = runtime_error(
            cudaFree(group->signals[rank]), error, stage,
            "free peer all-reduce signal area");
        group->signals[rank] = nullptr;
        if (status == RILEY_CUDA_STATUS_SUCCESS) {
          status = signal_status;
        }
      }
    }
    status = scope.leave(status, error, stage, operation);
    if (first == RILEY_CUDA_STATUS_SUCCESS) {
      first = status;
    }
  }
  return first;
}

void release_children(Group* group, uint32_t retained) noexcept {
  for (uint32_t rank = 0; rank < retained; ++rank) {
    (void)release_child(group->contexts[rank]);
  }
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_peer_all_reduce_group_create(
    RileyCudaContext* const* contexts, uint32_t rank_count,
    uint64_t max_elements, RileyCudaPeerAllReduceGroup** out_group,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_group == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                            "out_group is null");
  }
  *out_group = nullptr;
  if (contexts == nullptr || rank_count < 2 ||
      rank_count > Group::kMaximumRanks) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                            "two to eight contexts are required");
  }
  if (max_elements == 0 || max_elements > (UINT64_C(1) << 40)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                            "max_elements must be in 1..2^40");
  }
  for (uint32_t rank = 0; rank < rank_count; ++rank) {
    if (contexts[rank] == nullptr) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                              "a context is null");
    }
    for (uint32_t other = 0; other < rank; ++other) {
      if (contexts[other]->ordinal == contexts[rank]->ordinal) {
        return validation_error(error, RILEY_CUDA_STATUS_INVALID_DEVICE,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                                "every rank must use a distinct device");
      }
    }
  }
  for (uint32_t rank = 0; rank < rank_count; ++rank) {
    for (uint32_t other = 0; other < rank_count; ++other) {
      if (other == rank) {
        continue;
      }
      int accessible = 0;
      const CUresult result = cuDeviceCanAccessPeer(
          &accessible, contexts[rank]->device, contexts[other]->device);
      if (result != CUDA_SUCCESS) {
        return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreate);
      }
      if (accessible == 0) {
        return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
                                "a device pair has no peer access");
      }
    }
  }
  ControlLeases leases;
  if (!leases.acquire(contexts, rank_count)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreate,
        "a CUDA primary context has an active graph capture or broad control operation");
  }

  void* storage = std::calloc(1, sizeof(Group));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kCreate,
                     "host allocation failed");
  }
  auto* group = new (storage) Group{};
  group->rank_count = rank_count;
  group->max_elements = max_elements;
  uint32_t retained = 0;
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  for (uint32_t rank = 0; rank < rank_count; ++rank) {
    group->contexts[rank] = contexts[rank];
    if (!retain_child(contexts[rank])) {
      status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE, kCreate,
                              "context child-resource counter overflow");
      break;
    }
    retained = rank + 1;
  }

  const size_t staging_bytes = static_cast<size_t>(max_elements) * 2;
  const size_t signal_bytes = Group::kSignalWords * sizeof(uint32_t);
  for (uint32_t rank = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && rank < rank_count; ++rank) {
    CurrentContext scope(contexts[rank]);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CREATE, kCreate);
    for (uint32_t other = 0;
         status == RILEY_CUDA_STATUS_SUCCESS && other < rank_count; ++other) {
      if (other == rank) {
        continue;
      }
      // Peer access is a property of the context pair and stays enabled after
      // close; enabling it twice is not an error.
      const CUresult result = cuCtxEnablePeerAccess(contexts[other]->context, 0);
      if (result != CUDA_SUCCESS &&
          result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        status = driver_error(result, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                              "enable peer access");
      }
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(cudaMalloc(&group->staging[rank], staging_bytes),
                             error, RILEY_CUDA_ERROR_STAGE_CREATE,
                             "allocate peer all-reduce staging buffer");
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      void* signals = nullptr;
      status = runtime_error(cudaMalloc(&signals, signal_bytes), error,
                             RILEY_CUDA_ERROR_STAGE_CREATE,
                             "allocate peer all-reduce signal area");
      group->signals[rank] = static_cast<uint32_t*>(signals);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(cudaMemset(group->signals[rank], 0, signal_bytes),
                             error, RILEY_CUDA_ERROR_STAGE_CREATE,
                             "clear peer all-reduce signal area");
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(cudaDeviceSynchronize(), error,
                             RILEY_CUDA_ERROR_STAGE_CREATE,
                             "complete peer all-reduce setup");
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE, kCreate);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    RileyCudaErrorInfo ignored{};
    ignored.struct_size = sizeof(ignored);
    (void)free_rank_memory(group, &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
                           "roll back peer all-reduce group");
    release_children(group, retained);
    group->~Group();
    std::free(group);
    return status;
  }
  *out_group = group;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_peer_all_reduce_group_close(
    RileyCudaPeerAllReduceGroup** group, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (group == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kClose,
                            "group pointer is null");
  }
  if (*group == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  Group* const owned = *group;
  ControlLeases leases;
  if (!leases.acquire(owned->contexts, owned->rank_count)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kClose,
        "a CUDA primary context has an active graph capture or broad control operation");
  }
  // Recorded launches may still be in flight on any rank: drain every device
  // before its staging memory, which peers read, is freed.
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  for (uint32_t rank = 0; rank < owned->rank_count; ++rank) {
    CurrentContext scope(owned->contexts[rank]);
    RileyCudaStatus rank_status =
        scope.enter(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kClose);
    if (rank_status == RILEY_CUDA_STATUS_SUCCESS) {
      rank_status = runtime_error(cudaDeviceSynchronize(), error,
                                  RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                                  "drain peer all-reduce before close");
    }
    rank_status = scope.leave(rank_status, error,
                              RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kClose);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = rank_status;
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    // Keep ownership: freeing memory a peer may still read is not safe.
    return status;
  }
  status = free_rank_memory(owned, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
  release_children(owned, owned->rank_count);
  owned->~Group();
  std::free(owned);
  *group = nullptr;
  return status;
}
