#pragma once

#include <cstddef>
#include <cstdint>

// A cold, fixed-capacity index for a thread-owned resource ledger. The ledger
// remains authoritative for acquisition, release order and completion failure.
// This index only avoids rescanning that ledger for every repeated operand.
template <typename T, size_t Capacity>
struct RileyCudaCommandBatchIndex {
  static_assert(Capacity != 0 && (Capacity & (Capacity - 1)) == 0);
  T* slots[Capacity]{};

  size_t find(T* pointer) const noexcept {
    uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    size_t slot = static_cast<size_t>(key) & (Capacity - 1);
    for (size_t probe = 0; probe < Capacity; ++probe) {
      if (slots[slot] == nullptr || slots[slot] == pointer) return slot;
      slot = (slot + 1) & (Capacity - 1);
    }
    return Capacity;
  }

  void clear() noexcept {
    for (auto& slot : slots) slot = nullptr;
  }
};
