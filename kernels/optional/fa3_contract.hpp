#pragma once
#include "fa3_api.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace riley_fa3 {
struct Layout {
    int batch = 0, total_q = 0, max_q = 0, max_k = 0, columns = 0, rounded_b = 0;
    size_t table = 0, qptr = 0, lengths = 0, scheduler = 0, lse = 0, bytes = 0;
    std::vector<int> metadata;
};
inline size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
inline int make_layout(const RileyFa3Spec *s, Layout &l) {
    if (!s || s->abi_version != 1 || s->mode > 1 || !s->batch || s->batch > 32 ||
        !s->total_q || s->total_q > 1024 || !s->physical_pages || s->physical_pages > 8192 ||
        !s->q_indptr || !s->kv_lengths || !s->page_indptr || !s->pages ||
        !s->page_entries || s->page_entries > 8192) return RILEY_FA3_INVALID;
    if (s->q_indptr[0] || s->page_indptr[0] ||
        s->q_indptr[s->batch] != int(s->total_q) ||
        s->page_indptr[s->batch] != int(s->page_entries)) return RILEY_FA3_INVALID;
    l = Layout{};
    l.batch = s->batch; l.total_q = s->total_q;
    for (unsigned i = 0; i < s->batch; ++i) {
        // Check signed boundaries before subtraction or indexing caller arrays.
        int q0 = s->q_indptr[i], q1 = s->q_indptr[i+1];
        int p0 = s->page_indptr[i], p1 = s->page_indptr[i+1], k = s->kv_lengths[i];
        if (q0 < 0 || q1 <= q0 || q1 > int(s->total_q) || p0 < 0 || p1 < p0 ||
            p1 > int(s->page_entries) || k < 1 || k > 4096 || q1-q0 > k ||
            p1-p0 != (k+15)/16 || (s->mode == 1 && q1-q0 != 1)) return RILEY_FA3_INVALID;
        for (int p = p0; p < p1; ++p)
            if (s->pages[p] < 0 || s->pages[p] >= int(s->physical_pages)) return RILEY_FA3_INVALID;
        l.max_q = std::max(l.max_q, q1-q0); l.max_k = std::max(l.max_k, k);
    }
    l.columns = (l.max_k+15)/16;
    l.max_k = l.columns*16; // Upstream page-table shape uses seqlen_k/page_size.
    l.rounded_b = (l.batch+3)/4*4;
    l.qptr = align256(l.batch*l.columns*sizeof(int));
    l.lengths = align256(l.qptr+(l.batch+1)*sizeof(int));
    l.scheduler = align256(l.lengths+l.batch*sizeof(int));
    // Four batch vectors plus semaphore; decode's unused head-swizzle vector is harmless.
    l.lse = align256(l.scheduler+(4*l.rounded_b+1)*sizeof(int));
    l.bytes = align256(l.lse+l.total_q*9*sizeof(float));
    l.metadata.assign(l.scheduler/sizeof(int), 0);
    for (unsigned i = 0; i < s->batch; ++i) {
        std::copy(s->pages+s->page_indptr[i], s->pages+s->page_indptr[i+1],
                  l.metadata.begin()+i*l.columns);
        l.metadata[l.lengths/sizeof(int)+i] = s->kv_lengths[i];
    }
    std::copy(s->q_indptr, s->q_indptr+s->batch+1, l.metadata.begin()+l.qptr/sizeof(int));
    return RILEY_FA3_OK;
}
}
