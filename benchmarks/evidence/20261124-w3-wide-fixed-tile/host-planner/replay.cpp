// Host execution of the staged planner with simulated block barriers and scans.
#include <algorithm>
#include <array>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using IdType = int32_t;
using __nv_bfloat16 = uint16_t;
constexpr uint32_t kPage = 16;
constexpr uint32_t kCtaTileQ = 64;
constexpr uint32_t kPlannerThreads = 1024;
constexpr uint32_t kGroup = 5;
constexpr uint32_t kRowWidth = 8;
constexpr uint32_t kPagesPerRequest = 80;
constexpr uint32_t kSequences = 6;
constexpr IdType kUnused = -1;
constexpr __nv_bfloat16 kFill = 0x5555;
constexpr __nv_bfloat16 kNaN = 0x7fc0;
constexpr float CUDART_NAN_F = std::numeric_limits<float>::quiet_NaN();
__nv_bfloat16 __float2bfloat16_rn(float) { return kNaN; }
void atomicOr(uint32_t* target, uint32_t mask) { *target |= mask; }

struct Coordinate { uint32_t x; };
static Coordinate threadIdx{};
static constexpr Coordinate blockDim{kPlannerThreads};

class Task {
 public:
  struct promise_type;
  using Handle = std::coroutine_handle<promise_type>;
  struct promise_type {
    Task get_return_object() { return Task(Handle::from_promise(*this)); }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() { std::terminate(); }
  };

  Task(Task&& other) noexcept : handle(std::exchange(other.handle, {})) {}
  ~Task() { if (handle) handle.destroy(); }
  Handle handle;

 private:
  explicit Task(Handle task) : handle(task) {}
};

enum class Collective { None, Barrier, Scan, Exit };
static std::array<Collective, kPlannerThreads> yielded{};
static std::array<uint32_t, kPlannerThreads> scan_input{}, scan_offset{};
static uint32_t scan_total = 0;

struct Barrier {
  bool await_ready() const noexcept { return false; }
  void await_suspend(Task::Handle) const noexcept { yielded[threadIdx.x] = Collective::Barrier; }
  void await_resume() const noexcept {}
};

struct Scan {
  uint32_t value;
  uint32_t* total;
  Scan(uint32_t v, uint32_t*, uint32_t* t) : value(v), total(t) {}
  bool await_ready() const noexcept { return false; }
  void await_suspend(Task::Handle) const noexcept {
    yielded[threadIdx.x] = Collective::Scan;
    scan_input[threadIdx.x] = value;
  }
  uint32_t await_resume() const noexcept {
    *total = scan_total;
    return scan_offset[threadIdx.x];
  }
};

#include "planner.inc"

void execute(std::vector<Task>& threads, std::mt19937& random) {
  std::array<uint32_t, kPlannerThreads> order{};
  std::iota(order.begin(), order.end(), 0U);
  for (int step = 0; step < 20; ++step) {
    std::shuffle(order.begin(), order.end(), random);
    for (uint32_t thread : order) {
      threadIdx.x = thread;
      yielded[thread] = Collective::None;
      auto handle = threads[thread].handle;
      assert(!handle.done());
      handle.resume();
      if (handle.done()) yielded[thread] = Collective::Exit;
    }
    const Collective event = yielded.front();
    if (event == Collective::None || !std::all_of(yielded.begin(), yielded.end(),
                                                 [event](Collective value) { return value == event; })) {
      throw std::runtime_error("threads reached different block collectives");
    }
    if (event == Collective::Exit) return;
    if (event == Collective::Scan) {
      std::exclusive_scan(scan_input.begin(), scan_input.end(), scan_offset.begin(), 0U);
      scan_total = scan_offset.back() + scan_input.back();
    }
  }
  throw std::runtime_error("planner did not finish");
}

uint32_t ceil_div(uint32_t value, uint32_t by) { return (value + by - 1) / by; }

struct Layout {
  std::vector<IdType> q, kv, tails, o, indices, requests, query_tiles, kv_parts, chunks, merge;
  std::vector<__nv_bfloat16> partial, output;
  uint32_t active = 0, count = 0;

  Layout(uint32_t rows, uint32_t work, uint32_t parts)
      : q(rows + 1, kUnused), kv(q), tails(rows, kUnused), o(q),
        indices(rows * kPagesPerRequest, kUnused), requests(work, 0),
        query_tiles(requests), kv_parts(requests), chunks(q), merge(q),
        partial(rows * parts * kRowWidth, kFill), output(rows * kRowWidth, kFill) {}
  bool operator==(const Layout&) const = default;
};

struct Fixture {
  const uint32_t rows, prompt_chunk, generated_chunk;
  std::vector<uint32_t> slots, positions, kinds, pages, offsets;
  std::vector<uint16_t> valid_tokens;
  uint32_t dynamic_active;
  bool dynamic, use_kinds;

  Fixture(uint32_t r, uint32_t live, uint32_t prompt, uint32_t generated, bool dynamic_count, bool row_kinds)
      : rows(r), prompt_chunk(prompt), generated_chunk(generated), slots(live), positions(live), kinds(live),
        pages(kSequences * kPagesPerRequest), offsets(kSequences + 1), valid_tokens(pages.size(), kPage),
        dynamic_active(live), dynamic(dynamic_count), use_kinds(row_kinds) {
    std::iota(pages.begin(), pages.end(), 0U);
    for (uint32_t i = 0; i <= kSequences; ++i) offsets[i] = i * kPagesPerRequest;
  }

  uint32_t active() const { return std::min(rows, dynamic_active); }
  uint32_t max_parts() const {
    uint32_t chunk = prompt_chunk && generated_chunk ? std::min(prompt_chunk, generated_chunk)
                                                    : std::max(prompt_chunk, generated_chunk);
    return chunk == 0 ? 1 : ceil_div(kPagesPerRequest * kPage, chunk);
  }
  uint32_t work_capacity() const { return (ceil_div(rows * kGroup, kCtaTileQ) + rows) * max_parts(); }
  Layout empty_layout() const { return Layout(rows, work_capacity(), max_parts()); }
};

Layout expected(const Fixture& f) {
  Layout out = f.empty_layout();
  out.active = f.active();
  const bool split = f.prompt_chunk || f.generated_chunk;
  uint32_t requests = 0, used_pages = 0, partial = 0;
  std::fill(out.output.begin() + out.active * kRowWidth, out.output.end(), 0);
  for (uint32_t begin = 0; begin < out.active;) {
    const uint32_t seq = f.slots[begin];
    uint32_t end = begin + 1;
    while (end < out.active && f.slots[end] == seq && f.positions[end] == f.positions[end - 1] + 1) ++end;
    const uint32_t context = f.positions[end - 1] + 1;
    const uint32_t chunk = f.use_kinds && f.kinds[end - 1] == 1 ? f.prompt_chunk : f.generated_chunk;
    const uint32_t pages = ceil_div(context, kPage);
    bool valid = seq < kSequences && pages <= kPagesPerRequest;
    uint32_t first_page = 0;
    if (valid) {
      first_page = f.offsets[seq];
      valid = f.offsets[seq + 1] - first_page >= pages;
      for (uint32_t page = 0; valid && page < pages; ++page) {
        uint32_t wanted = std::min(kPage, context - page * kPage);
        const uint16_t tokens = f.valid_tokens[first_page + page];
        valid = wanted <= tokens && tokens <= kPage && f.pages[first_page + page] < f.pages.size();
      }
    }
    const uint32_t parts = valid && chunk ? ceil_div(context, chunk) : 1;
    out.q[requests] = static_cast<IdType>(begin);
    out.kv[requests] = static_cast<IdType>(used_pages);
    out.o[requests] = static_cast<IdType>(split ? partial : begin);
    out.chunks[requests] = chunk ? static_cast<IdType>(chunk) : 1 << 30;
    out.tails[requests] = valid ? static_cast<IdType>(context - (pages - 1) * kPage) : 1;
    if (valid) {
      std::copy_n(f.pages.begin() + first_page, pages, out.indices.begin() + used_pages);
      used_pages += pages;
      for (uint32_t tile = 0; tile < ceil_div((end - begin) * kGroup, kCtaTileQ); ++tile) {
        for (uint32_t part = 0; part < parts; ++part) {
          out.requests[out.count] = static_cast<IdType>(requests);
          out.query_tiles[out.count] = static_cast<IdType>(tile);
          out.kv_parts[out.count++] = static_cast<IdType>(part);
        }
      }
    }
    for (uint32_t row = begin; row < end; ++row) {
      if (split) out.merge[row] = static_cast<IdType>(partial);
      if (!valid) {
        auto& values = split ? out.partial : out.output;
        std::fill_n(values.begin() + (split ? partial : row) * kRowWidth, kRowWidth, kNaN);
      }
      partial += split ? parts : 0;
    }
    ++requests;
    begin = end;
  }
  std::fill(out.q.begin() + requests, out.q.end(), static_cast<IdType>(out.active));
  std::fill(out.kv.begin() + requests, out.kv.end(), static_cast<IdType>(used_pages));
  std::fill(out.o.begin() + requests, out.o.end(), static_cast<IdType>(split ? partial : out.active));
  std::fill(out.tails.begin() + requests, out.tails.end(), 0);
  if (split) std::fill(out.merge.begin() + out.active, out.merge.end(), static_cast<IdType>(partial));
  return out;
}

void check(Fixture& f, std::mt19937& random) {
  const RowBatch batch{
      f.offsets.data(), f.pages.data(), f.valid_tokens.data(), f.slots.data(), f.positions.data(),
      kSequences, f.pages.size(), f.slots.size(), f.pages.size(),
      f.dynamic ? &f.dynamic_active : nullptr, f.use_kinds ? f.kinds.data() : nullptr,
  };
  Layout actual = f.empty_layout();
  auto mask = std::make_unique<bool[]>(f.work_capacity());
  std::fill_n(mask.get(), f.work_capacity(), true);
  std::vector<Task> workers;
  for (uint32_t thread = 0; thread < kPlannerThreads; ++thread) {
    auto add = [&]<uint32_t Rows>() {
      workers.push_back(plan_kernel<Rows>(
          batch, f.rows, kPagesPerRequest, kGroup, kRowWidth, f.work_capacity(), f.prompt_chunk,
          f.generated_chunk, actual.q.data(), actual.kv.data(), actual.tails.data(), actual.o.data(),
          actual.indices.data(), actual.requests.data(), actual.query_tiles.data(), actual.kv_parts.data(),
          actual.chunks.data(), &actual.active, &actual.count, mask.get(), actual.merge.data(),
          actual.partial.data(), actual.output.data()));
    };
    if (f.rows <= kPlannerThreads) add.operator()<1024>();
    else add.operator()<2048>();
  }
  execute(workers, random);
  if (actual != expected(f)) throw std::runtime_error("GPU planner metadata differs from serial plan");
  for (uint32_t i = 0; i < f.work_capacity(); ++i) assert(mask[i] == (i < actual.count));
}

int main() {
  std::mt19937 random(0x482622U);
  constexpr std::array live_counts{0U, 1U, 1023U, 1024U, 1025U, 2045U, 2048U, 2050U};
  constexpr std::pair<uint32_t, uint32_t> chunk_pairs[] = {{0, 0}, {512, 0}, {0, 256}, {512, 256}};
  uint32_t cases = 0;
  for (uint32_t rows : {1024U, 2048U}) {
    for (uint32_t live : live_counts) {
      for (auto [prompt, generated] : chunk_pairs) {
        for (bool interleaved : {false, true}) {
          for (bool invalid : {false, true}) {
            for (bool dynamic : {false, true}) {
              Fixture f(rows, live, prompt, generated, dynamic, !interleaved);
              for (uint32_t i = 0; i < live; ++i) {
                f.slots[i] = interleaved ? i % kSequences : i / 600;
                f.positions[i] = interleaved ? i / kSequences : i % 600;
                f.kinds[i] = (f.slots[i] % 2 == 0);
              }
              if (invalid) {
                f.pages[f.offsets[1]] = static_cast<uint32_t>(f.pages.size());
                f.valid_tokens[f.offsets[3] + 2] = 1;
              }
              if (dynamic && live > 0) --f.dynamic_active;
              check(f, random);
              ++cases;
            }
          }
        }
      }
    }
  }
  std::cout << "verified " << cases << " planner fixtures\n";
}
