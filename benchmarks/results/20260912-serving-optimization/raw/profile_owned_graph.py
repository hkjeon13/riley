#!/usr/bin/env python3
"""Instrument an isolated source copy and summarize native owned-graph diagnostics.

This never builds or runs CUDA. The checkout containing this script cannot be
patched. Instrumentation is runtime-enabled by RILEY_OWNED_GRAPH_PROFILE=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

SOURCE = Path("kernels/src/graph_resources.cu")
REPOSITORY_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / SOURCE).is_file()), None
)
MARKER = "// RILEY_ISOLATED_OWNED_GRAPH_DIAGNOSTICS_V1"
SCHEMA = "riley.owned-graph-diagnostic.v1"

# The two CUDA events bracket one graph launch. The original synchronization is
# untouched; no per-node event or synchronization is inserted into the graph.
HELPERS = r'''
// RILEY_ISOLATED_OWNED_GRAPH_DIAGNOSTICS_V1
#include <chrono>
#include <cstdio>
namespace {
using RileyDiagClock = std::chrono::steady_clock;
using RileyDiagTime = RileyDiagClock::time_point;
static uint64_t riley_diag_ns(RileyDiagTime begin, RileyDiagTime end) noexcept {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
}
static bool riley_diag_enabled() noexcept {
  const char* value=std::getenv("RILEY_OWNED_GRAPH_PROFILE");
  return value!=nullptr && std::strcmp(value,"1")==0;
}
static uint64_t riley_diag_inventory(cudaGraph_t graph) noexcept {
  if(!riley_diag_enabled()) return 0;
  static std::atomic<uint64_t> next{0};
  const uint64_t id=next.fetch_add(1)+1;
  size_t count=0,kernels=0,copies=0,others=0;
  uint64_t h2d=0,d2h=0,d2d=0;
  size_t unknown_copies=0;
  cudaError_t copy_status=cudaSuccess;
  auto status=cudaGraphGetNodes(graph,nullptr,&count);
  if(status==cudaSuccess){
    auto* nodes=static_cast<cudaGraphNode_t*>(std::malloc(count*sizeof(cudaGraphNode_t)));
    if(nodes==nullptr && count!=0) status=cudaErrorMemoryAllocation;
    if(status==cudaSuccess) status=cudaGraphGetNodes(graph,nodes,&count);
    if(status==cudaSuccess) for(size_t i=0;i<count;++i){
      cudaGraphNodeType type{};status=cudaGraphNodeGetType(nodes[i],&type);
      if(status!=cudaSuccess) break;
      if(type==cudaGraphNodeTypeKernel) ++kernels;
      else if(type==cudaGraphNodeTypeMemcpy){
        ++copies;
        cudaMemcpy3DParms parameters{};
        const auto queried=cudaGraphMemcpyNodeGetParams(nodes[i],&parameters);
        if(queried!=cudaSuccess){copy_status=queried;++unknown_copies;continue;}
        // For linear pointers CUDA extent.width is in bytes; array extents use
        // elements and need channel metadata. Never label those element counts
        // as bytes, and never guess the direction of cudaMemcpyDefault.
        if(parameters.srcArray!=nullptr || parameters.dstArray!=nullptr){++unknown_copies;continue;}
        uint64_t bytes=parameters.extent.width;
        if(parameters.extent.height!=0 && bytes>UINT64_MAX/parameters.extent.height){++unknown_copies;continue;}
        bytes*=parameters.extent.height;
        if(parameters.extent.depth!=0 && bytes>UINT64_MAX/parameters.extent.depth){++unknown_copies;continue;}
        bytes*=parameters.extent.depth;
        uint64_t* total=nullptr;
        if(parameters.kind==cudaMemcpyHostToDevice) total=&h2d;
        else if(parameters.kind==cudaMemcpyDeviceToHost) total=&d2h;
        else if(parameters.kind==cudaMemcpyDeviceToDevice) total=&d2d;
        if(total==nullptr || *total>UINT64_MAX-bytes){++unknown_copies;continue;}
        *total+=bytes;
      }
      else ++others;
    }
    std::free(nodes);
  }
  const bool bytes_complete=status==cudaSuccess && copy_status==cudaSuccess && unknown_copies==0;
  char h2d_text[32]="null",d2h_text[32]="null",d2d_text[32]="null";
  if(bytes_complete){
    std::snprintf(h2d_text,sizeof(h2d_text),"%llu",static_cast<unsigned long long>(h2d));
    std::snprintf(d2h_text,sizeof(d2h_text),"%llu",static_cast<unsigned long long>(d2h));
    std::snprintf(d2d_text,sizeof(d2d_text),"%llu",static_cast<unsigned long long>(d2d));
  }
  std::fprintf(stderr,"{\"schema\":\"riley.owned-graph-diagnostic.v1\",\"kind\":\"graph\",\"capture_id\":%llu,\"inventory_status\":%d,\"nodes\":%zu,\"kernel_nodes\":%zu,\"memcpy_nodes\":%zu,\"other_nodes\":%zu,\"memcpy_params_status\":%d,\"memcpy_unknown_nodes\":%zu,\"memcpy_bytes_complete\":%s,\"h2d_bytes\":%s,\"d2h_bytes\":%s,\"d2d_bytes\":%s}\n",
      static_cast<unsigned long long>(id),static_cast<int>(status),count,kernels,copies,others,static_cast<int>(copy_status),unknown_copies,bytes_complete?"true":"false",h2d_text,d2h_text,d2d_text);
  return id;
}
struct RileyOwnedGraphDiagnostic {
  bool enabled=false;
  cudaEvent_t begin=nullptr,end=nullptr;
  cudaError_t timing_status=cudaSuccess;
  cudaStream_t stream=nullptr;
  RileyDiagTime setup_begin{},stage_begin{},stage_end{},launch_begin{},launch_end{},wait_begin{};
  explicit RileyOwnedGraphDiagnostic(bool active,cudaStream_t s) noexcept : enabled(active&&riley_diag_enabled()),stream(s) {
    if(!enabled) return;
    setup_begin=RileyDiagClock::now();
    timing_status=cudaEventCreate(&begin);
    if(timing_status==cudaSuccess) timing_status=cudaEventCreate(&end);
    stage_begin=RileyDiagClock::now();
  }
  void before_launch() noexcept {
    if(!enabled) return;
    stage_end=RileyDiagClock::now();
    if(timing_status==cudaSuccess) timing_status=cudaEventRecord(begin,stream);
    launch_begin=RileyDiagClock::now();
  }
  void after_launch() noexcept {
    if(!enabled) return;
    launch_end=RileyDiagClock::now();
    if(timing_status==cudaSuccess) timing_status=cudaEventRecord(end,stream);
  }
  void before_wait() noexcept {if(enabled) wait_begin=RileyDiagClock::now();}
  void finish(uint64_t capture,uint64_t replay,uint32_t position,uint64_t bytes,cudaError_t launched,cudaError_t completed) noexcept {
    if(!enabled) return;
    const auto wait_end=RileyDiagClock::now();
    float milliseconds=0;
    if(timing_status==cudaSuccess && completed==cudaSuccess && launched==cudaSuccess)
      timing_status=cudaEventElapsedTime(&milliseconds,begin,end);
    char gpu[64]="null";
    if(timing_status==cudaSuccess && completed==cudaSuccess && launched==cudaSuccess && std::isfinite(milliseconds))
      std::snprintf(gpu,sizeof(gpu),"%.3f",static_cast<double>(milliseconds)*1000000.0);
    std::fprintf(stderr,"{\"schema\":\"riley.owned-graph-diagnostic.v1\",\"kind\":\"replay\",\"capture_id\":%llu,\"replay_id\":%llu,\"position\":%u,\"host_staging_bytes\":%llu,\"event_setup_ns\":%llu,\"host_staging_ns\":%llu,\"host_launch_ns\":%llu,\"host_wait_ns\":%llu,\"cuda_graph_span_ns\":%s,\"event_status\":%d,\"launch_status\":%d,\"completion_status\":%d}\n",
      static_cast<unsigned long long>(capture),static_cast<unsigned long long>(replay),position,
      static_cast<unsigned long long>(bytes),static_cast<unsigned long long>(riley_diag_ns(setup_begin,stage_begin)),
      static_cast<unsigned long long>(riley_diag_ns(stage_begin,stage_end)),static_cast<unsigned long long>(riley_diag_ns(launch_begin,launch_end)),
      static_cast<unsigned long long>(riley_diag_ns(wait_begin,wait_end)),gpu,static_cast<int>(timing_status),static_cast<int>(launched),static_cast<int>(completed));
    close_events(); // Before the enclosing CurrentContext restores its caller.
  }
  void close_events() noexcept {
    // cudaEventDestroy does not wait for event completion. Never add a second
    // synchronization, including on a failed graph launch or completion.
    if(begin!=nullptr) {(void)cudaEventDestroy(begin);begin=nullptr;}
    if(end!=nullptr) {(void)cudaEventDestroy(end);end=nullptr;}
  }
  ~RileyOwnedGraphDiagnostic() noexcept {close_events();}
};
} // namespace
'''


PROJECTION_HELPERS = r'''
// Optional, perturbing first-layer projection timing. Event objects are made
// before capture and remain owned until every graph/exec has been destroyed.
namespace {
struct RileyProjectionDiagnostic {
  cudaEvent_t events[2][2][2]{}; // capture slot, canonical/override, begin/end
  bool recorded[2][2][2]{};
  uint64_t captures[2]{};
  uint32_t slot=0,target_j=0;
  bool active=false;
  cudaError_t prepare(uint32_t slots) noexcept {
    if(!riley_diag_enabled()) return cudaSuccess;
    const char* value=std::getenv("RILEY_OWNED_GRAPH_PROJECTION");
    if(value==nullptr || std::strcmp(value,"off")==0)return cudaSuccess;
    if(std::strcmp(value,"q")==0) target_j=0;
    else if(std::strcmp(value,"gate")==0) target_j=4;
    else if(std::strcmp(value,"down")==0) target_j=6;
    else return cudaErrorInvalidValue;
    if(slots<1 || slots>2 || has_events()) return cudaErrorInvalidValue;
    for(uint32_t s=0;s<slots;++s)for(uint32_t variant=0;variant<2;++variant)for(uint32_t edge=0;edge<2;++edge){
      const auto status=cudaEventCreate(&events[s][variant][edge]);
      if(status!=cudaSuccess) return status;
    }
    active=true;return cudaSuccess;
  }
  bool has_events() const noexcept {
    for(const auto& capture:events)for(const auto& variant:capture)for(auto event:variant)if(event!=nullptr)return true;
    return false;
  }
  void begin_capture(uint32_t selected_slot) noexcept {slot=selected_slot;}
  void bind_capture(uint64_t id) noexcept {if(active && slot<2)captures[slot]=id;}
  cudaError_t record(uint64_t layer,size_t projection,uint32_t variant,uint32_t edge,cudaStream_t stream) noexcept {
    if(!active || layer!=0 || projection!=target_j)return cudaSuccess;
    if(slot>=2 || variant>=2 || edge>=2)return cudaErrorInvalidValue;
    // The External flag retains actual event-record graph nodes. Ordinary
    // capture-only event dependencies cannot supply elapsed event timestamps.
    const auto status=cudaEventRecordWithFlags(events[slot][variant][edge],stream,cudaEventRecordExternal);
    if(status==cudaSuccess)recorded[slot][variant][edge]=true;
    return status;
  }
  void finish(uint64_t capture,uint64_t replay,uint32_t position,cudaError_t launched,cudaError_t completed) noexcept {
    if(!active)return;
    uint32_t selected=2;
    for(uint32_t s=0;s<2;++s)if(captures[s]==capture && capture!=0)selected=s;
    if(selected==2)return;
    for(uint32_t variant=0;variant<2;++variant){
      if(!recorded[selected][variant][0] && !recorded[selected][variant][1])continue;
      float milliseconds=0;char duration[64]="null";
      auto status=launched!=cudaSuccess?launched:completed;
      if(status==cudaSuccess && (!recorded[selected][variant][0] || !recorded[selected][variant][1]))status=cudaErrorInvalidValue;
      if(status==cudaSuccess)status=cudaEventElapsedTime(&milliseconds,events[selected][variant][0],events[selected][variant][1]);
      if(status==cudaSuccess && std::isfinite(milliseconds))std::snprintf(duration,sizeof(duration),"%.3f",static_cast<double>(milliseconds)*1000000.0);
      std::fprintf(stderr,"{\"schema\":\"riley.owned-graph-diagnostic.v1\",\"kind\":\"projection\",\"capture_id\":%llu,\"replay_id\":%llu,\"position\":%u,\"layer\":0,\"projection\":\"%s\",\"operator\":\"%s\",\"cuda_span_ns\":%s,\"event_status\":%d,\"launch_status\":%d,\"completion_status\":%d}\n",
        static_cast<unsigned long long>(capture),static_cast<unsigned long long>(replay),position,target_j==0?"q":target_j==4?"gate":"down",variant==0?"canonical":"override",duration,static_cast<int>(status),static_cast<int>(launched),static_cast<int>(completed));
    }
  }
  cudaError_t close() noexcept {
    for(auto& capture:events)for(auto& variant:capture)for(auto& event:variant)if(event!=nullptr){
      const auto status=cudaEventDestroy(event);event=nullptr;
      if(status!=cudaSuccess)return status; // Caller retains owner on ambiguity.
    }
    active=false;return cudaSuccess;
  }
};
} // namespace
'''


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"source anchor must occur exactly once (found {count}): {old[:90]!r}")
    return text.replace(old, new, 1)


def instrument_text(text: str, projection_events: bool = False) -> str:
    if MARKER in text:
        raise ValueError("source is already instrumented; use a fresh isolated source copy")
    original_syncs = text.count("cudaStreamSynchronize(")
    legacy_launch = "  auto launched = cudaGraphLaunch(r->exec, r->stream->stream);"
    selected_launch = "  auto launched = cudaGraphLaunch(selected_exec, r->stream->stream);"
    if (text.count(legacy_launch), text.count(selected_launch)) == (1, 0):
        dual = False
        if "prefill_exec" in text or "selected_exec" in text:
            raise ValueError("legacy launch source contains unexpected dual-graph state")
    elif (text.count(legacy_launch), text.count(selected_launch)) == (0, 1):
        dual = True
    else:
        raise ValueError("source must contain exactly one recognized legacy or selected graph launch")
    text = replace_once(text, "#include <cmath>\n", "#include <cmath>\n" + HELPERS + (PROJECTION_HELPERS if projection_events else ""))
    text = replace_once(text, "  cudaGraphExec_t exec = nullptr;", "  cudaGraphExec_t exec = nullptr;\n  uint64_t diagnostic_capture_id=0,diagnostic_prefill_capture_id=0,diagnostic_selected_capture_id=0,diagnostic_replay_id=0;")
    if dual:
        text = replace_once(text,
            "      r->prefill_graph=r->graph;r->prefill_exec=r->exec;",
            "      r->prefill_graph=r->graph;r->prefill_exec=r->exec;\n"
            "      r->diagnostic_prefill_capture_id=r->diagnostic_capture_id;")
        # Both the selected handle initialization and its validated-position
        # update must retain the supported source shape. Do not infer selection
        # from position again: the launched handle is the profiling authority.
        for anchor in ("  auto selected_exec = r->exec;", "      selected_exec=pos<128?r->prefill_exec:r->exec;"):
            if text.count(anchor) != 1:
                raise ValueError(f"dual graph selection anchor must occur exactly once: {anchor!r}")
    text = replace_once(text,
        "  r->terminal = true;\n  std::memmove(r->input->host_data, source, static_cast<size_t>(bytes));",
        "  RileyOwnedGraphDiagnostic diagnostic(r->decode_capacity!=0,r->stream->stream);\n"
        "  r->terminal = true;\n  std::memmove(r->input->host_data, source, static_cast<size_t>(bytes));")
    launch = selected_launch if dual else legacy_launch
    capture_id = "selected_exec==r->prefill_exec?r->diagnostic_prefill_capture_id:r->diagnostic_capture_id" if dual else "r->diagnostic_capture_id"
    text = replace_once(text, launch,
        f"  r->diagnostic_selected_capture_id={capture_id};\n"
        f"  diagnostic.before_launch();\n{launch}\n  diagnostic.after_launch();")
    text = replace_once(text,
        "  auto completed = cudaStreamSynchronize(r->stream->stream);",
        "  diagnostic.before_wait();\n  auto completed = cudaStreamSynchronize(r->stream->stream);")
    text = replace_once(text,
        "  if (completed == cudaSuccess) r->completion_unknown = false;",
        "  uint32_t diagnostic_position=0;\n  if(bytes>=8) std::memcpy(&diagnostic_position,source+4,4);\n"
        "  diagnostic.finish(r->diagnostic_selected_capture_id,++r->diagnostic_replay_id,diagnostic_position,bytes,launched,completed);\n"
        "  if (completed == cudaSuccess) r->completion_unknown = false;")
    text = replace_once(text,
        "    r->output_byte_offset = transfer_bytes; r->terminal = false;",
        "    r->output_byte_offset = transfer_bytes; r->terminal = false;\n"
        "    r->diagnostic_capture_id=riley_diag_inventory(r->graph);")
    text = replace_once(text,
        "  std::memmove(destination, static_cast<uint8_t*>(r->output->host_data) + r->output_byte_offset, static_cast<size_t>(bytes));",
        "  const auto diagnostic_read_begin=RileyDiagClock::now();\n"
        "  std::memmove(destination, static_cast<uint8_t*>(r->output->host_data) + r->output_byte_offset, static_cast<size_t>(bytes));\n"
        "  const auto diagnostic_read_end=RileyDiagClock::now();\n"
        '  if(r->decode_capacity!=0 && riley_diag_enabled()) std::fprintf(stderr,"{\\"schema\\":\\"riley.owned-graph-diagnostic.v1\\",\\"kind\\":\\"read\\",\\"capture_id\\":%llu,\\"replay_id\\":%llu,\\"host_read_bytes\\":%llu,\\"host_read_ns\\":%llu}\\n",\n'
        "      static_cast<unsigned long long>(r->diagnostic_selected_capture_id),static_cast<unsigned long long>(r->diagnostic_replay_id),static_cast<unsigned long long>(bytes),static_cast<unsigned long long>(riley_diag_ns(diagnostic_read_begin,diagnostic_read_end)));")
    if projection_events:
        text = instrument_projection_events(text, dual)
    if text.count("cudaStreamSynchronize(") != original_syncs:
        raise ValueError("instrumentation changed the synchronization count")
    return text


def instrument_projection_events(text: str, dual: bool) -> str:
    text = replace_once(text, "  uint64_t diagnostic_capture_id=0,diagnostic_prefill_capture_id=0,diagnostic_selected_capture_id=0,diagnostic_replay_id=0;",
        "  uint64_t diagnostic_capture_id=0,diagnostic_prefill_capture_id=0,diagnostic_selected_capture_id=0,diagnostic_replay_id=0;\n  RileyProjectionDiagnostic diagnostic_projection;")
    close_anchor = ("  if ((*resources)->graph != nullptr || (*resources)->exec != nullptr ||\n      (*resources)->prefill_graph != nullptr || (*resources)->prefill_exec != nullptr) {" if dual else
                    "  if ((*resources)->graph != nullptr || (*resources)->exec != nullptr) {")
    text = replace_once(text, close_anchor, close_anchor[:-3] + " || (*resources)->diagnostic_projection.has_events()) {")
    text = replace_once(text,
        "    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);",
        "    if(status==RILEY_CUDA_STATUS_SUCCESS){\n"
        "      status=runtime_error((*resources)->diagnostic_projection.close(),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);\n"
        "      if(status!=RILEY_CUDA_STATUS_SUCCESS)(*resources)->completion_unknown=true;\n"
        "    }\n    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);")
    # This unique full-model preparation seam runs before either capture.
    prepare_anchor = "  std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,static_cast<size_t>(transfer));"
    slots = 2 if dual else 1
    text = replace_once(text, prepare_anchor,
        "  if(profile==2 && riley_diag_enabled()){\n"
        "    CurrentContext diagnostic_scope(r->owner);\n"
        "    status=diagnostic_scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare projection events\");\n"
        f"    if(status==RILEY_CUDA_STATUS_SUCCESS)status=runtime_error(r->diagnostic_projection.prepare({slots}),error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare projection events\");\n"
        "    status=diagnostic_scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare projection events\");\n"
        "    if(status!=RILEY_CUDA_STATUS_SUCCESS){release_states();return status;}\n"
        "  }\n" + prepare_anchor)
    if dual:
        text = replace_once(text, "  auto record_stage = [&](bool prefill) noexcept {",
            "  auto record_stage = [&](bool prefill) noexcept {\n    r->diagnostic_projection.begin_capture(prefill?0:1);")
    text = replace_once(text, "    r->diagnostic_capture_id=riley_diag_inventory(r->graph);",
        "    r->diagnostic_capture_id=riley_diag_inventory(r->graph);\n    r->diagnostic_projection.bind_capture(r->diagnostic_capture_id);")
    text = replace_once(text, "      auto gemm=[&](size_t j,size_t out){",
        "      auto gemm=[&](size_t j,size_t out){\n"
        "        auto timed=[&](uint32_t variant,auto launch) noexcept {\n"
        "          auto result=kernel(r->diagnostic_projection.record(l,j,variant,0,r->stream->stream));\n"
        "          if(result==RILEY_CUDA_STATUS_SUCCESS)result=launch();\n"
        "          if(result==RILEY_CUDA_STATUS_SUCCESS)result=kernel(r->diagnostic_projection.record(l,j,variant,1,r->stream->stream));\n"
        "          return result;\n        };\n")
    canonical = 'enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[out],states[l*7+j],error,"decode GEMM")'
    override = 'kernel(enqueue_compiled_prefill_gemm(r->stream->stream,d[inputs[j]]->device_data,weights[weight_ids[j]]->device_data,d[out]->device_data,ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4))'
    text = replace_once(text, canonical, f"timed(0,[&]() noexcept {{return {canonical};}})")
    text = replace_once(text, override, f"timed(1,[&]() noexcept {{return {override};}})")
    text = replace_once(text,
        "  diagnostic.finish(r->diagnostic_selected_capture_id,++r->diagnostic_replay_id,diagnostic_position,bytes,launched,completed);",
        "  diagnostic.finish(r->diagnostic_selected_capture_id,++r->diagnostic_replay_id,diagnostic_position,bytes,launched,completed);\n"
        "  r->diagnostic_projection.finish(r->diagnostic_selected_capture_id,r->diagnostic_replay_id,diagnostic_position,launched,completed);")
    return text


def instrument(source_root: Path, apply: bool, projection_events: bool = False) -> dict[str, object]:
    root = source_root.resolve(strict=True)
    target = (root / SOURCE).resolve(strict=True)
    if REPOSITORY_ROOT is not None and (root == REPOSITORY_ROOT or target == (REPOSITORY_ROOT / SOURCE).resolve()):
        raise ValueError("refusing to instrument the tooling checkout; choose an isolated source copy")
    if not target.is_relative_to(root):
        raise ValueError("source target resolves outside the selected source root")
    original = target.read_bytes()
    updated = instrument_text(original.decode("utf-8"), projection_events).encode("utf-8")
    report = {
        "schema_version": "riley.owned-graph-instrumentation.v1",
        "source_root": str(root), "source_file": str(SOURCE),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "instrumented_sha256": hashlib.sha256(updated).hexdigest(),
        "source_topology": "dual_graph_v1" if b"cudaGraphLaunch(selected_exec," in original else "single_graph_v1",
        "applied": apply,
        "runtime_enable": "RILEY_OWNED_GRAPH_PROFILE=1",
        "projection_events": projection_events,
        "projection_selection": "RILEY_OWNED_GRAPH_PROJECTION=q|gate|down|off (default off)" if projection_events else None,
        "performance_claim_eligible": False,
    }
    if apply:
        target.write_bytes(updated)
    return report


def summarize(log: Path, prefill_tokens: int) -> dict[str, object]:
    records = []
    for number, line in enumerate(log.read_text().splitlines(), 1):
        if SCHEMA not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed diagnostic at line {number}") from exc
        if record.get("schema") != SCHEMA or record.get("kind") not in {"graph", "replay", "read", "projection"}:
            raise ValueError(f"unexpected diagnostic at line {number}")
        records.append(record)
    replays = [r for r in records if r["kind"] == "replay"]
    if not replays:
        raise ValueError("no owned-graph replay diagnostics found")
    metrics = ["event_setup_ns", "host_staging_ns", "host_launch_ns", "host_wait_ns", "cuda_graph_span_ns"]
    groups = {}
    for phase, rows in (("prefill", [r for r in replays if r["position"] < prefill_tokens]),
                        ("decode", [r for r in replays if r["position"] >= prefill_tokens])):
        successful = [r for r in rows if r["launch_status"] == 0 and r["completion_status"] == 0]
        groups[phase] = {"replays": len(rows), "successful_replays": len(successful),
                         "capture_ids": sorted({r["capture_id"] for r in rows}), "metrics": {}}
        for metric in metrics:
            values = [r[metric] for r in successful if r[metric] is not None]
            groups[phase]["metrics"][metric] = {
                "measured_count": len(values), "unmeasured_count": len(successful)-len(values),
                "median": statistics.median(values) if values else None,
                "sum": sum(values) if values else None,
            }
    projections = []
    projection_records = [r for r in records if r["kind"] == "projection"]
    for phase in ("prefill", "decode"):
        for projection in ("q", "gate", "down"):
            for operator in ("canonical", "override"):
                rows = [r for r in projection_records if ("prefill" if r["position"] < prefill_tokens else "decode") == phase and r["projection"] == projection and r["operator"] == operator]
                if not rows:
                    continue
                values = [r["cuda_span_ns"] for r in rows if r["cuda_span_ns"] is not None and r["event_status"] == 0 and r["launch_status"] == 0 and r["completion_status"] == 0]
                projections.append({"phase": phase, "layer": 0, "projection": projection,
                    "operator": operator, "records": len(rows), "measured_count": len(values),
                    "unmeasured_or_failed_count": len(rows)-len(values),
                    "capture_ids": sorted({r["capture_id"] for r in rows}),
                    "median_cuda_span_ns": statistics.median(values) if values else None})
    return {
        "schema_version": "riley.owned-graph-diagnostic-summary.v1",
        "performance_claim_eligible": False,
        "classification": {"source": "position", "prefill_tokens": prefill_tokens},
        "graph_inventories": [r for r in records if r["kind"] == "graph"],
        "groups": groups,
        "failed_replays": sum(r["launch_status"] != 0 or r["completion_status"] != 0 for r in replays),
        "read_records": sum(r["kind"] == "read" for r in records),
        "projection_groups": projections,
        "unmeasured": ["per_kernel_cuda_time", "Rust_metadata_pack", "request_latency_without_instrumentation"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    patch = sub.add_parser("instrument")
    patch.add_argument("--source-root", type=Path, required=True)
    patch.add_argument("--apply", action="store_true", help="write the validated patch in an isolated source copy")
    patch.add_argument("--projection-events", action="store_true", help="also insert perturbing first-layer projection event nodes")
    summary = sub.add_parser("summarize")
    summary.add_argument("--log", type=Path, required=True)
    summary.add_argument("--prefill-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize" and args.prefill_tokens < 1:
            raise ValueError("prefill token count must be positive")
        result = instrument(args.source_root, args.apply, args.projection_events) if args.command == "instrument" else summarize(args.log, args.prefill_tokens)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
