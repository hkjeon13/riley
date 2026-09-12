#!/usr/bin/env python3
"""Instrument a frozen batch8 copy with one selected decode operator family.

No build or GPU action occurs here. Preview is the default; --apply only accepts
an isolated copy of the pinned source. Runtime examples (one mode per process):
  RILEY_OWNED_GRAPH_PROFILE=1 RILEY_DECODE_OPERATOR=rope_attention
  RILEY_OWNED_GRAPH_PROFILE=1 RILEY_DECODE_OPERATOR=qkv RILEY_DECODE_OPERATOR_LAYERS=all
This adapter measures decode only. Historical prefill projection events are
rejected because they do not wrap the active packed M16 prefill path.
Whole-graph timing still adds host events when profiling is enabled. Operator
and projection defaults are off, which adds no captured timing nodes. Initial
norm, embedding, argmax and transfers are outside the operator coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

_HISTORICAL = Path(__file__).with_name("profile_owned_graph.py")
_SPEC = importlib.util.spec_from_file_location("riley_historical_owned_graph", _HISTORICAL)
assert _SPEC is not None and _SPEC.loader is not None
whole = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(whole)
SOURCE = whole.SOURCE
REPOSITORY_ROOT = whole.REPOSITORY_ROOT
SOURCE_SHA256 = "bb978cc6557be27b844d356ba38cf462b37934dc283e8a89f2df8a3f3a0b132f"
HISTORICAL_SHA256 = "9f0f3a526ed1a47c3e6a37e33111a21356ac5b2d58294ff797ec0a40a5091b78"
SCHEMA = "riley.decode-operator-diagnostic.v1"
MARKER = "// RILEY_ISOLATED_DECODE_OPERATOR_DIAGNOSTICS_V1"
FAMILIES = ("qkv", "rope_attention", "o", "post_norm", "gate_up", "swiglu", "down", "next_norm", "head")

# No implicit CUDA destructor: captured events belong to the aggregate resource
# owner and are destroyed under its CurrentContext after all graph handles.
HELPERS = r'''
// RILEY_ISOLATED_DECODE_OPERATOR_DIAGNOSTICS_V1
namespace {
enum class RileyDecodeOperator { off,qkv,rope_attention,o,post_norm,gate_up,swiglu,down,next_norm,head };
struct RileyDecodeOperatorDiagnostic {
  cudaEvent_t events[31][2]{};
  bool recorded[31][2]{};
  uint64_t captures[2]{};
  uint32_t slot=0,mask=0;
  bool eligible=false,active=false;
  RileyDecodeOperator selected=RileyDecodeOperator::off;
  const char* name="off";
  bool selected_layer(uint32_t layer) const noexcept {return layer<31 && (mask&(uint32_t(1)<<layer))!=0;}
  bool has_events() const noexcept {
    for(const auto& pair:events)for(auto event:pair)if(event!=nullptr)return true;
    return false;
  }
  cudaError_t prepare() noexcept {
    if(!riley_diag_enabled())return cudaSuccess;
    eligible=true;
    const char* value=std::getenv("RILEY_DECODE_OPERATOR");
    if(value==nullptr || std::strcmp(value,"off")==0)return cudaSuccess;
    const char* names[]={"off","qkv","rope_attention","o","post_norm","gate_up","swiglu","down","next_norm","head"};
    for(uint32_t i=1;i<10;++i)if(std::strcmp(value,names[i])==0){selected=static_cast<RileyDecodeOperator>(i);name=names[i];break;}
    if(selected==RileyDecodeOperator::off || has_events())return cudaErrorInvalidValue;
    const char* projection=std::getenv("RILEY_OWNED_GRAPH_PROJECTION");
    if(projection!=nullptr && std::strcmp(projection,"off")!=0)return cudaErrorInvalidValue;
    const char* layers=std::getenv("RILEY_DECODE_OPERATOR_LAYERS");
    if(layers==nullptr || std::strcmp(layers,"0,15,29")==0)mask=(1u<<0)|(1u<<15)|(1u<<29);
    else if(std::strcmp(layers,"all")==0)mask=(1u<<30)-1;
    else return cudaErrorInvalidValue;
    if(selected==RileyDecodeOperator::head)mask=1u<<30; // Head is not layer 30.
    for(uint32_t layer=0;layer<31;++layer)if(selected_layer(layer))for(uint32_t edge=0;edge<2;++edge){
      const auto status=cudaEventCreate(&events[layer][edge]);
      if(status!=cudaSuccess)return status; // Partial allocation stays with owner.
    }
    active=true;return cudaSuccess;
  }
  void begin_capture(uint32_t value) noexcept {slot=value;}
  void bind_capture(uint64_t id) noexcept {
    if(!eligible || slot>=2 || id==0)return;
    captures[slot]=id;
    uint32_t count=0,edges=0;
    for(uint32_t layer=0;layer<31;++layer)if(selected_layer(layer)){
      ++count;for(bool edge:recorded[layer])if(edge)++edges;
    }
    std::fprintf(stderr,"{\"schema\":\"riley.decode-operator-diagnostic.v1\",\"kind\":\"capture\",\"capture_id\":%llu,\"phase\":\"%s\",\"operator\":\"%s\",\"layer_mask\":%u,\"interval_count\":%u,\"expected_event_records\":%u,\"recorded_event_records\":%u,\"source_sha256\":\"@SOURCE_SHA256@\",\"tool_sha256\":\"@TOOL_SHA256@\",\"historical_tool_sha256\":\"@HISTORICAL_SHA256@\",\"projection_events_compiled\":@PROJECTION_EVENTS@}\n",
      static_cast<unsigned long long>(id),slot==0?"prefill":"decode",name,mask,count,slot==1?2*count:0,slot==1?edges:0);
  }
  cudaError_t record(bool packed,RileyDecodeOperator family,uint64_t layer,uint32_t edge,cudaStream_t stream) noexcept {
    if(!active || !packed || family!=selected || layer>=31 || !selected_layer(static_cast<uint32_t>(layer)))return cudaSuccess;
    if(slot!=1 || edge>=2 || events[layer][edge]==nullptr)return cudaErrorInvalidValue;
    const auto status=cudaEventRecordWithFlags(events[layer][edge],stream,cudaEventRecordExternal);
    if(status==cudaSuccess)recorded[layer][edge]=true;
    return status;
  }
  void finish(uint64_t capture,uint64_t replay,uint32_t position,cudaError_t launched,cudaError_t completed) noexcept {
    if(!active || capture==0 || capture!=captures[1])return;
    char line[8192]{};size_t used=0;
    auto append=[&](const char* format,auto... args) noexcept {
      if(used>=sizeof(line))return;
      const int wrote=std::snprintf(line+used,sizeof(line)-used,format,args...);
      if(wrote<0 || static_cast<size_t>(wrote)>=sizeof(line)-used){used=sizeof(line);return;}
      used+=static_cast<size_t>(wrote);
    };
    append("{\"schema\":\"riley.decode-operator-diagnostic.v1\",\"kind\":\"operation\",\"capture_id\":%llu,\"replay_id\":%llu,\"position\":%u,\"operator\":\"%s\",\"launch_status\":%d,\"completion_status\":%d,\"intervals\":[",
      static_cast<unsigned long long>(capture),static_cast<unsigned long long>(replay),position,name,static_cast<int>(launched),static_cast<int>(completed));
    bool first=true,complete=true;double sum=0;
    for(uint32_t layer=0;layer<31;++layer)if(selected_layer(layer)){
      auto status=launched!=cudaSuccess?launched:completed;
      if(status==cudaSuccess && (position<128 || position>=160 || !recorded[layer][0] || !recorded[layer][1]))status=cudaErrorInvalidValue;
      float milliseconds=0;
      if(status==cudaSuccess)status=cudaEventElapsedTime(&milliseconds,events[layer][0],events[layer][1]);
      if(status==cudaSuccess && (!std::isfinite(milliseconds) || milliseconds<0))status=cudaErrorInvalidValue;
      char duration[64]="null",layer_text[16]="null";
      if(layer<30)std::snprintf(layer_text,sizeof(layer_text),"%u",layer);
      if(status==cudaSuccess){const double ns=static_cast<double>(milliseconds)*1000000.0;sum+=ns;std::snprintf(duration,sizeof(duration),"%.3f",ns);}
      else complete=false;
      append("%s{\"layer\":%s,\"cuda_span_ns\":%s,\"event_status\":%d}",first?"":",",layer_text,duration,static_cast<int>(status));first=false;
    }
    char total[64]="null";if(complete)std::snprintf(total,sizeof(total),"%.3f",sum);
    append("],\"cuda_sum_ns\":%s}\n",total);
    if(used<sizeof(line))std::fputs(line,stderr);
    else std::fputs("{\"schema\":\"riley.decode-operator-diagnostic.v1\",\"kind\":\"serialization_error\"}\n",stderr);
  }
  cudaError_t close() noexcept {
    for(auto& pair:events)for(auto& event:pair)if(event!=nullptr){
      const auto status=cudaEventDestroy(event);event=nullptr;
      if(status!=cudaSuccess)return status; // Caller retains owner on ambiguity.
    }
    active=false;return cudaSuccess;
  }
};
} // namespace
'''

# Each expression occurs once in the pinned graph4 recorder. Wrapping the
# enqueue expression retains its existing condition, input, output and order.
OPERATOR_EXPRESSIONS = {
    "qkv": 'enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,packed_buffers[60],packed_states[2*l],error,"packed decode QKV GEMM")',
    "rope_attention": """kernel(enqueue_compiled_packed_decode_rope_attention(r->stream->stream,qkv,qkv+1152,qkv+1536,d[4]->device_data,
            static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,
            static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,
            d[5]->device_data,d[17]->device_data,d[18]->device_data,d[19]->device_data))""",
    "o": "gemm(3,3)",
    "post_norm": 'kernel(enqueue_compiled_norm_rows(r->stream->stream,buffer(3),buffer(1),weights[5]->device_data,buffer(11),buffer(2),1,rows))',
    "gate_up": 'enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,packed_buffers[61],packed_states[2*l+1],error,"packed decode gate/up GEMM")',
    "swiglu": 'kernel(enqueue_compiled_swiglu_rows(r->stream->stream,gate,up,buffer(12),rows))',
    "down": "gemm(6,5)",
    "next_norm": 'kernel(enqueue_compiled_norm_rows(r->stream->stream,buffer(5),buffer(11),(l+1<layers?w[3+9*(l+1)]:w[1])->device_data,buffer(1),buffer(2),2,rows))',
    "head": 'enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[13],states[layers*7],error,"decode head")',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def helper_text(projection_events: bool = False) -> str:
    return (HELPERS.replace("@SOURCE_SHA256@", SOURCE_SHA256)
            .replace("@TOOL_SHA256@", sha256(Path(__file__).read_bytes()))
            .replace("@HISTORICAL_SHA256@", HISTORICAL_SHA256)
            .replace("@PROJECTION_EVENTS@", "true" if projection_events else "false"))


def instrument_text(original: str, projection_events: bool = False) -> str:
    if projection_events:
        raise ValueError("batch8 decode profiler does not support historical prefill projection events")
    if MARKER in original or whole.MARKER in original:
        raise ValueError("source is already instrumented; use a fresh isolated source copy")
    if sha256(original.encode()) != SOURCE_SHA256:
        raise ValueError("unknown source SHA256; only the frozen batch8 graph recorder is supported")
    if sha256(_HISTORICAL.read_bytes()) != HISTORICAL_SHA256:
        raise ValueError("historical whole-graph profiler SHA256 changed")
    patch = whole.replace_once
    text = whole.instrument_text(original, projection_events=projection_events)
    text = patch(text, whole.HELPERS, whole.HELPERS + helper_text(projection_events))
    text = patch(text, "  size_t count=0,kernels=0,copies=0,others=0;", "  size_t count=0,kernels=0,copies=0,others=0,event_records=0,event_waits=0;")
    text = patch(text, "      else ++others;", "      else {++others;if(type==cudaGraphNodeTypeEventRecord)++event_records;else if(type==cudaGraphNodeTypeWaitEvent)++event_waits;}")
    inventory_end = "  return id;\n}\nstruct RileyOwnedGraphDiagnostic"
    text = patch(text, inventory_end,
        '  std::fprintf(stderr,"{\\"schema\\":\\"riley.decode-operator-diagnostic.v1\\",\\"kind\\":\\"inventory\\",\\"capture_id\\":%llu,\\"inventory_status\\":%d,\\"event_record_nodes\\":%zu,\\"event_wait_nodes\\":%zu}\\n",static_cast<unsigned long long>(id),static_cast<int>(status),event_records,event_waits);\n' + inventory_end)
    member = "  uint64_t diagnostic_capture_id=0,diagnostic_prefill_capture_id=0,diagnostic_selected_capture_id=0,diagnostic_replay_id=0;"
    text = patch(text, member, member + "\n  RileyDecodeOperatorDiagnostic diagnostic_operator;")
    close_guard = "      (*resources)->prefill_graph != nullptr || (*resources)->prefill_exec != nullptr"
    text = patch(text, close_guard, close_guard + " || (*resources)->diagnostic_operator.has_events()")
    leave = "    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);"
    text = patch(text, leave,
        "    if(status==RILEY_CUDA_STATUS_SUCCESS){\n"
        "      status=runtime_error((*resources)->diagnostic_operator.close(),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);\n"
        "      if(status!=RILEY_CUDA_STATUS_SUCCESS)(*resources)->completion_unknown=true;\n"
        "    }\n" + leave)
    prepare = "  std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,static_cast<size_t>(transfer));"
    text = patch(text, prepare,
        "  if(profile==2 && packed_buffers!=nullptr && riley_diag_enabled()){\n"
        "    CurrentContext diagnostic_scope(r->owner);\n"
        "    status=diagnostic_scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare decode operator events\");\n"
        "    if(status==RILEY_CUDA_STATUS_SUCCESS)status=runtime_error(r->diagnostic_operator.prepare(),error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare decode operator events\");\n"
        "    status=diagnostic_scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,\"prepare decode operator events\");\n"
        "    if(status!=RILEY_CUDA_STATUS_SUCCESS){release_states();return status;}\n"
        "  }\n" + prepare)
    capture = "  auto record_stage = [&](bool prefill) noexcept {"
    text = patch(text, capture, capture + "\n    r->diagnostic_operator.begin_capture(prefill?0:1);")
    bind = "    r->diagnostic_capture_id=riley_diag_inventory(r->graph);"
    text = patch(text, bind, bind + "\n    r->diagnostic_operator.bind_capture(r->diagnostic_capture_id);")
    kernel = '    auto kernel=[&](cudaError_t e){return runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"decode kernel");};'
    text = patch(text, kernel, kernel + "\n"
        "    auto time_decode_operator=[&](RileyDecodeOperator family,uint64_t layer,auto enqueue) noexcept {\n"
        "      auto result=kernel(r->diagnostic_operator.record(packed_decode,family,layer,0,r->stream->stream));\n"
        "      if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue();\n"
        "      if(result==RILEY_CUDA_STATUS_SUCCESS)result=kernel(r->diagnostic_operator.record(packed_decode,family,layer,1,r->stream->stream));\n"
        "      return result;\n    };")
    for family, expression in OPERATOR_EXPRESSIONS.items():
        layer = "30" if family == "head" else "l"
        text = patch(text, expression, f"time_decode_operator(RileyDecodeOperator::{family},{layer},[&]() noexcept {{return {expression};}})")
    finish = "  diagnostic.finish(r->diagnostic_selected_capture_id,++r->diagnostic_replay_id,diagnostic_position,bytes,launched,completed);"
    text = patch(text, finish, finish + "\n  r->diagnostic_operator.finish(r->diagnostic_selected_capture_id,r->diagnostic_replay_id,diagnostic_position,launched,completed);")
    if text.count("cudaStreamSynchronize(") != original.count("cudaStreamSynchronize(") or "cudaEventSynchronize(" in text:
        raise ValueError("instrumentation changed the synchronization contract")
    return text


def instrument(source_root: Path, apply: bool = False, projection_events: bool = False) -> dict:
    root = source_root.resolve(strict=True)
    unresolved = root / SOURCE
    target = unresolved.resolve(strict=True)
    if REPOSITORY_ROOT is not None and (root.is_relative_to(REPOSITORY_ROOT) or target.is_relative_to(REPOSITORY_ROOT) or target.samefile(REPOSITORY_ROOT / SOURCE)):
        raise ValueError("refusing to instrument the tooling checkout; choose an isolated source copy")
    if not target.is_relative_to(root) or unresolved.is_symlink() or target.stat().st_nlink != 1:
        raise ValueError("source must be a private regular file inside the isolated root")
    original = target.read_bytes()
    updated = instrument_text(original.decode(), projection_events).encode()
    receipt = {"schema_version": "riley.decode-operator-instrumentation.v1", "source_root": str(root),
        "source_file": str(SOURCE), "original_sha256": sha256(original), "instrumented_sha256": sha256(updated),
        "tool_sha256": sha256(Path(__file__).read_bytes()), "historical_tool_sha256": HISTORICAL_SHA256,
        "applied": apply, "projection_events": projection_events, "families": list(FAMILIES),
        "runtime_enable": "RILEY_OWNED_GRAPH_PROFILE=1", "runtime_operator": "RILEY_DECODE_OPERATOR=<family>|off (default off)",
        "runtime_layers": "RILEY_DECODE_OPERATOR_LAYERS=0,15,29|all (default 0,15,29; head is one interval)",
        "runtime_exclusivity": "operator and historical projection modes require separate processes",
        "synchronizations_added": 0, "performance_claim_eligible": False}
    if apply:
        # Exclusive temporary creation and atomic replacement: no partial source
        # on a write error, no shared inode mutation, and no implicit overwrite of
        # an already instrumented source. Recheck before the sole replacement.
        descriptor, temporary = tempfile.mkstemp(prefix=".decode-operators-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(updated)
            os.chmod(temporary, target.stat().st_mode & 0o777)
            if target.read_bytes() != original:
                raise ValueError("source changed during instrumentation")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return receipt


def metric(values: list[float | None]) -> dict:
    valid = [value for value in values if value is not None]
    return {"measured_count": len(valid), "unmeasured_count": len(values)-len(valid),
            "median": statistics.median(valid) if valid else None, "sum": sum(valid) if valid else None}


def finite_span(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def read_records(log: Path, schema: str) -> list[dict]:
    records = []
    for number, line in enumerate(log.read_text().splitlines(), 1):
        if schema not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed diagnostic at line {number}") from exc
        if row.get("schema") != schema:
            raise ValueError(f"unexpected diagnostic schema at line {number}")
        records.append(row)
    return records


def summarize(log: Path, baseline_log: Path | None = None) -> dict:
    native = whole.summarize(log, 128)
    own = read_records(log, SCHEMA)
    if any(row.get("kind") not in {"inventory", "capture", "operation"} for row in own):
        raise ValueError("unknown or failed operator diagnostic record")
    captures, inventories = {}, {}
    for row in own:
        if row["kind"] in {"capture", "inventory"}:
            table = captures if row["kind"] == "capture" else inventories
            if row["capture_id"] in table:
                raise ValueError("duplicate capture ID; summarize one process log at a time")
            table[row["capture_id"]] = row
    if not captures:
        raise ValueError("no packed graph capture diagnostics found")
    for capture, cfg in captures.items():
        if cfg["source_sha256"] != SOURCE_SHA256 or cfg["historical_tool_sha256"] != HISTORICAL_SHA256:
            raise ValueError("capture provenance does not match supported source")
        if cfg["phase"] not in {"prefill", "decode"} or cfg["operator"] not in ("off", *FAMILIES):
            raise ValueError("unknown capture phase or family")
        mask = cfg["layer_mask"]
        allowed_masks = {0} if cfg["operator"] == "off" else ({1 << 30} if cfg["operator"] == "head" else {(1 << 0) | (1 << 15) | (1 << 29), (1 << 30)-1})
        if mask not in allowed_masks or cfg["interval_count"] != mask.bit_count():
            raise ValueError("invalid layer selection")
        expected = 2 * mask.bit_count() if cfg["phase"] == "decode" else 0
        if cfg["expected_event_records"] != expected or cfg["recorded_event_records"] != expected:
            raise ValueError("capture has missing event edges")
        inventory = inventories.get(capture)
        if inventory is None or inventory["inventory_status"] != 0:
            raise ValueError("missing or failed external-event inventory")
        # Historical projection mode has its own events. It is mutually exclusive
        # with active decode mode; report its nodes without relabeling them.
        if cfg["operator"] != "off" and (inventory["event_record_nodes"] != expected or inventory["event_wait_nodes"] != 0):
            raise ValueError("external-event node count disagrees with selected family")
    old_replays = {(r["capture_id"], r["replay_id"]): r for r in read_records(log, whole.SCHEMA) if r["kind"] == "replay"}
    for (capture, _), row in old_replays.items():
        cfg = captures.get(capture)
        if cfg is not None:
            position = row["position"]
            if (cfg["phase"] == "prefill" and position != 127) or (cfg["phase"] == "decode" and not 128 <= position < 160):
                raise ValueError("whole replay position disagrees with its captured stage")
    operations = {}
    for row in own:
        if row["kind"] != "operation":
            continue
        key = (row["capture_id"], row["replay_id"])
        if key in operations:
            raise ValueError("duplicate operator replay")
        cfg = captures.get(key[0])
        old = old_replays.get(key)
        if cfg is None or cfg["phase"] != "decode" or cfg["operator"] == "off" or row["operator"] != cfg["operator"]:
            raise ValueError("operator replay has incorrect capture or family")
        if not 128 <= row["position"] < 160 or old is None or any(row[field] != old[field] for field in ("position", "launch_status", "completion_status")):
            raise ValueError("operator replay is not bound to its actual completed decode replay")
        expected_layers = [None if i == 30 else i for i in range(31) if cfg["layer_mask"] & (1 << i)]
        spans = {}
        for interval in row["intervals"]:
            layer = interval["layer"]
            if layer not in expected_layers or layer in spans:
                raise ValueError("unexpected or duplicate interval layer")
            value = interval.get("cuda_span_ns")
            spans[layer] = value if row["launch_status"] == 0 and row["completion_status"] == 0 and interval["event_status"] == 0 and finite_span(value) else None
        values = [spans.get(layer) for layer in expected_layers]
        total = sum(values) if all(value is not None for value in values) else None
        emitted = row.get("cuda_sum_ns")
        if emitted is not None and (total is None or not finite_span(emitted) or abs(emitted-total) > 0.001 * (len(values)+1)):
            raise ValueError("operator sum contradicts measured intervals")
        # A missing aggregate field is unmeasured, never reconstructed as success.
        if emitted is None:
            total = None
        operations[key] = {"capture_id": key[0], "replay_id": key[1], "position": row["position"], "operator": row["operator"], "cuda_sum_ns": total, "layers": dict(zip(expected_layers, values))}
    # Missing whole operator records count as unknown, rather than dropping slow
    # or failed replays from the distribution.
    for key, row in old_replays.items():
        cfg = captures.get(key[0])
        if cfg is not None and cfg["phase"] == "decode" and cfg["operator"] != "off" and key not in operations:
            missing = {None if i == 30 else i: None for i in range(31) if cfg["layer_mask"] & (1 << i)}
            operations[key] = {"capture_id": key[0], "replay_id": key[1], "position": row["position"], "operator": cfg["operator"], "cuda_sum_ns": None, "layers": missing}
    groups = []
    for family in FAMILIES:
        rows = [row for row in operations.values() if row["operator"] == family]
        if not rows:
            continue
        layers = sorted({layer for row in rows for layer in row["layers"]}, key=lambda item: -1 if item is None else item)
        groups.append({"operator": family, "replays": len(rows), "sum_across_selected_layers_ns": metric([row["cuda_sum_ns"] for row in rows]),
            "layers": [{"layer": layer, "cuda_span_ns": metric([row["layers"].get(layer) for row in rows])} for layer in layers],
            "positions": [{"position": pos, "cuda_sum_ns": metric([row["cuda_sum_ns"] for row in rows if row["position"] == pos])} for pos in sorted({row["position"] for row in rows})]})
    result = {"schema_version": "riley.decode-operator-summary.v1", "performance_claim_eligible": False,
        "log_sha256": sha256(log.read_bytes()), "captures": list(captures.values()), "external_event_inventories": list(inventories.values()),
        "whole_graph": native, "operator_groups": groups,
        "replay_totals": [{k: v for k, v in row.items() if k != "layers"} for row in operations.values()],
        "interpretation": ["CUDA spans include the perturbation from external event nodes; these are not serving performance results.",
            "Sum is formed per replay before taking its median; failed or missing intervals produce a null replay sum.",
            "Host wait includes GPU execution and is not added to GPU operator or whole-graph spans.",
            "Only one family is selected per process. Unselected families and initial norm/embedding/argmax/transfers are unmeasured.",
            "Position 127 is the batched prefill replay; captured handle IDs bind all decode operator results.",
            "Different captures/logs require matching binary, workload, GPU and environment receipts for comparison."]}
    if baseline_log is not None:
        baseline = summarize(baseline_log)
        if any(cfg["operator"] != "off" for cfg in baseline["captures"]) or any(inv["event_record_nodes"] or inv["event_wait_nodes"] for inv in baseline["external_event_inventories"]):
            raise ValueError("perturbation baseline must have all captured timing modes off")
        provenance = lambda report: {(r["source_sha256"], r["tool_sha256"], r["historical_tool_sha256"], r["projection_events_compiled"]) for r in report["captures"]}
        if provenance(result) != provenance(baseline):
            raise ValueError("perturbation logs have different instrumentation provenance")
        comparisons = []
        for phase in ("prefill", "decode"):
            before = baseline["whole_graph"]["groups"][phase]["metrics"]["cuda_graph_span_ns"]["median"]
            after = native["groups"][phase]["metrics"]["cuda_graph_span_ns"]["median"]
            comparisons.append({"phase": phase, "off_median_ns": before, "selected_median_ns": after, "ratio": after/before if before and after is not None else None})
        result["whole_graph_perturbation"] = {"baseline_log_sha256": baseline["log_sha256"], "groups": comparisons, "matched_runtime_provenance_verified": False, "performance_claim_eligible": False}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    patch = sub.add_parser("instrument")
    patch.add_argument("--source-root", type=Path, required=True)
    patch.add_argument("--apply", action="store_true")
    patch.add_argument("--projection-events", action="store_true", help="unsupported for batch8 decode: rejected because the active M16 prefill path is not instrumented")
    summary = sub.add_parser("summarize")
    summary.add_argument("--log", type=Path, required=True)
    summary.add_argument("--baseline-log", type=Path, help="same diagnostic build with both captured timing modes off")
    args = parser.parse_args(argv)
    try:
        result = instrument(args.source_root, args.apply, args.projection_events) if args.command == "instrument" else summarize(args.log, args.baseline_log)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
