from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "profile_decode_rope_attention_baseline.py"
SPEC = importlib.util.spec_from_file_location("decode_operator_profiler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)
CAMPAIGN = SCRIPT.parents[1] / "results/20260912-serving-optimization"
BASELINE_BUILD = CAMPAIGN / "raw/http-token-build.json"


def log_records(family="rope_attention", samples=((1, 100, 100), (100, 1, 100), (100, 100, 1))):
    mask = 0 if family == "off" else (1 << 0) | (1 << 15) | (1 << 29)
    count = mask.bit_count()
    records = []
    for capture, phase in ((1, "prefill"), (2, "decode")):
        events = count*2 if phase == "decode" else 0
        records += [{"schema": profiler.SCHEMA, "kind": "capture", "capture_id": capture,
            "phase": phase, "operator": family, "layer_mask": mask, "interval_count": count,
            "expected_event_records": events, "recorded_event_records": events,
            "source_sha256": profiler.SOURCE_SHA256, "historical_tool_sha256": profiler.HISTORICAL_SHA256,
            "tool_sha256": profiler.sha256(SCRIPT.read_bytes()), "projection_events_compiled": False,
            "baseline_source_commit":profiler.BASELINE_SOURCE_COMMIT,"baseline_build_sha256":profiler.BASELINE_BUILD_SHA256},
            {"schema": profiler.SCHEMA, "kind": "inventory", "capture_id": capture,
             "inventory_status": 0, "event_record_nodes": events, "event_wait_nodes": 0}]
    for replay, sample in enumerate((None, *samples), 1):
        position = 126+replay
        records.append({"schema": profiler.whole.SCHEMA, "kind": "replay", "capture_id": 1 if sample is None else 2,
            "replay_id": replay, "position": position, "event_setup_ns": 2, "host_staging_ns": 3,
            "host_launch_ns": 4, "host_wait_ns": 10000, "cuda_graph_span_ns": 500 if sample is None else 300,
            "event_status": 0, "launch_status": 0, "completion_status": 0})
        if sample is not None and family != "off":
            records.append({"schema": profiler.SCHEMA, "kind": "operation", "capture_id": 2,
                "replay_id": replay, "position": position, "operator": family,
                "launch_status": 0, "completion_status": 0,
                "intervals": [{"layer": layer, "cuda_span_ns": value, "event_status": 0} for layer, value in zip((0, 15, 29), sample)],
                "cuda_sum_ns": sum(sample) if all(value is not None for value in sample) else None})
    return records


def summarize_records(records, baseline=None):
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "operators.log"
        log.write_text("unrelated native message\n" + "\n".join(json.dumps(row) for row in records))
        before = None
        if baseline is not None:
            before = Path(directory) / "off.log"
            before.write_text("\n".join(json.dumps(row) for row in baseline))
        return profiler.summarize(log, before)


class DecodeOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(CAMPAIGN / "batch7-source-overlay.tar.gz") as archive:
            cls.original = archive.extractfile(str(profiler.SOURCE)).read().decode()
        assert profiler.sha256(cls.original.encode()) == profiler.SOURCE_SHA256
        cls.checkout_bytes = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_bytes()

    def test_preview_apply_and_isolation_guards(self):
        with self.assertRaisesRegex(ValueError, "tooling checkout"):
            profiler.instrument(profiler.REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / profiler.SOURCE
            target.parent.mkdir(parents=True)
            target.write_text(self.original)
            preview = profiler.instrument(root, baseline_build=BASELINE_BUILD)
            self.assertEqual(target.read_text(), self.original)
            applied = profiler.instrument(root, True, False, BASELINE_BUILD)
            self.assertEqual(preview["instrumented_sha256"], applied["instrumented_sha256"])
            self.assertFalse(applied["performance_claim_eligible"])
            with self.assertRaisesRegex(ValueError, "already instrumented"):
                profiler.instrument(root, True, baseline_build=BASELINE_BUILD)
            self.assertEqual(list(target.parent.glob(".decode-operators-*")), [])
            target.unlink()
            target.symlink_to(profiler.REPOSITORY_ROOT / profiler.SOURCE)
            with self.assertRaisesRegex(ValueError, "tooling checkout"):
                profiler.instrument(root, True, baseline_build=BASELINE_BUILD)
            target.unlink()
            os.link(profiler.REPOSITORY_ROOT / profiler.SOURCE, target)
            with self.assertRaisesRegex(ValueError, "tooling checkout"):
                profiler.instrument(root, True, baseline_build=BASELINE_BUILD)
        self.assertEqual((profiler.REPOSITORY_ROOT / profiler.SOURCE).read_bytes(), self.checkout_bytes)

    def test_uninstrumented_prefill_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support historical prefill"):
            profiler.instrument_text(self.original, projection_events=True)

    def test_baseline_build_identity_is_required_and_byte_pinned(self):
        self.assertEqual(profiler.validate_baseline_build(BASELINE_BUILD)["sha256"],profiler.BASELINE_BUILD_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/profiler.SOURCE;target.parent.mkdir(parents=True);target.write_text(self.original)
            with self.assertRaisesRegex(ValueError,"explicit --baseline-build"):
                profiler.instrument(root)
            changed=root/'build.json';changed.write_bytes(BASELINE_BUILD.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,"receipt SHA256"):
                profiler.instrument(root,True,baseline_build=changed)
            self.assertEqual(target.read_text(),self.original)

    def test_unknown_sources_and_missing_or_duplicate_anchors_fail_closed(self):
        for change in (self.original + "\n", self.original.replace("gemm(3,3)", "gemm(3,4)")):
            with self.assertRaisesRegex(ValueError, "unknown source SHA256"):
                profiler.instrument_text(change)
        # Even an explicitly repinned fixture must retain exact single anchors.
        for changed in (self.original.replace(profiler.REGION_START, "missing_region()"), self.original + "\n" + profiler.REGION_END):
            with patch.object(profiler, "SOURCE_SHA256", profiler.sha256(changed.encode())):
                with self.assertRaisesRegex(ValueError, "anchor must occur exactly once"):
                    profiler.instrument_text(changed)

    def test_all_operator_enqueues_and_lifecycle_seams_preserve_original_sync(self):
        for projection in (False,):
            source = profiler.instrument_text(self.original, projection)
            self.assertEqual(source.count("cudaStreamSynchronize("), 1)
            self.assertNotIn("cudaEventSynchronize(", source)
            self.assertIn("cudaGraphNodeTypeEventRecord", source)
            self.assertIn("cudaGraphNodeTypeWaitEvent", source)
            for expression in (profiler.ROPE_EXPRESSION,profiler.ATTENTION_EXPRESSION):
                self.assertEqual(source.count(expression),1)
            start=self.original.index(profiler.REGION_START)
            end=self.original.index(profiler.REGION_END)+len(profiler.REGION_END)
            original_region=self.original[start:end]
            self.assertIn(profiler.START_EVENT+original_region+profiler.END_EVENT,source)
            self.assertEqual(source.count("record(packed_decode,RileyDecodeOperator::rope_attention"),2)
            self.assertNotIn("time_decode_operator",source)
            import re
            self.assertEqual(re.findall(r"\benqueue_\w+\(",source),re.findall(r"\benqueue_\w+\(",self.original))
            self.assertLess(source.index("r->diagnostic_operator.prepare()"), source.index("  auto record_stage"))
            self.assertLess(source.index("cudaStreamSynchronize("), source.index("r->diagnostic_operator.finish("))
            self.assertLess(source.index("cudaGraphDestroy((*resources)->prefill_graph)"), source.index("diagnostic_operator.close()"))
            self.assertLess(source.index("diagnostic_operator.close()"), source.index("status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose)"))
            self.assertIn("|| (*resources)->diagnostic_operator.has_events()", source)
            self.assertLess(source.index("if ((*resources)->completion_unknown)"), source.index("diagnostic_operator.close()"))
            self.assertEqual("struct RileyProjectionDiagnostic" in source, projection)
            if projection:
                self.assertIn("timed(1,[&]() noexcept {return kernel(enqueue_compiled_prefill_gemm_rows", source)
                self.assertIn("diagnostic_projection.close()", source)

    def test_per_replay_sum_precedes_median_without_host_double_counting(self):
        summary = summarize_records(log_records())
        metric = summary["operator_groups"][0]["sum_across_selected_layers_ns"]
        self.assertEqual(metric, {"median": 201, "sum": 603, "measured_count": 3, "unmeasured_count": 0})
        # Sum of layer medians would be 300; whole span is 300; host wait 10000.
        self.assertEqual(sum(row["cuda_span_ns"]["median"] for row in summary["operator_groups"][0]["layers"]), 300)
        self.assertEqual(summary["whole_graph"]["groups"]["prefill"]["replays"], 1)
        self.assertEqual(summary["whole_graph"]["groups"]["decode"]["replays"], 3)
        self.assertFalse(summary["performance_claim_eligible"])

    def test_failed_missing_intervals_and_missing_whole_records_remain_null(self):
        rows = log_records(samples=((1, None, 3), (4, 5, 6), (7, 8, 9)))
        operations = [row for row in rows if row["kind"] == "operation"]
        operations[1]["intervals"].pop()
        operations[1]["cuda_sum_ns"] = None
        rows.remove(operations[2])
        summary = summarize_records(rows)
        self.assertEqual([row["cuda_sum_ns"] for row in summary["replay_totals"]], [None, None, None])
        self.assertEqual(summary["operator_groups"][0]["sum_across_selected_layers_ns"]["unmeasured_count"], 3)
        rows = log_records(samples=((1, 2, 3),))
        for row in rows:
            if row["kind"] in ("replay", "operation") and row["position"] == 128:
                row["completion_status"] = 999
                if row["kind"] == "operation":
                    row["cuda_sum_ns"] = None
        summary = summarize_records(rows)
        self.assertIsNone(summary["replay_totals"][0]["cuda_sum_ns"])

    def test_invalid_node_capture_and_sum_bindings_are_rejected(self):
        source = log_records()
        for change in ("nodes", "capture", "sum", "position", "duplicate", "source", "build", "projection"):
            rows = copy.deepcopy(source)
            operation = next(row for row in rows if row["kind"] == "operation")
            if change == "nodes":
                next(row for row in rows if row["kind"] == "inventory" and row["capture_id"] == 2)["event_record_nodes"] = 2
            elif change == "capture":
                operation["capture_id"] = 1
            elif change == "sum":
                operation["cuda_sum_ns"] = 10000
            elif change == "position":
                operation["position"] = 127
            elif change == "duplicate":
                rows.append(copy.deepcopy(operation))
            elif change == "build":
                next(row for row in rows if row["kind"] == "capture")["baseline_build_sha256"] = "wrong"
            elif change == "projection":
                next(row for row in rows if row["kind"] == "capture")["projection_events_compiled"] = True
            else:
                next(row for row in rows if row["kind"] == "capture")["source_sha256"] = "wrong"
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_records(rows)

    def test_perturbation_requires_off_captured_nodes_and_matching_provenance(self):
        on, off = log_records(), log_records("off")
        report = summarize_records(on, off)["whole_graph_perturbation"]
        self.assertEqual(report["groups"][1]["ratio"], 1)
        self.assertFalse(report["performance_claim_eligible"])
        self.assertFalse(report["matched_runtime_provenance_verified"])
        with self.assertRaisesRegex(ValueError, "all captured timing modes off"):
            summarize_records(on, on)
        for row in off:
            if row["kind"] == "capture":
                row["tool_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "provenance"):
            summarize_records(on, off)

    def test_cpp_external_event_selection_lifecycle_failures_and_json(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        stub = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
using cudaEvent_t=void*;using cudaStream_t=void*;
enum cudaError_t {cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorUnknown=999};
constexpr unsigned int cudaEventRecordExternal=1;
bool capturing=false,enabled=true;int creates=0,destroys=0,records=0,elapsed=0,fail_create_at=0,fail_destroy_at=0;
bool riley_diag_enabled(){return enabled;}
cudaError_t cudaEventCreate(cudaEvent_t* event){assert(!capturing);++creates;if(creates==fail_create_at)return cudaErrorUnknown;*event=reinterpret_cast<void*>(static_cast<uintptr_t>(creates));return cudaSuccess;}
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,cudaStream_t,unsigned int flags){assert(capturing&&event!=nullptr&&flags==cudaEventRecordExternal);++records;return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){assert(!capturing);++elapsed;*ms=.005F;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t event){assert(!capturing&&event!=nullptr);++destroys;return destroys==fail_destroy_at?cudaErrorUnknown:cudaSuccess;}
'''
        main = r'''
int main(){
  unsetenv("RILEY_OWNED_GRAPH_PROJECTION");unsetenv("RILEY_DECODE_OPERATOR");unsetenv("RILEY_DECODE_OPERATOR_LAYERS");
  RileyDecodeOperatorDiagnostic off;assert(off.prepare()==cudaSuccess);assert(!off.has_events()&&creates==0);
  off.begin_capture(0);off.bind_capture(1);off.begin_capture(1);off.bind_capture(2);
  setenv("RILEY_DECODE_OPERATOR","rope_attention",1);
  RileyDecodeOperatorDiagnostic diag;assert(diag.prepare()==cudaSuccess);assert(creates==6);
  diag.begin_capture(0);capturing=true;
  for(uint32_t l=0;l<30;++l)for(uint32_t edge=0;edge<2;++edge)assert(diag.record(false,RileyDecodeOperator::rope_attention,l,edge,nullptr)==cudaSuccess);
  capturing=false;diag.bind_capture(11);assert(records==0);
  diag.begin_capture(1);capturing=true;
  for(uint32_t l=0;l<30;++l)for(uint32_t edge=0;edge<2;++edge){
    assert(diag.record(true,RileyDecodeOperator::off,l,edge,nullptr)==cudaSuccess);
    assert(diag.record(true,RileyDecodeOperator::rope_attention,l,edge,nullptr)==cudaSuccess);
  }
  capturing=false;diag.bind_capture(12);assert(records==6);
  diag.finish(11,1,127,cudaSuccess,cudaSuccess);assert(elapsed==0);
  diag.finish(12,2,128,cudaSuccess,cudaSuccess);assert(elapsed==3);
  diag.finish(12,3,129,cudaSuccess,cudaErrorUnknown);assert(elapsed==3);
  diag.finish(12,4,130,cudaErrorUnknown,cudaSuccess);assert(elapsed==3);
  diag.recorded[15][1]=false;diag.finish(12,5,131,cudaSuccess,cudaSuccess);assert(elapsed==5);
  diag.finish(12,6,160,cudaSuccess,cudaSuccess);assert(elapsed==5);
  diag.finish(99,7,132,cudaSuccess,cudaSuccess);assert(elapsed==5);
  assert(diag.close()==cudaSuccess);assert(destroys==6);assert(!diag.has_events());assert(diag.close()==cudaSuccess);assert(destroys==6);
  fail_create_at=creates+3;RileyDecodeOperatorDiagnostic partial;
  assert(partial.prepare()==cudaErrorUnknown);assert(partial.has_events());assert(partial.close()==cudaSuccess);assert(destroys==8);
  setenv("RILEY_DECODE_OPERATOR_LAYERS","all",1);RileyDecodeOperatorDiagnostic all;
  assert(all.prepare()==cudaSuccess);assert(all.mask==(1u<<30)-1);assert(all.close()==cudaSuccess);assert(destroys==68);
  setenv("RILEY_DECODE_OPERATOR","head",1);RileyDecodeOperatorDiagnostic head;
  assert(head.prepare()==cudaErrorInvalidValue);assert(!head.has_events());
  setenv("RILEY_DECODE_OPERATOR","rope_attention",1);
  setenv("RILEY_OWNED_GRAPH_PROJECTION","q",1);RileyDecodeOperatorDiagnostic overlap;assert(overlap.prepare()==cudaErrorInvalidValue);assert(!overlap.has_events());
  unsetenv("RILEY_OWNED_GRAPH_PROJECTION");setenv("RILEY_DECODE_OPERATOR_LAYERS","30",1);RileyDecodeOperatorDiagnostic bad_layers;assert(bad_layers.prepare()==cudaErrorInvalidValue);
  setenv("RILEY_DECODE_OPERATOR","unknown",1);RileyDecodeOperatorDiagnostic bad_family;assert(bad_family.prepare()==cudaErrorInvalidValue);
  enabled=false;RileyDecodeOperatorDiagnostic disabled;assert(disabled.prepare()==cudaSuccess);assert(!disabled.has_events());
  enabled=true;setenv("RILEY_DECODE_OPERATOR","rope_attention",1);setenv("RILEY_DECODE_OPERATOR_LAYERS","0,15,29",1);
  RileyDecodeOperatorDiagnostic ambiguous;assert(ambiguous.prepare()==cudaSuccess);fail_destroy_at=destroys+1;
  assert(ambiguous.close()==cudaErrorUnknown);assert(ambiguous.has_events()); // Owning close must retain; do not retry.
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "operators.cpp"
            binary = Path(directory) / "operators"
            source.write_text(stub + profiler.helper_text() + main)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        rows = [json.loads(line) for line in run.stderr.splitlines()]
        operations = [row for row in rows if row["kind"] == "operation"]
        self.assertEqual(len(operations), 5)
        self.assertEqual(operations[0]["cuda_sum_ns"], 15000)
        for row in operations[1:5]:
            self.assertIsNone(row["cuda_sum_ns"])
        self.assertIsNone(operations[3]["intervals"][1]["cuda_span_ns"])
        self.assertEqual(operations[3]["intervals"][0]["cuda_span_ns"], 5000)
        self.assertEqual([row["expected_event_records"] for row in rows if row["kind"] == "capture"], [0, 0, 0, 6])

    def test_composed_helpers_compile_and_inventory_external_record_and_wait_nodes(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        # Use the actual composed source prefix, including the modified old
        # inventory emitter and the decode helper class. No CUDA SDK or GPU
        # is claimed by this deterministic API/syntax test.
        source = profiler.instrument_text(self.original, projection_events=False)
        helpers = source.split(profiler.whole.MARKER, 1)[1].split("// The ledger remains", 1)[0]
        stub = r'''
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
using cudaGraph_t=void*;using cudaGraphNode_t=void*;using cudaEvent_t=void*;using cudaStream_t=void*;
enum cudaError_t {cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaErrorInvalidValue=1};
enum cudaGraphNodeType {cudaGraphNodeTypeKernel=0,cudaGraphNodeTypeMemcpy=1,cudaGraphNodeTypeEventRecord=2,cudaGraphNodeTypeWaitEvent=3};
enum cudaMemcpyKind {cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3};
constexpr unsigned int cudaEventRecordExternal=1;
struct cudaExtent {size_t width=0,height=0,depth=0;};
struct cudaMemcpy3DParms {void* srcArray=nullptr;void* dstArray=nullptr;cudaExtent extent{};cudaMemcpyKind kind=cudaMemcpyHostToDevice;};
cudaError_t cudaGraphGetNodes(cudaGraph_t,cudaGraphNode_t* nodes,size_t* count){if(nodes)for(uintptr_t i=0;i<6;++i)nodes[i]=reinterpret_cast<void*>(i+1);*count=6;return cudaSuccess;}
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,cudaGraphNodeType* type){auto id=reinterpret_cast<uintptr_t>(node);*type=id==1?cudaGraphNodeTypeKernel:id<5?cudaGraphNodeTypeMemcpy:id==5?cudaGraphNodeTypeEventRecord:cudaGraphNodeTypeWaitEvent;return cudaSuccess;}
cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,cudaMemcpy3DParms* p){p->extent={4,1,1};p->kind=static_cast<cudaMemcpyKind>(reinterpret_cast<uintptr_t>(node)-1);return cudaSuccess;}
cudaError_t cudaEventCreate(cudaEvent_t* event){*event=(void*)1;return cudaSuccess;}
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return cudaSuccess;}
cudaError_t cudaEventRecordWithFlags(cudaEvent_t,cudaStream_t,unsigned int){return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){*ms=.01F;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t){return cudaSuccess;}
'''
        main = r'''
int main(){
  setenv("RILEY_OWNED_GRAPH_PROFILE","1",1);unsetenv("RILEY_OWNED_GRAPH_PROJECTION");unsetenv("RILEY_DECODE_OPERATOR");
  const auto capture=riley_diag_inventory(nullptr);
  RileyOwnedGraphDiagnostic whole(true,nullptr);whole.before_launch();whole.after_launch();whole.before_wait();whole.finish(capture,1,127,4,cudaSuccess,cudaSuccess);
  RileyDecodeOperatorDiagnostic operators;operators.prepare();operators.begin_capture(0);operators.bind_capture(capture);operators.close();
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp, binary = Path(directory)/"composed.cpp", Path(directory)/"composed"
            cpp.write_text(stub + helpers + main)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        rows = [json.loads(line) for line in run.stderr.splitlines()]
        inventory = next(row for row in rows if row["kind"] == "inventory")
        self.assertEqual(inventory["event_record_nodes"], 1)
        self.assertEqual(inventory["event_wait_nodes"], 1)
        graph = next(row for row in rows if row["kind"] == "graph")
        self.assertEqual(graph["other_nodes"], 2)
        self.assertEqual(graph["kernel_nodes"], 1)
        self.assertEqual(graph["h2d_bytes"], 4)
        self.assertEqual(graph["d2h_bytes"], 4)
        self.assertEqual(graph["d2d_bytes"], 4)
        self.assertFalse(next(row for row in rows if row["kind"] == "capture")["projection_events_compiled"])

    def test_actual_combined_region_preserves_order_and_each_failure_short_circuit(self):
        compiler=shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        source=profiler.instrument_text(self.original)
        start=source.index(profiler.START_EVENT)
        end=source.index(profiler.END_EVENT,start)+len(profiler.END_EVENT)
        region=source[start:end]
        stub=r'''
#include <cassert>
#include <cstdint>
#include <vector>
using cudaStream_t=void*;
using cudaError_t=int;
constexpr int RILEY_CUDA_STATUS_SUCCESS=0;
enum class RileyDecodeOperator {off,rope_attention};
std::vector<int> calls;int fail=-1;bool active=true;
int visit(int id){calls.push_back(id);return id==fail?999:0;}
struct Diagnostic{int record(bool packed,RileyDecodeOperator,uint64_t,uint32_t edge,cudaStream_t){return packed&&active?visit(edge==0?0:3):0;}};
struct Stream{cudaStream_t stream=nullptr;};
struct Owner{Stream* stream;Diagnostic diagnostic_operator;};
struct Buffer{void* device_data;uint64_t byte_len=128;};
template<class... T> int enqueue_compiled_packed_decode_rope_kv(T...){return visit(1);}
template<class... T> int enqueue_compiled_packed_decode_attention_two_warp(T...){return visit(2);}
template<class... T> int enqueue_compiled_rope_rows(T...){return visit(4);}
template<class... T> int enqueue_qkv_rope(T...){return visit(5);}
template<class... T> int enqueue_compiled_kv_write_rows(T...){return visit(6);}
template<class... T> int enqueue_decode_kv_attention(T...){return visit(7);}
int execute(bool packed_decode,int initial=0){
 uint8_t bytes[4096]{};Buffer storage{bytes};Buffer* d[20];for(auto& p:d)p=&storage;
 Buffer* packed_buffers[62];for(auto& p:packed_buffers)p=&storage;
 Stream stream;Owner owner{&stream,{}};auto* r=&owner;
 auto buffer=[&](int i){return d[i]->device_data;};auto kernel=[](int error){return error;};
 const uint64_t l=0,k=192,h=576,physical=40,capacity=10;const uint32_t profile=2,rows=1;
 int s=initial;
'''
        main=r'''
 return s;
}
int main(){
 for(int failure=-1;failure<=3;++failure){calls.clear();fail=failure;active=true;
  const int status=execute(true);assert(status==(failure<0?0:999));
  const int last=failure<0?3:failure;assert(calls.size()==static_cast<size_t>(last+1));
  for(int i=0;i<=last;++i)assert(calls[i]==i);
 }
 calls.clear();fail=-1;active=false;assert(execute(true)==0);assert((calls==std::vector<int>{1,2}));
 calls.clear();active=true;assert(execute(false)==0);assert((calls==std::vector<int>{4,6}));
 calls.clear();assert(execute(true,999)==999);assert(calls.empty());
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp,binary=Path(directory)/'region.cpp',Path(directory)/'region'
            cpp.write_text(stub+region+main)
            build=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)


if __name__ == "__main__":
    unittest.main()
