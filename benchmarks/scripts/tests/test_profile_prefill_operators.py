from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "profile_prefill_operators.py"
SPEC = importlib.util.spec_from_file_location("prefill_operator_profiler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


def log_records(family="qkv", samples=((1, 100, 100), (100, 1, 100), (100, 100, 1))):
    layers = tuple(range(30)) if len(samples[0]) == 30 else (0, 15, 29)
    mask = 0 if family == "off" else sum(1 << layer for layer in layers)
    records = []
    for capture, phase in ((1, "prefill"), (2, "decode")):
        events = mask.bit_count()*2 if phase == "prefill" else 0
        records += [{"schema": profiler.SCHEMA, "kind": "capture", "capture_id": capture,
            "phase": phase, "operator": family, "layer_mask": mask, "interval_count": mask.bit_count(),
            "expected_event_records": events, "recorded_event_records": events,
            "source_sha256": profiler.SOURCE_SHA256, "historical_tool_sha256": profiler.HISTORICAL_SHA256,
            "tool_sha256": profiler.sha256(SCRIPT.read_bytes())},
            {"schema": profiler.SCHEMA, "kind": "inventory", "capture_id": capture,
             "inventory_status": 0, "event_record_nodes": events, "event_wait_nodes": 0}]
    replay = 0
    for sample in samples:
        for position in range(127, 159):
            replay += 1
            records.append({"schema": profiler.whole.SCHEMA, "kind": "replay", "capture_id": 1 if position == 127 else 2,
                "replay_id": replay, "position": position, "event_setup_ns": 2, "host_staging_ns": 3,
                "host_launch_ns": 4, "host_wait_ns": 10000, "cuda_graph_span_ns": 500 if position == 127 else 300,
                "event_status": 0, "launch_status": 0, "completion_status": 0})
            if position == 127 and family != "off":
                records.append({"schema": profiler.SCHEMA, "kind": "operation", "capture_id": 1,
                    "replay_id": replay, "position": 127, "operator": family,
                    "launch_status": 0, "completion_status": 0,
                    "intervals": [{"layer": layer, "cuda_span_ns": value, "event_status": 0} for layer, value in zip(layers, sample)],
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


def compile_run(source):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise unittest.SkipTest("C++ compiler unavailable")
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory)/"probe.cpp", Path(directory)/"probe"
        cpp.write_text(source)
        build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)], capture_output=True, text=True)
        if build.returncode:
            raise AssertionError(build.stderr)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode:
            raise AssertionError(run.stderr)
        return [json.loads(line) for line in run.stderr.splitlines()]


class PrefillOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()

    def test_preview_apply_and_isolation_guards(self):
        with self.assertRaisesRegex(ValueError, "tooling checkout"):
            profiler.instrument(profiler.REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / profiler.SOURCE
            target.parent.mkdir(parents=True)
            target.write_text(self.original)
            preview = profiler.instrument(root)
            self.assertEqual(target.read_text(), self.original)
            applied = profiler.instrument(root, True)
            self.assertEqual(preview["instrumented_sha256"], applied["instrumented_sha256"])
            self.assertFalse(applied["performance_claim_eligible"])
            self.assertEqual(applied["families"], ["qkv", "o", "gate_up", "down", "attention"])
            with self.assertRaisesRegex(ValueError, "already instrumented"):
                profiler.instrument(root, True)
            self.assertEqual(list(target.parent.glob(".prefill-operators-*")), [])
            target.unlink()
            target.symlink_to(profiler.REPOSITORY_ROOT / profiler.SOURCE)
            with self.assertRaisesRegex(ValueError, "tooling checkout"):
                profiler.instrument(root, True)
            target.unlink()
            os.link(profiler.REPOSITORY_ROOT / profiler.SOURCE, target)
            with self.assertRaisesRegex(ValueError, "tooling checkout"):
                profiler.instrument(root, True)
        self.assertEqual((profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text(), self.original)

    def test_unknown_source_and_strict_group_anchors_fail_closed(self):
        for changed in (self.original+"\n", self.original.replace("gemm(0,3)", "changed()")):
            with self.assertRaisesRegex(ValueError, "unknown source SHA256"):
                profiler.instrument_text(changed)
        for changed in (self.original.replace("gemm(1,6)", "changed()"), self.original + "\n// gemm(3,3)\n"):
            with patch.object(profiler, "SOURCE_SHA256", profiler.sha256(changed.encode())):
                with self.assertRaisesRegex(ValueError, "anchor must occur exactly once"):
                    profiler.instrument_text(changed)

    def test_exact_groups_prefill_gating_and_cleanup_preserve_original_sync(self):
        source = profiler.instrument_text(self.original)
        self.assertEqual(source.count("cudaStreamSynchronize("), 1)
        self.assertNotIn("cudaEventSynchronize(", source)
        self.assertNotIn("RileyProjectionDiagnostic", source)
        for family, projections in profiler.PROJECTION_GROUPS.items():
            self.assertEqual(source.count(f"time_prefill_operator(RileyPrefillOperator::{family},l,"), 1)
            first, *rest = projections
            self.assertIn(f"auto result=gemm({first[0]},{first[1]});", source)
            for j, out in rest:
                self.assertIn(f"if(result==RILEY_CUDA_STATUS_SUCCESS)result=gemm({j},{out});", source)
            positions = [source.index(f"gemm({j},{out})") for j, out in projections]
            self.assertEqual(positions, sorted(positions))
        for family, expression in profiler.OPERATOR_EXPRESSIONS.items():
            self.assertIn(f"time_prefill_operator(RileyPrefillOperator::{family},l,[&]() noexcept {{return {expression};}})", source)
            self.assertEqual(source.count(expression), 1)
        # The packed decode kernel and projection launch expressions stay exact.
        for line in self.original.splitlines():
            if "packed decode QKV GEMM" in line or "packed decode gate/up GEMM" in line or "enqueue_compiled_packed_decode_attention(" in line:
                self.assertIn(line, source)
        self.assertIn("record(batched,family,layer,0", source)
        self.assertLess(source.index("diagnostic_prefill_operator.prepare()"), source.index("  auto record_stage"))
        self.assertLess(source.index("cudaStreamSynchronize("), source.index("r->diagnostic_prefill_operator.finish("))
        self.assertLess(source.index("cudaGraphDestroy((*resources)->prefill_graph)"), source.index("diagnostic_prefill_operator.close()"))
        self.assertLess(source.index("diagnostic_prefill_operator.close()"), source.index("status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose)"))
        self.assertIn("|| (*resources)->diagnostic_prefill_operator.has_events()", source)
        self.assertLess(source.index("if ((*resources)->completion_unknown)"), source.index("diagnostic_prefill_operator.close()"))
        self.assertIn("r->diagnostic_prefill_capture_id=r->diagnostic_capture_id;", source)

    def test_per_replay_sum_precedes_median_and_decode_stays_unsampled(self):
        summary = summarize_records(log_records())
        total = summary["operator_groups"][0]["sum_across_selected_layers_ns"]
        self.assertEqual(total, {"median": 201, "sum": 603, "measured_count": 3, "unmeasured_count": 0})
        self.assertEqual(sum(row["cuda_span_ns"]["median"] for row in summary["operator_groups"][0]["layers"]), 300)
        self.assertEqual(summary["whole_graph"]["groups"]["prefill"]["replays"], 3)
        self.assertEqual(summary["whole_graph"]["groups"]["decode"]["replays"], 93)
        self.assertEqual(summary["whole_graph"]["projection_groups"], [])
        self.assertEqual(len(summary["replay_totals"]), 3)
        self.assertTrue(all(row["position"] == 127 for row in summary["replay_totals"]))
        self.assertFalse(summary["performance_claim_eligible"])
        all_layers = summarize_records(log_records(samples=(tuple(range(30)),)*3))
        self.assertEqual(len(all_layers["operator_groups"][0]["layers"]), 30)
        self.assertEqual(all_layers["operator_groups"][0]["sum_across_selected_layers_ns"]["median"], 435)
        self.assertEqual(summarize_records(log_records("off"))["operator_groups"], [])

    def test_missing_and_failed_intervals_and_records_remain_null(self):
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
            if row["kind"] in ("replay", "operation") and row["position"] == 127:
                row["completion_status"] = 999
                if row["kind"] == "operation":
                    row["cuda_sum_ns"] = None
        self.assertIsNone(summarize_records(rows)["replay_totals"][0]["cuda_sum_ns"])
        rows = log_records(samples=((1, 2, 3),))
        next(row for row in rows if row["kind"] == "operation").pop("cuda_sum_ns")
        self.assertIsNone(summarize_records(rows)["replay_totals"][0]["cuda_sum_ns"])

    def test_node_capture_replay_provenance_and_sum_mismatches_are_rejected(self):
        source = log_records()
        for change in ("nodes", "decode_nodes", "capture", "sum", "position", "duplicate", "duplicate_whole", "source"):
            rows = copy.deepcopy(source)
            operation = next(row for row in rows if row["kind"] == "operation")
            if change in ("nodes", "decode_nodes"):
                capture = 1 if change == "nodes" else 2
                next(row for row in rows if row["kind"] == "inventory" and row["capture_id"] == capture)["event_record_nodes"] = 2
            elif change == "capture":
                operation["capture_id"] = 2
            elif change == "sum":
                operation["cuda_sum_ns"] = 10000
            elif change == "position":
                operation["position"] = 128
            elif change == "duplicate":
                rows.append(copy.deepcopy(operation))
            elif change == "duplicate_whole":
                rows.append(copy.deepcopy(next(row for row in rows if row["kind"] == "replay")))
            else:
                next(row for row in rows if row["kind"] == "capture")["source_sha256"] = "wrong"
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_records(rows)

    def test_perturbation_requires_off_captured_nodes_and_matching_provenance(self):
        on, off = log_records(), log_records("off")
        report = summarize_records(on, off)["whole_graph_perturbation"]
        self.assertEqual(report["groups"][0]["ratio"], 1)
        self.assertFalse(report["performance_claim_eligible"])
        self.assertFalse(report["matched_runtime_provenance_verified"])
        with self.assertRaisesRegex(ValueError, "all captured timing modes off"):
            summarize_records(on, on)
        for row in off:
            if row["kind"] == "capture":
                row["tool_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different instrumentation provenance"):
            summarize_records(on, off)

    def test_cpp_cold_all30_sampling_failures_cleanup_and_json(self):
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
  unsetenv("RILEY_OWNED_GRAPH_PROJECTION");unsetenv("RILEY_DECODE_OPERATOR");unsetenv("RILEY_PREFILL_OPERATOR");unsetenv("RILEY_PREFILL_OPERATOR_LAYERS");
  RileyPrefillOperatorDiagnostic off;assert(off.prepare()==cudaSuccess);assert(!off.has_events()&&creates==0);
  off.begin_capture(0);off.bind_capture(1);off.begin_capture(1);off.bind_capture(2);
  setenv("RILEY_PREFILL_OPERATOR","qkv",1);
  RileyPrefillOperatorDiagnostic diag;assert(diag.prepare()==cudaSuccess);assert(creates==60);assert(diag.mask==(1u<<30)-1);
  diag.begin_capture(0);capturing=true;
  for(uint32_t l=0;l<30;++l)for(uint32_t edge=0;edge<2;++edge){
    assert(diag.record(true,RileyPrefillOperator::attention,l,edge,nullptr)==cudaSuccess);
    assert(diag.record(true,RileyPrefillOperator::qkv,l,edge,nullptr)==cudaSuccess);
  }
  capturing=false;diag.bind_capture(11);assert(records==60);
  diag.begin_capture(1);capturing=true;
  for(uint32_t l=0;l<30;++l)for(uint32_t edge=0;edge<2;++edge)assert(diag.record(false,RileyPrefillOperator::qkv,l,edge,nullptr)==cudaSuccess);
  capturing=false;diag.bind_capture(12);assert(records==60);
  diag.finish(12,2,128,cudaSuccess,cudaSuccess);assert(elapsed==0);
  diag.finish(11,1,127,cudaSuccess,cudaSuccess);assert(elapsed==30);
  diag.finish(11,33,127,cudaSuccess,cudaErrorUnknown);assert(elapsed==30);
  diag.finish(11,65,127,cudaErrorUnknown,cudaSuccess);assert(elapsed==30);
  diag.recorded[15][1]=false;diag.finish(11,97,127,cudaSuccess,cudaSuccess);assert(elapsed==59);
  diag.finish(11,129,128,cudaSuccess,cudaSuccess);assert(elapsed==59);
  diag.finish(99,161,127,cudaSuccess,cudaSuccess);assert(elapsed==59);
  assert(diag.close()==cudaSuccess);assert(destroys==60);assert(!diag.has_events());assert(diag.close()==cudaSuccess);assert(destroys==60);
  fail_create_at=creates+3;RileyPrefillOperatorDiagnostic partial;
  assert(partial.prepare()==cudaErrorUnknown);assert(partial.has_events());assert(partial.close()==cudaSuccess);assert(destroys==62);
  setenv("RILEY_PREFILL_OPERATOR_LAYERS","0,15,29",1);setenv("RILEY_PREFILL_OPERATOR","gate_up",1);
  RileyPrefillOperatorDiagnostic sampled;assert(sampled.prepare()==cudaSuccess);assert(sampled.mask==((1u<<0)|(1u<<15)|(1u<<29)));
  sampled.begin_capture(0);capturing=true;
  for(uint32_t l=0;l<30;++l)for(uint32_t edge=0;edge<2;++edge)assert(sampled.record(true,RileyPrefillOperator::gate_up,l,edge,nullptr)==cudaSuccess);
  capturing=false;sampled.bind_capture(20);sampled.finish(20,193,127,cudaSuccess,cudaSuccess);assert(sampled.close()==cudaSuccess);assert(destroys==68);
  for(const char* variable: {"RILEY_OWNED_GRAPH_PROJECTION","RILEY_DECODE_OPERATOR"}){
    setenv(variable,"qkv",1);RileyPrefillOperatorDiagnostic overlap;assert(overlap.prepare()==cudaErrorInvalidValue);assert(!overlap.has_events());unsetenv(variable);
  }
  setenv("RILEY_PREFILL_OPERATOR_LAYERS","30",1);RileyPrefillOperatorDiagnostic bad_layers;assert(bad_layers.prepare()==cudaErrorInvalidValue);
  setenv("RILEY_PREFILL_OPERATOR","unknown",1);RileyPrefillOperatorDiagnostic bad_family;assert(bad_family.prepare()==cudaErrorInvalidValue);
  enabled=false;RileyPrefillOperatorDiagnostic disabled;assert(disabled.prepare()==cudaSuccess);assert(!disabled.has_events());
  enabled=true;setenv("RILEY_PREFILL_OPERATOR","down",1);setenv("RILEY_PREFILL_OPERATOR_LAYERS","0,15,29",1);
  RileyPrefillOperatorDiagnostic ambiguous;assert(ambiguous.prepare()==cudaSuccess);fail_destroy_at=destroys+1;
  assert(ambiguous.close()==cudaErrorUnknown);assert(ambiguous.has_events()); // Owning close must retain; do not retry.
}
'''
        rows = compile_run("#include <initializer_list>\n" + stub + profiler.helper_text() + main)
        operations = [row for row in rows if row["kind"] == "operation"]
        self.assertEqual(len(operations), 6)
        self.assertAlmostEqual(operations[0]["cuda_sum_ns"], 150000, delta=0.01)
        self.assertEqual(len(operations[0]["intervals"]), 30)
        for row in operations[1:5]:
            self.assertIsNone(row["cuda_sum_ns"])
        self.assertIsNone(operations[3]["intervals"][15]["cuda_span_ns"])
        self.assertEqual(operations[3]["intervals"][0]["cuda_span_ns"], 5000)
        self.assertEqual(operations[-1]["cuda_sum_ns"], 15000)
        self.assertEqual([r["expected_event_records"] for r in rows if r["kind"] == "capture"], [0, 0, 60, 0, 6])

    def test_actual_composed_helpers_compile_and_emit_external_node_inventory(self):
        source = profiler.instrument_text(self.original)
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
  setenv("RILEY_OWNED_GRAPH_PROFILE","1",1);unsetenv("RILEY_PREFILL_OPERATOR");
  const auto capture=riley_diag_inventory(nullptr);
  RileyOwnedGraphDiagnostic whole(true,nullptr);whole.before_launch();whole.after_launch();whole.before_wait();whole.finish(capture,1,127,4,cudaSuccess,cudaSuccess);
  RileyPrefillOperatorDiagnostic operators;operators.prepare();operators.begin_capture(0);operators.bind_capture(capture);operators.close();
}
'''
        rows = compile_run(stub + helpers + main)
        inventory = next(row for row in rows if row["kind"] == "inventory")
        self.assertEqual(inventory["event_record_nodes"], 1)
        self.assertEqual(inventory["event_wait_nodes"], 1)
        graph = next(row for row in rows if row["kind"] == "graph")
        self.assertEqual((graph["other_nodes"], graph["kernel_nodes"]), (2, 1))
        self.assertEqual((graph["h2d_bytes"], graph["d2h_bytes"], graph["d2d_bytes"]), (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
