from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "profile_owned_graph.py"
SPEC = importlib.util.spec_from_file_location("profile_owned_graph", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profiler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profiler
SPEC.loader.exec_module(profiler)


# Exact original single-graph patch seams, kept independently of the production
# source's later dual-graph additions. This is an instrumentation contract
# fixture, not a compilable runtime or numerical-correctness fixture.
LEGACY_SOURCE = '''#include <cmath>
struct RileyCudaGraphResources {
  cudaGraphExec_t exec = nullptr;
};
void replay() {
  r->terminal = true;
  std::memmove(r->input->host_data, source, static_cast<size_t>(bytes));
  auto launched = cudaGraphLaunch(r->exec, r->stream->stream);
  auto completed = cudaStreamSynchronize(r->stream->stream);
  if (completed == cudaSuccess) r->completion_unknown = false;
}
void read() {
  std::memmove(destination, static_cast<uint8_t*>(r->output->host_data) + r->output_byte_offset, static_cast<size_t>(bytes));
}
void captured() {
    r->output_byte_offset = transfer_bytes; r->terminal = false;
}
// Full single-row model DAG.
void record() { existing_arithmetic_and_dependencies(); }
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
#endif
'''


def legacy_projection_fixture() -> str:
    # The original recorder's exact GEMM expressions and preparation/close
    # seams, independent of the newer production topology.
    return LEGACY_SOURCE.replace("void record() { existing_arithmetic_and_dependencies(); }", '''void record() {
  std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,static_cast<size_t>(transfer));
  status=record_reserved_sequence(r,staging,transfer,[&]() noexcept {
      auto gemm=[&](size_t j,size_t out){
        auto status=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[out],states[l*7+j],error,"decode GEMM");
        if(status==RILEY_CUDA_STATUS_SUCCESS&&profile==2){
          status=kernel(enqueue_compiled_prefill_gemm(r->stream->stream,d[inputs[j]]->device_data,weights[weight_ids[j]]->device_data,d[out]->device_data,ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4));
        }
        return status;
      };
    },error);
}
void close() {
  if ((*resources)->graph != nullptr || (*resources)->exec != nullptr) {
    destroy_graphs();
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
  }
}''')


class OwnedGraphInstrumentationTests(unittest.TestCase):
    def test_copy_is_required_and_preview_does_not_modify_source(self) -> None:
        original = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_bytes()
        with self.assertRaisesRegex(ValueError, "tooling checkout"):
            profiler.instrument(profiler.REPOSITORY_ROOT, False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / profiler.SOURCE
            target.parent.mkdir(parents=True)
            target.write_bytes(original)
            preview = profiler.instrument(root, False)
            self.assertEqual(target.read_bytes(), original)
            applied = profiler.instrument(root, True)
            self.assertEqual(preview["instrumented_sha256"], applied["instrumented_sha256"])
            self.assertFalse(applied["performance_claim_eligible"])
            with self.assertRaisesRegex(ValueError, "already instrumented"):
                profiler.instrument(root, True)
        self.assertEqual((profiler.REPOSITORY_ROOT / profiler.SOURCE).read_bytes(), original)

    def test_missing_or_duplicate_anchor_cannot_partially_patch(self) -> None:
        original = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()
        for changed in (original.replace("#include <cmath>\n", ""), "#include <cmath>\n" + original):
            with self.subTest(changed=changed[:35]):
                with self.assertRaisesRegex(ValueError, "anchor must occur exactly once"):
                    profiler.instrument_text(changed)

    def test_graph_math_and_original_single_completion_are_unchanged(self) -> None:
        for original in [LEGACY_SOURCE, (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()]:
            patched = profiler.instrument_text(original)
            self.assertEqual(original.count("cudaStreamSynchronize("), patched.count("cudaStreamSynchronize("))
            self.assertNotIn("cudaEventSynchronize(", patched)
            # Only a host-side diagnostic ID is added while moving the first
            # capture into its prefill slot. Arithmetic/dependencies stay exact.
            without_id = patched.replace("\n      r->diagnostic_prefill_capture_id=r->diagnostic_capture_id;", "")
            start = '// Full single-row model DAG.'
            end = '#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)'
            self.assertEqual(original.split(start, 1)[1].split(end, 1)[0], without_id.split(start, 1)[1].split(end, 1)[0])
            self.assertIn("diagnostic.before_launch();\n  auto launched", patched)
            self.assertIn("diagnostic.before_wait();\n  auto completed", patched)

    def test_single_and_dual_graph_selection_are_unambiguous(self) -> None:
        single = profiler.instrument_text(LEGACY_SOURCE)
        self.assertIn("r->diagnostic_selected_capture_id=r->diagnostic_capture_id;", single)
        original = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()
        self.assertIn("cudaGraphLaunch(selected_exec,", original)
        dual = profiler.instrument_text(original)
        self.assertIn("r->diagnostic_prefill_capture_id=r->diagnostic_capture_id;", dual)
        self.assertIn("r->diagnostic_selected_capture_id=selected_exec==r->prefill_exec?r->diagnostic_prefill_capture_id:r->diagnostic_capture_id;", dual)
        self.assertIn("diagnostic.finish(r->diagnostic_selected_capture_id,", dual)
        read_part = dual.split("const auto diagnostic_read_begin=", 1)[1]
        self.assertIn("static_cast<unsigned long long>(r->diagnostic_selected_capture_id)", read_part)
        selection = next(anchor for anchor in profiler.DUAL_SELECTIONS if anchor in original)
        for alternative in profiler.DUAL_SELECTIONS:
            supported = original.replace(selection, alternative)
            self.assertIn(alternative, profiler.instrument_text(supported))
        for changed in [
            original.replace(selection, "      selected_exec=other_exec;"),
            original + "\n" + selection,
            original + "\n" + next(anchor for anchor in profiler.DUAL_SELECTIONS if anchor != selection),
            original.replace("r->prefill_graph=r->graph;r->prefill_exec=r->exec;", "changed_prefill_binding();"),
            original.replace("cudaGraphLaunch(selected_exec,", "cudaGraphLaunch(r->exec,"),
            original + "\n  auto launched = cudaGraphLaunch(r->exec, r->stream->stream);\n",
        ]:
            with self.assertRaises(ValueError):
                profiler.instrument_text(changed)

    def test_batched_prefill_receipt_and_position_classification(self) -> None:
        original = (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()
        self.assertIn(profiler.DUAL_SELECTIONS[1], original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / profiler.SOURCE
            target.parent.mkdir(parents=True)
            target.write_text(original)
            receipt = profiler.instrument(root, False)
            self.assertEqual(receipt["source_topology"], "dual_graph_prefill128_v1")
            self.assertFalse(receipt["projection_events"])
            log = root / "batched-stderr.log"
            records = []
            for replay, position in enumerate(range(127, 159), 1):
                records.append({"schema": profiler.SCHEMA, "kind": "replay",
                    "capture_id": 1 if position == 127 else 2,
                    "replay_id": replay, "position": position, "event_setup_ns": 10,
                    "host_staging_ns": 1, "host_launch_ns": 2, "host_wait_ns": 3,
                    "cuda_graph_span_ns": 1280 if position == 127 else 30,
                    "launch_status": 0, "completion_status": 0})
            log.write_text("\n".join(json.dumps(record) for record in records))
            result = profiler.summarize(log, 128)
            self.assertEqual(result["groups"]["prefill"]["replays"], 1)
            self.assertEqual(result["groups"]["decode"]["replays"], 31)
            self.assertEqual(result["groups"]["prefill"]["capture_ids"], [1])
            self.assertEqual(result["groups"]["decode"]["capture_ids"], [2])
            self.assertEqual(result["groups"]["prefill"]["metrics"]["cuda_graph_span_ns"]["sum"], 1280)

    def test_escaping_and_runtime_emitter_compile_and_emit_valid_json(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        # Compile and execute the inserted helper against a deterministic CUDA
        # API test double. This checks C++ syntax, JSON escaping and null timing
        # on failed completion; it does not claim a real CUDA build or GPU test.
        stub = r'''
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cmath>
using cudaGraph_t=void*;using cudaGraphNode_t=void*;using cudaEvent_t=void*;using cudaStream_t=void*;
enum cudaError_t {cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaErrorUnknown=999};
enum cudaGraphNodeType {cudaGraphNodeTypeKernel=0,cudaGraphNodeTypeMemcpy=1};
enum cudaMemcpyKind {cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3};
struct cudaExtent {size_t width=0,height=0,depth=0;};
struct cudaMemcpy3DParms {void* srcArray=nullptr;void* dstArray=nullptr;cudaExtent extent{};cudaMemcpyKind kind=cudaMemcpyHostToDevice;};
bool unsupported_copy=false;
cudaError_t cudaGraphGetNodes(cudaGraph_t,cudaGraphNode_t* nodes,size_t* count){if(nodes){nodes[0]=(void*)1;nodes[1]=(void*)2;nodes[2]=(void*)3;nodes[3]=(void*)4;}*count=4;return cudaSuccess;}
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,cudaGraphNodeType* type){*type=node==(void*)1?cudaGraphNodeTypeKernel:cudaGraphNodeTypeMemcpy;return cudaSuccess;}
cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,cudaMemcpy3DParms* p){
  p->extent={node==(void*)2?128UL:node==(void*)3?98344UL:4UL,1,1};
  p->kind=node==(void*)2?cudaMemcpyHostToDevice:node==(void*)3?cudaMemcpyDeviceToHost:cudaMemcpyDeviceToDevice;
  if(unsupported_copy && node==(void*)2)p->srcArray=(void*)1;
  return cudaSuccess;
}
cudaError_t cudaEventCreate(cudaEvent_t* event){*event=(void*)1;return cudaSuccess;}
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){*ms=1.25F;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t){return cudaSuccess;}
'''
        main = r'''
int main(){
  setenv("RILEY_OWNED_GRAPH_PROFILE","1",1);
  auto id=riley_diag_inventory(nullptr);
  for(int fail=0;fail<2;++fail){
    RileyOwnedGraphDiagnostic diag(true,nullptr);diag.before_launch();diag.after_launch();diag.before_wait();
    diag.finish(id,fail+1,127+fail,98344,cudaSuccess,fail?cudaErrorUnknown:cudaSuccess);
  }
  unsupported_copy=true;riley_diag_inventory(nullptr);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "diagnostic.cpp"
            binary = Path(directory) / "diagnostic"
            source.write_text(stub + profiler.HELPERS + main)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            records = [json.loads(line) for line in run.stderr.splitlines()]
        self.assertEqual(records[0]["nodes"], 4)
        self.assertEqual(records[0]["kernel_nodes"], 1)
        self.assertEqual(records[0]["memcpy_nodes"], 3)
        self.assertEqual(records[0]["h2d_bytes"], 128)
        self.assertEqual(records[0]["d2h_bytes"], 98344)
        self.assertEqual(records[0]["d2d_bytes"], 4)
        self.assertEqual(records[1]["cuda_graph_span_ns"], 1_250_000)
        self.assertIsNone(records[2]["cuda_graph_span_ns"])
        self.assertEqual(records[2]["completion_status"], 999)
        self.assertEqual(records[3]["memcpy_unknown_nodes"], 1)
        self.assertFalse(records[3]["memcpy_bytes_complete"])
        self.assertIsNone(records[3]["h2d_bytes"])
        self.assertIsNone(records[3]["d2h_bytes"])

    def test_summary_preserves_unmeasured_spans_and_phase_boundary(self) -> None:
        records = []
        for position, span in [(0, None), (127, 120.5), (128, 30.0), (158, 32.0)]:
            records.append({"schema": profiler.SCHEMA, "kind": "replay", "capture_id": 1,
                "replay_id": position + 1, "position": position, "event_setup_ns": 10,
                "host_staging_ns": 1, "host_launch_ns": 2, "host_wait_ns": 3,
                "cuda_graph_span_ns": span, "launch_status": 0, "completion_status": 0})
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "stderr.log"
            log.write_text("unrelated compiler/server log\n" + "\n".join(json.dumps(r) for r in records))
            result = profiler.summarize(log, 128)
            self.assertEqual(result["groups"]["prefill"]["replays"], 2)
            self.assertEqual(result["groups"]["decode"]["replays"], 2)
            gpu = result["groups"]["prefill"]["metrics"]["cuda_graph_span_ns"]
            self.assertEqual(gpu["unmeasured_count"], 1)
            self.assertEqual(gpu["median"], 120.5)
            log.write_text('{"schema":"' + profiler.SCHEMA + '"')
            with self.assertRaisesRegex(ValueError, "malformed diagnostic"):
                profiler.summarize(log, 128)

    def test_optional_projection_patch_is_bounded_and_prepares_before_capture(self) -> None:
        for original in [legacy_projection_fixture(), (profiler.REPOSITORY_ROOT / profiler.SOURCE).read_text()]:
            normal = profiler.instrument_text(original)
            patched = profiler.instrument_text(original, projection_events=True)
            self.assertNotIn("struct RileyProjectionDiagnostic", normal)
            self.assertEqual(original.count("cudaStreamSynchronize("), patched.count("cudaStreamSynchronize("))
            self.assertNotIn("cudaEventSynchronize(", patched)
            self.assertIn("cudaEventRecordWithFlags(events[slot][variant][edge],stream,cudaEventRecordExternal)", patched)
            prepared = patched.index("CurrentContext diagnostic_scope(r->owner)")
            captured = patched.index("record_reserved_sequence(r,staging,transfer,", prepared)
            self.assertLess(prepared, captured)
            self.assertIn("timed(0,[&]() noexcept {return enqueue_canonical_gemm", patched)
            self.assertIn("timed(1,[&]() noexcept {return kernel(enqueue_compiled_prefill_gemm", patched)
            self.assertIn("|| (*resources)->diagnostic_projection.has_events()) {", patched)
            self.assertIn("(*resources)->completion_unknown=true;", patched)
            self.assertLess(patched.index("diagnostic.before_wait();"), patched.index("r->diagnostic_projection.finish("))
            if "prefill_exec" in original:
                self.assertIn("diagnostic_projection.prepare(2)", patched)
                self.assertIn("diagnostic_projection.begin_capture(prefill?0:1)", patched)
            else:
                self.assertIn("diagnostic_projection.prepare(1)", patched)

    def test_projection_helper_event_lifecycle_selection_and_failures(self) -> None:
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
bool capturing=false;int creates=0,destroys=0,records=0,elapsed=0,fail_create_at=0;
bool riley_diag_enabled(){return true;}
cudaError_t cudaEventCreate(cudaEvent_t* event){assert(!capturing);++creates;if(creates==fail_create_at)return cudaErrorUnknown;*event=reinterpret_cast<void*>(static_cast<uintptr_t>(creates));return cudaSuccess;}
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,cudaStream_t,unsigned int flags){assert(capturing&&event!=nullptr&&flags==cudaEventRecordExternal);++records;return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){assert(!capturing);++elapsed;*ms=.005F;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t event){assert(!capturing&&event!=nullptr);++destroys;return cudaSuccess;}
'''
        main = r'''
int main(){
  unsetenv("RILEY_OWNED_GRAPH_PROJECTION");
  RileyProjectionDiagnostic default_off;assert(default_off.prepare(2)==cudaSuccess);
  assert(!default_off.has_events() && creates==0);
  setenv("RILEY_OWNED_GRAPH_PROJECTION","off",1);
  RileyProjectionDiagnostic explicit_off;assert(explicit_off.prepare(2)==cudaSuccess);
  assert(!explicit_off.has_events() && creates==0);
  setenv("RILEY_OWNED_GRAPH_PROJECTION","gate",1);
  RileyProjectionDiagnostic diag;assert(diag.prepare(2)==cudaSuccess);assert(creates==8);
  for(uint32_t slot=0;slot<2;++slot){
    diag.begin_capture(slot);capturing=true;
    diag.record(1,4,slot==0?1:0,0,nullptr); // A later layer is unsampled.
    diag.record(0,0,slot==0?1:0,0,nullptr); // A different projection is unsampled.
    diag.record(0,4,slot==0?1:0,0,nullptr);
    diag.record(0,4,slot==0?1:0,1,nullptr);
    capturing=false;diag.bind_capture(11+slot);
  }
  assert(records==4);
  diag.finish(11,1,127,cudaSuccess,cudaSuccess);
  diag.finish(12,2,128,cudaSuccess,cudaSuccess);
  diag.finish(12,3,129,cudaSuccess,cudaErrorUnknown);
  diag.finish(99,4,130,cudaSuccess,cudaSuccess); // No other capture's events.
  assert(elapsed==2);assert(diag.close()==cudaSuccess);assert(destroys==8);
  assert(!diag.has_events());assert(diag.close()==cudaSuccess);assert(destroys==8);
  fail_create_at=creates+3;
  RileyProjectionDiagnostic partial;assert(partial.prepare(1)==cudaErrorUnknown);
  assert(partial.has_events());assert(partial.close()==cudaSuccess);assert(destroys==10);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "projection.cpp"
            binary = Path(directory) / "projection"
            source.write_text(stub + profiler.PROJECTION_HELPERS + main)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            records = [json.loads(line) for line in run.stderr.splitlines()]
            self.assertEqual([(r["capture_id"], r["operator"]) for r in records], [(11, "override"), (12, "canonical"), (12, "canonical")])
            self.assertEqual(records[0]["projection"], "gate")
            self.assertEqual(records[0]["cuda_span_ns"], 5000)
            self.assertIsNone(records[2]["cuda_span_ns"])
            log = Path(directory) / "projection.log"
            replay = {"schema": profiler.SCHEMA, "kind": "replay", "capture_id": 11, "position": 127,
                "event_setup_ns": 1, "host_staging_ns": 1, "host_launch_ns": 1, "host_wait_ns": 1,
                "cuda_graph_span_ns": 10, "launch_status": 0, "completion_status": 0}
            log.write_text(json.dumps(replay) + "\n" + run.stderr)
            summary = profiler.summarize(log, 128)
            self.assertEqual(len(summary["projection_groups"]), 2)
            self.assertEqual(summary["projection_groups"][1]["unmeasured_or_failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
