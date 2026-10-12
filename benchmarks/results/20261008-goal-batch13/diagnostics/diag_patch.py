import pathlib
p=pathlib.Path("kernels/src/graph_resources.cu");s=p.read_text()
old="""  auto launched = cudaGraphLaunch(selected_exec, r->stream->stream);
  // Synchronize even on launch error; that is the only authority to release
  // possibly submitted work. A failed synchronization intentionally pins owners.
  auto completed = cudaStreamSynchronize(r->stream->stream);
"""
new="""  static uint64_t d_launch=0,d_sync=0,d_n=0,d_pre=0;
  auto d_t0=std::chrono::steady_clock::now();
  auto launched = cudaGraphLaunch(selected_exec, r->stream->stream);
  auto d_t1=std::chrono::steady_clock::now();
  auto completed = cudaStreamSynchronize(r->stream->stream);
  auto d_t2=std::chrono::steady_clock::now();
  d_launch+=std::chrono::duration_cast<std::chrono::nanoseconds>(d_t1-d_t0).count();d_sync+=std::chrono::duration_cast<std::chrono::nanoseconds>(d_t2-d_t1).count();
  d_pre+=std::chrono::duration_cast<std::chrono::nanoseconds>(d_t0-d_enter).count();
  if(++d_n%512==0){std::fprintf(stderr,"DIAGFFI n=512 pre_us=%.1f launch_us=%.1f sync_us=%.1f\\n",d_pre/512e3,d_launch/512e3,d_sync/512e3);d_launch=d_sync=d_pre=0;}
"""
assert old in s;s=s.replace(old,new)
old2="""    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  // Any attempted replay invalidates the previous result, even preflight errors."""
assert s.count(old2)==1
s=s.replace(old2,old2.replace("  clear_error(error);","  auto d_enter=std::chrono::steady_clock::now();\n  clear_error(error);"))
s="#include <chrono>\n#include <cstdio>\n"+s
p.write_text(s)
p=pathlib.Path("crates/riley-runtime/src/llama/variable_session.rs");s=p.read_text()
old="""        if self.graph.replay_transfer(&self.input).and_then(|()|self.graph.read_transfer(&mut self.output[..output_bytes])).is_err() {"""
new="""        static FFI:AtomicU64=AtomicU64::new(0);static N:AtomicU64=AtomicU64::new(0);static LAST:AtomicU64=AtomicU64::new(0);static T0:std::sync::OnceLock<std::time::Instant>=std::sync::OnceLock::new();
        let t0=*T0.get_or_init(std::time::Instant::now);let begin=std::time::Instant::now();
        let outcome=self.graph.replay_transfer(&self.input);
        let end=std::time::Instant::now();FFI.fetch_add((end-begin).as_nanos() as u64,Ordering::Relaxed);
        let n=N.fetch_add(1,Ordering::Relaxed)+1;
        if n%512==0{let now=(end-t0).as_nanos() as u64;let prev=LAST.swap(now,Ordering::Relaxed);let ffi=FFI.swap(0,Ordering::Relaxed);eprintln!("DIAG iters=512 wall_us_per_iter={:.1} ffi_us_per_iter={:.1} host_us_per_iter={:.1}",(now-prev) as f64/512e3,ffi as f64/512e3,(now-prev-ffi) as f64/512e3);}
        if outcome.and_then(|()|self.graph.read_transfer(&mut self.output[..output_bytes])).is_err() {"""
assert old in s;p.write_text(s.replace(old,new))
