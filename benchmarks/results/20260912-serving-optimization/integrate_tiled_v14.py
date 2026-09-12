from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
def edit(n,a,b):
 p=r/n;s=p.read_text();assert a in s,(n,a);p.write_text(s.replace(a,b))
edit('kernels/src/prefill_shape_model.cuh','#include "decode_shape.cuh"','#include "decode_tiled.cuh"')
edit('kernels/src/prefill_shape_model.cuh','uint32_t capacity,uint32_t physical){','uint32_t capacity,uint32_t physical,bool tiled=false){')
for offset,N,K,I in [(6,1536,576,0),(7,1536,576,0),(8,576,1536,320)]:
 n='kernels/src/prefill_shape_model.cuh';p=r/n;s=p.read_text();line=next(l for l in s.splitlines() if 'if(capacity==1)enqueue_decode_projection' in l and f'w(base+{offset})' in l)
 tiled=line.replace('if(capacity==1)','if(capacity==1&&tiled)').replace('enqueue_decode_projection','enqueue_tile_projection')
 s=s.replace(line,tiled+'\n  else '+line.strip());p.write_text(s)
edit('kernels/src/graph_numerics_precise.cu','uint32_t rows,uint32_t physical) noexcept {','uint32_t rows,uint32_t physical,bool tiled) noexcept {')
edit('kernels/src/graph_numerics_precise.cu','publish,rows,physical);','publish,rows,physical,tiled);')
edit('kernels/src/ffi_internal.hpp','uint32_t*,uint32_t*,uint32_t,uint32_t) noexcept;\n', 'uint32_t*,uint32_t*,uint32_t,uint32_t) noexcept;\n') # targeted below
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();line=next(l for l in s.splitlines() if 'enqueue_compiled_v3_prefill_model(' in l);s=s.replace(line,line.replace('uint32_t,uint32_t) noexcept','uint32_t,uint32_t,bool) noexcept'));p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();start=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(');a=s[start:];a=a.replace('weight_count!=273','(weight_count!=273&&weight_count!=363)').replace('i<273;++i){if(!w[i]','i<weight_count;++i){if(!w[i]')
a=a.replace('RileyCudaCanonicalGemmBf16GraphState state{};', '''if(weight_count==363)for(size_t i=273;i<363;++i){
  if(w[i]->byte_len!=1769472)return reject(error,"V3 tiled weight extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(w[i]==w[j])return reject(error,"V3 tiled weight alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 RileyCudaCanonicalGemmBf16GraphState state{};''')
a=a.replace('auto record_shape=[&](uint32_t row_capacity) noexcept {return','auto record_shape=[&](uint32_t row_capacity) noexcept {\n  if(row_capacity==1&&weight_count==363)for(size_t l=0;l<30;++l)for(size_t j=0;j<3;++j)weights[3+l*9+6+j]=w[273+l*3+j]->device_data;\n  return')
a=a.replace('row_capacity,physical),error','row_capacity,physical,weight_count==363&&row_capacity==1),error');s=s[:start]+a;p.write_text(s)
edit('crates/riley-cuda/src/ffi.rs','devices.len()!=22||weights.len()!=273','devices.len()!=22||(weights.len()!=273&&weights.len()!=363)')
edit('crates/riley-cuda/src/ffi.rs','raw.as_ptr(),weights.as_ptr(),273,head.as_ptr()','raw.as_ptr(),weights.as_ptr(),weights.len() as u64,head.as_ptr()')
edit('crates/riley-runtime/src/llama/variable_session.rs','pub(crate) devices:Vec<riley_cuda::CudaDeviceBuffer>,','pub(crate) devices:Vec<riley_cuda::CudaDeviceBuffer>,\n    pub(crate) tiled:Vec<riley_cuda::CudaDeviceBuffer>,')
edit('crates/riley-runtime/src/llama/variable_session.rs','Ok(Self{devices,staging:', 'let tiled=(0..90).map(|_|context.allocate_device_buffer(1769472)).collect::<riley_cuda::CudaResult<Vec<_>>>()?;\n        Ok(Self{devices,tiled,staging:')
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();s=s.replace('for source in [include_bytes!("../../../../kernels/src/decode_shape.cuh").as_slice(),','for source in [include_bytes!("../../../../kernels/src/decode_tiled.cuh").as_slice(),\n            include_bytes!("variable_session.rs").as_slice(),\n            include_bytes!("graph_decode_full.rs").as_slice(),\n            include_bytes!("../../../../kernels/src/decode_shape.cuh").as_slice(),')
a='        let mut devices:Vec<_>=f.weights.borrow_graph_weight_parents().collect();let base=devices.len();'
b='''        // Pack immutable gate/up/down weights once before capture. The buffers
        // remain explicit reservation parents and are dropped after graph close.
        for (index,buffer) in f.weights.borrow_graph_weight_parents().enumerate() {
            for layer in 0..30 { for part in 0..3 {
                if weights[3+layer*9+6+part]!=index {continue;}
                let (n,k)=if part==2 {(576usize,1536usize)}else{(1536usize,576usize)};
                let mut source=vec![0u8;n*k*2];
                for (i,bytes) in source.chunks_mut(chunk.len()).enumerate() {
                    buffer.download_to_slice((i*chunk.len()) as u64,bytes,&mut f.io_staging,stream).map_err(cuda)?;
                }
                let mut packed=vec![0u8;source.len()];
                for row in (0..n).step_by(8) {for depth in (0..k).step_by(16) {for lane in 0..32 {for hi in 0..2 {
                    let from=((row+lane/4)*k+depth+2*(lane%4)+hi*8)*2;
                    let to=(((row/8)*(k/16)+depth/16)*128+hi*64+lane*2)*2;
                    packed[to..to+4].copy_from_slice(&source[from..from+4]);
                }}}}
                hash.update(&packed);
                for (i,bytes) in packed.chunks(chunk.len()).enumerate() {
                    scratch.tiled[layer*3+part].upload_from_slice((i*chunk.len()) as u64,bytes,&mut f.io_staging,stream).map_err(cuda)?;
                }
            }}
        }
'''+a
assert a in s;s=s.replace(a,b);s=s.replace('        devices.extend(tail);\n        let mut graph=', '        devices.extend(tail);\n        weights.extend(devices.len()..devices.len()+scratch.tiled.len());\n        devices.extend(scratch.tiled.iter_mut());\n        let mut graph=');p.write_text(s)
print('Integrated tiled parents, capture selection and kernels')
