from pathlib import Path
r=Path('/tmp/riley-g04-native-profile-source-260911')
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('native-trace-precise.bin','native-trace-tail.bin')
base=256
points=[]
def add(after,points_here):
 global s,base
 assert s.count(after)==1,(after,s.count(after))
 code=''
 for name,slot,size in points_here:
  code+=f'\n      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&l==26)s=copy(host+{base},d[{slot}]->device_data,{size},cudaMemcpyDeviceToHost);'
  points.append(dict(name=name,offset=base-256,bytes=size));base+=size
 s=s.replace(after,after+code)
add('      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(3,3);',[('old_res',1,1152),('o_proj',3,1152)])
add('      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm(r->stream->stream,d[3]->device_data,d[1]->device_data,weights[5]->device_data,d[11]->device_data,d[2]->device_data,1));',[('post_norm',2,1152),('float_res',11,2304)])
add('      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(5,10);',[('gate',9,3072),('up',10,3072)])
add('      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_swiglu(r->stream->stream,d[9]->device_data,d[10]->device_data,d[12]->device_data));',[('product',12,3072)])
add('      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(6,5);',[('down',5,1152)])
add('      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm(r->stream->stream,d[5]->device_data,d[11]->device_data,(l+1<layers?w[3+9*(l+1)]:w[1])->device_data,d[1]->device_data,d[2]->device_data,2));',[('next_norm',2,1152),('next_res',1,1152)])
p.write_text(s)
import json
Path('/tmp/riley-g04-native-profile-260911/tail-layout.json').write_text(json.dumps(points,indent=2)+'\n')
