#!/usr/bin/env python3
"""Independent-row precise RoPE/KV and packed SwiGLU feasibility, never serving qualification.

Prepare/build/run are separate. Only run initializes CUDA. Accepted Batch7's
entire precise TU is the unchanged M1 oracle; candidates preserve its arithmetic
bodies. Synthetic seeds0/15/29 are not checkpoint-layer activations.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess

HERE=Path(__file__).resolve().parent
SOURCE='kernels/src/graph_numerics_precise.cu'
SOURCE_SHA='fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f'
BUILD_SHA='4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395'
COMMIT='1a2be0df01fe49daa4d4db155ad5c44f34ead6df'
CONVENTIONS_SHA='e306157697d5584b1f5d59826326f1682e6be57c224108fed68e8a2c3146e3d9'
CONTRACT_SHA='5bed6d70342dec4f30c85ba67dd41cf560e6b7f4b0de5dedfe59424cecc9b641'
SUPPORT_SHA='fd6e20e76c8ea27c419b18fcc4c9d7abecfe6a5fb5e8a2a776f7c458b7c3e007'
SUPPORT_NATIVE_SHA='11eff6609013fdb114439ad00c575d92b74ad705d23f4265a7ba3b41a5303703'
DRIVER_SHA='266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4'
UUID='9087e4256acab722b8c9cc0423b39fb0'
SCHEMA='riley.multisequence-precise'
SEEDS=(0,15,29)
MODES=((1,1),(2,2),(4,3),(4,4))
PATTERNS=('bounded_fingerprint','signed_zero_impulses','tail_cancellation')
MAPPINGS=('identity64','reverse64','affine7_3_mod64')
COUNTS={'cases':3456,'synthetic_seeds':3,'position_tuples':32,'patterns':3,'mappings':3,'modes':4}
FLAGS=('exact_outputs','finite_outputs','guards_intact','inputs_unchanged','oracle_outputs_unchanged',
       'other_kv_slots_unchanged','inactive_outputs_zero','mapping_invariant','all_allocations_freed')
ROPE_SIGNATURE='''__global__ void compiled_packed_decode_rope_kv(
    const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,
    const float* cos,const float* sin,const uint32_t* metadata){
'''
ROPE_NEW='''__global__ void rope_rows(const __nv_bfloat16* packed,
    __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,
    const float* cos,const float* sin,const uint32_t* packet){
 const uint32_t row=blockIdx.y,active_rows=packet[6];
 qo+=row*576;
 if(row>=active_rows){
  for(int i=threadIdx.x+blockIdx.x*blockDim.x;i<576;i+=blockDim.x*gridDim.x)qo[i]=__ushort_as_bfloat16(0);
  return;
 }
 const __nv_bfloat16* q=packed+row*960;const __nv_bfloat16* k=q+576;const __nv_bfloat16* v=q+768;
 const uint32_t* metadata=packet+32+row*32;
'''
SWIGLU_SIGNATURE='__global__ void compiled_swiglu(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out){\n'
SWIGLU_NEW='''__global__ void swiglu_rows(const __nv_bfloat16* packed,__nv_bfloat16* out,const uint32_t* packet){
 const uint32_t row=blockIdx.y,active_rows=packet[6];
 out+=row*1536;
 if(row>=active_rows){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536)out[i]=__ushort_as_bfloat16(0);return;}
 const __nv_bfloat16* g=packed+row*3072;const __nv_bfloat16* u=g+1536;
'''
WRAPPERS='''cudaError_t enqueue_rope_rows(cudaStream_t stream,const void* packed,void* qo,void* keys,void* values,const void* cos,const void* sin,const void* packet,uint32_t bucket) noexcept {
 if(!packed||!qo||!keys||!values||!cos||!sin||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 rope_rows<<<dim3(2,bucket),256,0,stream>>>((const __nv_bfloat16*)packed,(__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const float*)cos,(const float*)sin,(const uint32_t*)packet);
 return cudaGetLastError();
}
cudaError_t enqueue_swiglu_rows(cudaStream_t stream,const void* packed,void* out,const void* packet,uint32_t bucket) noexcept {
 if(!packed||!out||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 swiglu_rows<<<dim3(6,bucket),256,0,stream>>>((const __nv_bfloat16*)packed,(__nv_bfloat16*)out,(const uint32_t*)packet);
 return cudaGetLastError();
}
'''


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def evidence(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path)}


def unique(pairs):
    out={}
    for key,value in pairs:
        require(key not in out,'duplicate JSON key');out[key]=value
    return out


def read(path):
    def invalid(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def check(ref):
    require(evidence(ref['path'])==ref,'evidence changed: '+ref['path'])


def load(path,digest,name):
    require(sha(path)==digest,'frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def transform(raw):
    require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'unknown accepted precise source')
    source=raw.decode();chunks=[];lineage={}
    for name,old,new in [('rope',ROPE_SIGNATURE,ROPE_NEW),('swiglu',SWIGLU_SIGNATURE,SWIGLU_NEW)]:
        require(source.count(old)==1,'precise kernel anchor differs')
        start=source.index(old);end=source.index('\n}',start)+2
        original=source[start:end]+'\n'
        changed=original.replace(old,new,1)
        require(changed.replace(new,old,1)==original,'accepted arithmetic body changed')
        chunks.append(changed);lineage[name+'_kernel_sha256']=hashlib.sha256(original.encode()).hexdigest()
    result='#include "ffi_internal.hpp"\n#include <cuda_bf16.h>\nnamespace riley_multisequence_precise_probe {\n'+''.join(chunks)+WRAPPERS+'}\n'
    return result.encode(),dict(lineage,arithmetic_body_preserved_by_reverse_transform=True,
             active_rows_from_fresh_device_header_byte24=True,host_scalar_active_rows=False)


def fixture_support(raw):
    require(hashlib.sha256(raw).hexdigest()==SUPPORT_NATIVE_SHA,'unfrozen attention fixture source')
    text=raw.decode();start=text.index('uint16_t word(');end=text.index('bool run_case(')
    result='// Strictly extracted shared CPU fixture/guard helpers; no main or CUDA kernels.\nnamespace {\n'+text[start:end]+'}\n'
    require('int main(' not in result and 'enqueue_' not in result and '__global__' not in result,'support extraction includes executable kernels/main')
    return result.encode()


def case_rows():
    for i in range(COUNTS['cases']):
        bucket,active=MODES[(i//3)%4];position=(i//36)%32
        yield {'case_id':i,'synthetic_seed':SEEDS[i//1152],'position_tuple':position,'pattern':PATTERNS[(i//12)%3],
               'mapping':MAPPINGS[i%3],'bucket':bucket,'active_rows':active,
               'positions':[128+(position+11*r)%32 for r in range(active)],
               'numerical_only_position159':any(128+(position+11*r)%32==159 for r in range(active))}


def case_index():
    return ''.join('\t'.join(map(str,[i,SEEDS[i//1152],(i//36)%32,(i//12)%3,(i//3)%4,i%3]))+'\n' for i in range(COUNTS['cases']))


def recipe():
    return {'synthetic_seeds':list(SEEDS),'checkpoint_activations':False,'shared_descriptor_bytes':1280,
            'row_view_offset':128,'row_stride':128,'active_count_device_offset':24,
            'packed_qkv_bf16_stride':960,'qkv_offsets':[0,576,768],'q_output_bf16_stride':576,
            'packed_gate_up_bf16_stride':3072,'gate_up_offsets':[0,1536],'product_bf16_stride':1536,
            'guard_bytes_each_side':256,'full_kv_bytes_each':393216,
            'normal_allocations':sum(11+8*r['active_rows'] for r in case_rows()),
            'graph_allocations':43,'graph_captures':1,'graph_replays':3,'graph_active_rows':[3,4,3],
            'graph_parameter_updates':0,'host_scalar_active_rows':False,
            'output_comparison':'all active Q/product bytes plus whole K/V pools; inactive Q/product are positive zero',
            'input_patterns':'shared bounded/signed-zero/cancellation BF16 recipe plus bounded packed GU and exact FP32 table bit fixtures',
            'fixture_authority':'host-owned synthetic rows and full disjoint ten-block reservations; not production admission',
            'inactive_poison':'row metadata and packed QKV/GU NaNs after canonical host validation; common header remains fresh',
            'position159_scope':'numerical-only; public P128/O32 final input remains158'}


def source_build(path,conventions):
    require(sha(path)==BUILD_SHA,'requires accepted Batch7 build receipt')
    value=conventions.check_snapshot(path)
    require(value['source_commit']==COMMIT and value['source_files'][SOURCE]==SOURCE_SHA,'accepted precise identity differs')
    return value


def prepare(build_path,conventions_path,support_path,contract_path,output):
    conventions=load(conventions_path,CONVENTIONS_SHA,'precise_compile_conventions')
    load(support_path,SUPPORT_SHA,'precise_shared_attention_driver')
    native=Path(support_path).with_name('native.cu');support_raw=fixture_support(native.read_bytes())
    require(sha(contract_path)==CONTRACT_SHA,'descriptor contract differs')
    build=source_build(build_path,conventions);root=Path(build['source_root']);output=Path(output).resolve()
    require(not output.is_relative_to(root),'output must be outside accepted snapshot')
    candidate,lineage=transform((root/SOURCE).read_bytes());output.mkdir(mode=0o700)
    for name,raw in [('candidate.cu',candidate),('fixture_support.hpp',support_raw),('ffi_internal.hpp',(root/'kernels/src/ffi_internal.hpp').read_bytes()),('cases.tsv',case_index().encode())]:
        with (output/name).open('xb') as stream:stream.write(raw)
    result={'schema_version':SCHEMA+'-fixtures.v1',**COUNTS,'recipe':recipe(),
            **{key:build[key] for key in ('source_root','source_commit','source_files','binaries')},
            'source_build':evidence(build_path),'runner':evidence(__file__),'host_source':evidence(HERE/'native.cu'),
            'conventions':evidence(conventions_path),'support_driver':evidence(support_path),'support_native':evidence(native),
            'descriptor_contract':evidence(contract_path),'lineage':lineage,
            'generated':{name:evidence(output/name) for name in ('candidate.cu','fixture_support.hpp','ffi_internal.hpp','cases.tsv')},
            'gpu_executed':False,'performance_claim':False,'production_integration':False}
    write(output/'fixtures.json',result);return validate_manifest(output/'fixtures.json')


def validate_manifest(path):
    path=Path(path).resolve(strict=True);value=read(path)
    require(value['schema_version']==SCHEMA+'-fixtures.v1' and all(value[k]==v for k,v in COUNTS.items()),'fixture coverage/schema')
    for key in ('source_build','runner','host_source','conventions','support_driver','support_native','descriptor_contract'):check(value[key])
    require(value['runner']==evidence(__file__) and value['host_source']==evidence(HERE/'native.cu'),'current helper differs')
    require(value['support_driver']['sha256']==SUPPORT_SHA and value['support_native']['sha256']==SUPPORT_NATIVE_SHA
            and value['descriptor_contract']['sha256']==CONTRACT_SHA,'shared dependency differs')
    build=source_build(value['source_build']['path'],load(value['conventions']['path'],CONVENTIONS_SHA,'precise_compile_conventions'))
    require(all(value[k]==build[k] for k in ('source_root','source_commit','source_files','binaries')),'source/binary identity')
    candidate,lineage=transform((Path(build['source_root'])/SOURCE).read_bytes())
    expected={'candidate.cu':candidate,'fixture_support.hpp':fixture_support(Path(value['support_native']['path']).read_bytes()),
              'ffi_internal.hpp':(Path(build['source_root'])/'kernels/src/ffi_internal.hpp').read_bytes(),'cases.tsv':case_index().encode()}
    require(set(value['generated'])==set(expected),'generated inventory')
    for name,raw in expected.items():
        require(value['generated'][name]==evidence(path.parent/name) and (path.parent/name).read_bytes()==raw,'generated file differs')
    require(value['lineage']==lineage and value['recipe']==recipe(),'fixture recipe/lineage differs')
    require(value['gpu_executed'] is False and value['performance_claim'] is False and value['production_integration'] is False,'fixture scope')
    return value


def commands(manifest,nvcc,options,directory,runtime):
    objects=[directory/name for name in ('oracle.o','candidate.o','host.o')]
    flags=options['flags'];arch=[flag for flag in flags if 'arch=compute_89' in flag]
    return [[str(nvcc),*flags,'-x','cu','-c',str(Path(manifest['source_root'])/SOURCE),'-o',str(objects[0])],
            [str(nvcc),*flags,'-x','cu','-c',manifest['generated']['candidate.cu']['path'],'-o',str(objects[1])],
            [str(nvcc),'-std=c++17','-O2',*arch,'-I'+str(directory.parent),'-c',manifest['host_source']['path'],'-o',str(objects[2])],
            [str(nvcc),'--cudart=shared',*map(str,objects),'-L'+str(runtime.parent),'-Xlinker','-rpath','-Xlinker',str(runtime.parent),'-o',str(directory/'probe')]]


def build_probe(path,nvcc):
    path=Path(path).resolve(strict=True);manifest=validate_manifest(path);nvcc=Path(nvcc).resolve(strict=True)
    conventions=load(manifest['conventions']['path'],CONVENTIONS_SHA,'precise_compile_conventions')
    options=conventions.production_flags(manifest,nvcc);runtime=Path(options['runtime_library_link'])
    version=subprocess.check_output([str(nvcc),'--version'],text=True);require(re.search(r'release 13\.',version),'production CUDA13 required')
    directory=path.parent/'build';directory.mkdir();argv=commands(manifest,nvcc,options,directory,runtime)
    with (directory/'compile.log').open('x') as log:
        for command in argv:subprocess.run(command,cwd=directory,stdout=log,stderr=log,check=True)
    require(validate_manifest(path)==manifest,'inputs changed during compile')
    for ref in options['cmake_artifacts']:check(ref)
    result={'schema_version':SCHEMA+'-build.v1','fixture_manifest':evidence(path),'compiler':evidence(nvcc),'compiler_version':version,
            'options':options,'commands':argv,'runtime':evidence(runtime),'runtime_link':str(runtime),
            'objects':[evidence(directory/name) for name in ('oracle.o','candidate.o','host.o')],
            'binary':evidence(directory/'probe'),'log':evidence(directory/'compile.log'),'gpu_executed':False,'performance_claim':False}
    write(path.parent/'compile.json',result);return result


def validate_compile(path,manifest_path):
    value=read(path);manifest=validate_manifest(manifest_path)
    require(value['schema_version']==SCHEMA+'-build.v1' and value['fixture_manifest']==evidence(manifest_path),'compile manifest binding')
    for key in ('compiler','runtime','binary','log'):check(value[key])
    for ref in value['objects']+value['options']['cmake_artifacts']:check(ref)
    options=load(manifest['conventions']['path'],CONVENTIONS_SHA,'precise_compile_conventions').production_flags(manifest,Path(value['compiler']['path']))
    directory=Path(path).parent/'build';runtime=Path(options['runtime_library_link'])
    require(options==value['options'] and value['runtime']==evidence(runtime) and value['runtime_link']==str(runtime),'precise flags/shared runtime differ')
    require(value['commands']==commands(manifest,Path(value['compiler']['path']),options,directory,runtime)
            and value['objects']==[evidence(directory/name) for name in ('oracle.o','candidate.o','host.o')]
            and value['binary']==evidence(directory/'probe') and value['log']==evidence(directory/'compile.log'),'compile command/objects differ')
    require(value['gpu_executed'] is False and value['performance_claim'] is False,'compile scope')
    return value


def transition_rows():
    for step,(seed,position,pattern,mode,mapping) in enumerate(((0,0,0,2,0),(15,17,1,3,2),(29,5,2,2,1))):
        bucket,active=MODES[mode];positions=[128+(position+11*r)%32 for r in range(active)]
        yield {'case_id':step,'synthetic_seed':seed,'position_tuple':position,'pattern':PATTERNS[pattern],
               'mapping':MAPPINGS[mapping],'bucket':bucket,'active_rows':active,'positions':positions,
               'numerical_only_position159':159 in positions,'step':step,'capture_count':1,'graph_nodes':2,
               'parameter_updates':0,'replay_index':step+1}


def validate_raw(path):
    def invalid(value):raise ValueError('nonfinite JSON: '+value)
    records=[json.loads(line,object_pairs_hook=unique,parse_constant=invalid) for line in Path(path).read_text().splitlines() if line]
    require(len(records)==COUNTS['cases']+5 and all(r['schema']==SCHEMA+'-native.v1' for r in records),'native record extent/schema')
    device,summary=records[0],records[-1]
    require(device['kind']=='device' and device['uuid_hex']==UUID and device['compute_major']==8
            and device['compute_minor']==9 and device['runtime_version']==13000,'GPU/runtime differs')
    def comparison(row,expected,transition):
        require(row['kind']==('graph_transition' if transition else 'case')
                and all(row[k]==v for k,v in expected.items()),'case identity/coverage differs')
        active,bucket=expected['active_rows'],expected['bucket'];widths=[active*576,active*1536,196608,196608]
        require(row['candidate_invocations']==2 and row['oracle_invocations']==2*active
                and [row[k] for k in ('q_words_compared','product_words_compared','key_words_compared','value_words_compared')]==widths
                and row['inactive_words_checked']==(bucket-active)*2112,'partial operation/output comparison')
        require(row['allocations_created']==(43 if transition else 11+8*active)
                and row['mapping_invariant_checked'] is (not transition),'allocation/mapping scope')
        require(all(type(row[k]) is bool for k in FLAGS) and type(row['passed']) is bool,'comparison flags')
        counts=row['mismatch_by_region']
        require(type(counts) is list and len(counts)==4 and all(type(n) is int and 0<=n<=width for n,width in zip(counts,widths))
                and type(row['mismatch_words']) is int and row['mismatch_words']==sum(counts),'mismatch accounting')
        require(row['exact_outputs']==(sum(counts)==0) and row['passed']==all(row[k] for k in FLAGS),'comparison success reconciliation')
        require(('first_mismatch' in row)==(sum(counts)>0),'missing/spurious first mismatch')
        if sum(counts):
            first=row['first_mismatch'];region=first['region']
            require(type(region) is int and 0<=region<4 and counts[region]>0
                    and type(first['word']) is int and 0<=first['word']<widths[region]
                    and all(type(first[k]) is int and 0<=first[k]<=65535 for k in ('oracle_bits','candidate_bits'))
                    and first['oracle_bits']!=first['candidate_bits'],'first mismatch differs')
        return not row['passed']
    failed=sum(comparison(row,expected,False) for row,expected in zip(records[1:1+COUNTS['cases']],case_rows()))
    transition_failed=sum(comparison(row,expected,True) for row,expected in zip(records[1+COUNTS['cases']:-1],transition_rows()))
    require(summary['kind']=='summary' and summary['completed'] is True and all(summary[k]==v for k,v in COUNTS.items()),'incomplete native summary')
    require(summary['failed_cases']==failed and summary['failed_graph_transitions']==transition_failed
            and summary['graph_transitions']==3 and summary['precise_bitwise_equal'] is (failed==transition_failed==0),'native summary reconciliation')
    require(all(summary[k]==1 for k in ('graphs_created','graphs_destroyed','execs_created','execs_destroyed')),'graph lifetime differs')
    require(summary['allocations_created']==summary['allocations_freed']==recipe()['normal_allocations']+recipe()['graph_allocations']
            and summary['live_bytes']==summary['cleanup_errors']==0 and summary['stream_destroyed'] is True,'native cleanup incomplete')
    require(summary['error']=='' and summary['performance_claim'] is False,'native error/scope')
    return device,summary


def run_probe(path,driver_directory,timeout):
    require(type(timeout) is int and timeout>0,'positive bounded timeout required')
    path=Path(path).resolve(strict=True);manifest=validate_manifest(path);compiled=validate_compile(path.parent/'compile.json',path)
    support=load(manifest['support_driver']['path'],SUPPORT_SHA,'precise_shared_attention_driver')
    driver_dir=Path(driver_directory).resolve(strict=True);driver=evidence(driver_dir/'libcuda.so.1')
    require(driver['sha256']==DRIVER_SHA and not os.environ.get('LD_PRELOAD'),'private driver/LD_PRELOAD differs')
    runtime=compiled['runtime'];env=dict(os.environ)
    env['LD_LIBRARY_PATH']=str(driver_dir)+':'+str(Path(compiled['runtime_link']).parent)
    command=[compiled['binary']['path'],'--cases',manifest['generated']['cases.tsv']['path'],'--device','0']
    raw,errors=path.parent/'native-results.jsonl',path.parent/'native-stderr.log'
    require(not any((path.parent/name).exists() for name in ('native-results.jsonl','native-stderr.log','process-exit.json','result.json')),'run artifacts exist; no retry in place')
    process=None;failure=None;cleanup_failure=None
    try:
        with raw.open('x') as stdout,errors.open('x') as stderr:
            process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            process.wait(timeout=timeout);require(process.returncode==0,'native probe incomplete')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)};raise
    finally:
        cleanup_exception=None
        try:
            if process is not None and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait(timeout=10)
        except BaseException as error:
            cleanup_failure={'type':type(error).__name__,'message':str(error)};cleanup_exception=error
        finally:
            write(path.parent/'process-exit.json',{'failure':failure,'cleanup_failure':cleanup_failure,
                  'returncode':process.returncode if process else None,'owned_process_reaped':process is not None and process.poll() is not None,
                  'process_scope':'single native executable; native source spawns no child processes'})
        if cleanup_exception is not None and failure is None:raise cleanup_exception
    require(validate_manifest(path)==manifest and validate_compile(path.parent/'compile.json',path)==compiled,'inputs changed while running')
    device,summary=validate_raw(raw)
    maps=support.validate_maps(device['runtime_maps'],driver,runtime)
    require(maps==support.validate_maps(summary['runtime_maps'],driver,runtime)
            and evidence(driver_dir/'libcuda.so.1')==driver,'actual native runtime changed')
    result={'schema_version':SCHEMA+'-result.v1','completed':True,'precise_bitwise_equal':summary['precise_bitwise_equal'],
            'fixture_manifest':evidence(path),'compile_receipt':evidence(path.parent/'compile.json'),
            'driver':driver,'runtime':runtime,'loaded_native_files':maps,'command':command,'ld_library_path':env['LD_LIBRARY_PATH'],
            'native_results':evidence(raw),'stderr':evidence(errors),'process_exit':evidence(path.parent/'process-exit.json'),
            'device':device,'summary':summary,'gpu_executed':True,'performance_claim':False,'production_integration':False}
    write(path.parent/'result.json',result);return validate_result(path.parent/'result.json')


def validate_result(path):
    path=Path(path).resolve(strict=True);result=read(path)
    require(result['schema_version']==SCHEMA+'-result.v1' and result['completed'] is True and result['gpu_executed'] is True
            and result['performance_claim'] is False and result['production_integration'] is False,'result completion/scope')
    for key in ('fixture_manifest','compile_receipt','driver','runtime','native_results','stderr','process_exit'):check(result[key])
    for key,name in (('fixture_manifest','fixtures.json'),('compile_receipt','compile.json'),('native_results','native-results.jsonl'),('stderr','native-stderr.log'),('process_exit','process-exit.json')):
        require(result[key]==evidence(path.parent/name),'result artifact path differs')
    manifest_path=Path(result['fixture_manifest']['path']);manifest=validate_manifest(manifest_path)
    compiled=validate_compile(result['compile_receipt']['path'],manifest_path)
    require(result['runtime']==compiled['runtime'] and result['driver']['sha256']==DRIVER_SHA,'result runtime binding')
    libraries=result['ld_library_path'].split(':')
    require(len(libraries)==2 and evidence(Path(libraries[0])/'libcuda.so.1')==result['driver']
            and libraries[1]==str(Path(compiled['runtime_link']).parent),'runtime search path differs')
    process=read(result['process_exit']['path'])
    require(process['failure'] is None and process['cleanup_failure'] is None and type(process['returncode']) is int
            and process['returncode']==0 and process['owned_process_reaped'] is True,'owned probe incomplete')
    device,summary=validate_raw(result['native_results']['path'])
    require(result['device']==device and result['summary']==summary and result['precise_bitwise_equal']==summary['precise_bitwise_equal'],'result/raw reconciliation')
    support=load(manifest['support_driver']['path'],SUPPORT_SHA,'precise_shared_attention_driver')
    maps=support.validate_maps(device['runtime_maps'],result['driver'],result['runtime'])
    require(maps==support.validate_maps(summary['runtime_maps'],result['driver'],result['runtime']) and result['loaded_native_files']==maps,'runtime maps differ')
    require(result['command']==[compiled['binary']['path'],'--cases',manifest['generated']['cases.tsv']['path'],'--device','0'],'result command differs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare_parser=sub.add_parser('prepare')
    for flag in ('build-receipt','conventions','support-driver','descriptor-contract','output'):prepare_parser.add_argument('--'+flag,type=Path,required=True)
    build_parser=sub.add_parser('build');build_parser.add_argument('--manifest',type=Path,required=True);build_parser.add_argument('--nvcc',type=Path,required=True)
    run_parser=sub.add_parser('run');run_parser.add_argument('--manifest',type=Path,required=True);run_parser.add_argument('--driver-library-dir',type=Path,required=True);run_parser.add_argument('--timeout',type=int,default=3600)
    validate_parser=sub.add_parser('validate');validate_parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    def interrupted(signum,_frame):raise InterruptedError('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.command=='prepare':result=prepare(args.build_receipt,args.conventions,args.support_driver,args.descriptor_contract,args.output)
    elif args.command=='build':result=build_probe(args.manifest,args.nvcc)
    elif args.command=='run':result=run_probe(args.manifest,args.driver_library_dir,args.timeout)
    else:result=validate_result(args.result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
