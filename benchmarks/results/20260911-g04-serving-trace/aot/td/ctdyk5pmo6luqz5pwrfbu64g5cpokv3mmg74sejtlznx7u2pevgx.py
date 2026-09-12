r"""
Compile-time auto-tuning block: 

import torch
from math import inf, nan
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /home/psyche/.cache/vllm/torch_compile_cache/torch_aot_compile/7cf101d73a71350b244cdb09374d2f9cefac6786ad2f9dd08ae0e454a22fb60e/inductor_cache/ai/cai6kiqp67mjii6dr23yficgojrq3fkq5dvrfcqfpcynrpkkc5dq.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[49152, 576][576, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 576][576, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[576][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-05), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_per_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_per_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 512, 'r0_': 590976}, 'kernel_num_gb': 0.000444032, 'kernel_flop': 0, 'backend_hash': '6B89DC6724FD69E85DEBF864D25FCBD574DAA4FCE3C5BD0184254249CB28DECC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 576
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tl.device_assert(((0 <= tmp1) & (tmp1 < 49152)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 49152")
    tmp3 = tl.load(in_ptr1 + (r0_1 + 576*tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 576.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (r0_1 + 576*x0), tmp3, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 576*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/psyche/.cache/vllm/torch_compile_cache/torch_aot_compile/7cf101d73a71350b244cdb09374d2f9cefac6786ad2f9dd08ae0e454a22fb60e/inductor_cache/wn/cwnw4jrflvinb4zrqtmvzkv5gjh6kpve2msv5vlny3px3bd26hjq.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': '6B89DC6724FD69E85DEBF864D25FCBD574DAA4FCE3C5BD0184254249CB28DECC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 3)
        x2 = xindex // 192
        x4 = xindex
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 8192)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 8192")
        tmp14 = tl.load(in_ptr2 + (64*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp7 * tmp14
        tmp16 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (32 + 64*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 - tmp18
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp6, tmp19, tmp20)
        tmp22 = tmp0 >= tmp5
        tmp23 = tl.full([1], 64, tl.int64)
        tmp24 = tmp0 < tmp23
        tmp25 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp26 < 0
        tmp30 = tl.where(tmp29, tmp28, tmp26)
        tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 8192)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 8192")
        tmp32 = tl.load(in_ptr2 + (64*tmp30 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp25 * tmp32
        tmp34 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (32 + 64*tmp30 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp33 + tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp22, tmp37, tmp38)
        tmp40 = tl.where(tmp6, tmp21, tmp39)
        tl.store(out_ptr0 + (x4), tmp40, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 64)
        x6 = ((xindex // 64) % 9)
        x7 = xindex // 576
        x9 = xindex
        tmp41 = (x5).to(tl.int32)
        tmp42 = tl.full([1], 0, tl.int64)
        tmp43 = tmp41 >= tmp42
        tmp44 = (x5).to(tl.int64)
        tmp45 = (tmp44).to(tl.int64)
        tmp46 = tl.full([1], 32, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr0 + (64*x6 + 960*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr1 + (x7), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp49 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp49)
        tl.device_assert(((0 <= tl.broadcast_to(tmp53, [XBLOCK])) & (tl.broadcast_to(tmp53, [XBLOCK]) < 8192)) | ~(tmp47 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp53, [XBLOCK]) < 8192")
        tmp55 = tl.load(in_ptr2 + (64*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp48 * tmp55
        tmp57 = tl.load(in_ptr0 + (32 + 64*x6 + 960*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr2 + (32 + 64*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 - tmp59
        tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
        tmp62 = tl.where(tmp47, tmp60, tmp61)
        tmp63 = tmp41 >= tmp46
        tmp64 = tl.full([1], 64, tl.int64)
        tmp65 = tmp41 < tmp64
        tmp66 = tl.load(in_ptr0 + (32 + 64*x6 + 960*x7 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x7), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp69 = tmp67 + tmp68
        tmp70 = tmp67 < 0
        tmp71 = tl.where(tmp70, tmp69, tmp67)
        tl.device_assert(((0 <= tl.broadcast_to(tmp71, [XBLOCK])) & (tl.broadcast_to(tmp71, [XBLOCK]) < 8192)) | ~(tmp63 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp71, [XBLOCK]) < 8192")
        tmp73 = tl.load(in_ptr2 + (64*tmp71 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp66 * tmp73
        tmp75 = tl.load(in_ptr0 + (64*x6 + 960*x7 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + (32 + 64*tmp71 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tmp75 * tmp76
        tmp78 = tmp74 + tmp77
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp63, tmp78, tmp79)
        tmp81 = tl.where(tmp47, tmp62, tmp80)
        tl.store(out_ptr1 + (x9), tmp81, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((128, 960), (960, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 9, 64), (576, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 24576, 73728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.00177152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    raw_stream0 = get_raw_stream(0)
raw_stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((128,), (1,), 'cuda:0', torch.int32, 0, (128,))
arg2_1 = generate_example_value((49152, 576), (576, 1), 'cuda:0', torch.bfloat16, 0, (49152, 576))
arg3_1 = generate_example_value((576,), (1,), 'cuda:0', torch.bfloat16, 0, (576,))
buf0 = generate_example_value((128, 576), (576, 1), 'cuda:0', torch.bfloat16, 0, (128, 576))
buf2 = generate_example_value((128, 576), (576, 1), 'cuda:0', torch.bfloat16, 0, (128, 576))
with torch.cuda._DeviceGuard(0):
    triton_per_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, 128, 576, stream=raw_stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf2

raw_stream0 = get_raw_stream(0)
buf3 = generate_example_value((128, 960), (960, 1), 'cuda:0', torch.bfloat16, 0, (128, 960))
arg5_1 = generate_example_value((128,), (1,), 'cuda:0', torch.int64, 0, (128,))
arg6_1 = generate_example_value((8192, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (8192, 64))
buf4 = generate_example_value((128, 3, 64), (192, 64, 1), 'cuda:0', torch.bfloat16, 0, (128, 3, 64))
buf5 = generate_example_value((128, 9, 64), (576, 64, 1), 'cuda:0', torch.bfloat16, 0, (128, 9, 64))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_1.run(buf3, arg5_1, arg6_1, buf4, buf5, 24576, 73728, stream=raw_stream0)
del buf3, arg5_1, arg6_1, buf4, buf5

"""
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/psyche/.cache/vllm/torch_compile_cache/torch_aot_compile/7cf101d73a71350b244cdb09374d2f9cefac6786ad2f9dd08ae0e454a22fb60e/inductor_cache/ai/cai6kiqp67mjii6dr23yficgojrq3fkq5dvrfcqfpcynrpkkc5dq.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[49152, 576][576, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 576][576, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[576][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-05), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 576][576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_per_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_per_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 512, 'r0_': 590976}, 'kernel_num_gb': 0.000444032, 'kernel_flop': 0, 'backend_hash': '6B89DC6724FD69E85DEBF864D25FCBD574DAA4FCE3C5BD0184254249CB28DECC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 576
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tl.device_assert(((0 <= tmp1) & (tmp1 < 49152)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 49152")
    tmp3 = tl.load(in_ptr1 + (r0_1 + 576*tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 576.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (r0_1 + 576*x0), tmp3, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 576*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/psyche/.cache/vllm/torch_compile_cache/torch_aot_compile/7cf101d73a71350b244cdb09374d2f9cefac6786ad2f9dd08ae0e454a22fb60e/inductor_cache/wn/cwnw4jrflvinb4zrqtmvzkv5gjh6kpve2msv5vlny3px3bd26hjq.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': '6B89DC6724FD69E85DEBF864D25FCBD574DAA4FCE3C5BD0184254249CB28DECC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 3)
        x2 = xindex // 192
        x4 = xindex
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 8192)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 8192")
        tmp14 = tl.load(in_ptr2 + (64*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp7 * tmp14
        tmp16 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (32 + 64*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 - tmp18
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp6, tmp19, tmp20)
        tmp22 = tmp0 >= tmp5
        tmp23 = tl.full([1], 64, tl.int64)
        tmp24 = tmp0 < tmp23
        tmp25 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp26 < 0
        tmp30 = tl.where(tmp29, tmp28, tmp26)
        tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 8192)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 8192")
        tmp32 = tl.load(in_ptr2 + (64*tmp30 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp25 * tmp32
        tmp34 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (32 + 64*tmp30 + ((-32) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp33 + tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp22, tmp37, tmp38)
        tmp40 = tl.where(tmp6, tmp21, tmp39)
        tl.store(out_ptr0 + (x4), tmp40, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 64)
        x6 = ((xindex // 64) % 9)
        x7 = xindex // 576
        x9 = xindex
        tmp41 = (x5).to(tl.int32)
        tmp42 = tl.full([1], 0, tl.int64)
        tmp43 = tmp41 >= tmp42
        tmp44 = (x5).to(tl.int64)
        tmp45 = (tmp44).to(tl.int64)
        tmp46 = tl.full([1], 32, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr0 + (64*x6 + 960*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr1 + (x7), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp49 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp49)
        tl.device_assert(((0 <= tl.broadcast_to(tmp53, [XBLOCK])) & (tl.broadcast_to(tmp53, [XBLOCK]) < 8192)) | ~(tmp47 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp53, [XBLOCK]) < 8192")
        tmp55 = tl.load(in_ptr2 + (64*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp48 * tmp55
        tmp57 = tl.load(in_ptr0 + (32 + 64*x6 + 960*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr2 + (32 + 64*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 - tmp59
        tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
        tmp62 = tl.where(tmp47, tmp60, tmp61)
        tmp63 = tmp41 >= tmp46
        tmp64 = tl.full([1], 64, tl.int64)
        tmp65 = tmp41 < tmp64
        tmp66 = tl.load(in_ptr0 + (32 + 64*x6 + 960*x7 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x7), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = (tl.full([XBLOCK], 8192, tl.int32)).to(tl.int32)
        tmp69 = tmp67 + tmp68
        tmp70 = tmp67 < 0
        tmp71 = tl.where(tmp70, tmp69, tmp67)
        tl.device_assert(((0 <= tl.broadcast_to(tmp71, [XBLOCK])) & (tl.broadcast_to(tmp71, [XBLOCK]) < 8192)) | ~(tmp63 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp71, [XBLOCK]) < 8192")
        tmp73 = tl.load(in_ptr2 + (64*tmp71 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp66 * tmp73
        tmp75 = tl.load(in_ptr0 + (64*x6 + 960*x7 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + (32 + 64*tmp71 + ((-32) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tmp75 * tmp76
        tmp78 = tmp74 + tmp77
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp63, tmp78, tmp79)
        tmp81 = tl.where(tmp47, tmp62, tmp80)
        tl.store(out_ptr1 + (x9), tmp81, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((128, 960), (960, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((128, 9, 64), (576, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 24576, 73728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.00177152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s72 = arg1_1
        s80 = s72
        assert_size_stride(arg0_1, (s72, ), (1, ), 'input')
        assert_size_stride(arg2_1, (49152, 576), (576, 1), 'input')
        assert_size_stride(arg3_1, (576, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_cuda((s72, 576), (576, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((s72, 576), (576, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 576, stream=raw_stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            assert_size_stride(arg4_1, (960, 576), (576, 1), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            buf3 = empty_strided_cuda((s72, 960), (960, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_default, linear], Original ATen: [vllm_ir.rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf2, reinterpret_tensor(arg4_1, (576, 960), (1, 576), 0), out=buf3)
            del arg4_1
            assert_size_stride(arg5_1, (s72, ), (1, ), 'input')
            assert_size_stride(arg6_1, (8192, 64), (64, 1), 'input')
            arg5_1 = copy_if_misaligned(arg5_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            buf4 = empty_strided_cuda((s72, 3, 64), (192, 64, 1), torch.bfloat16)
            buf5 = reinterpret_tensor(buf2, (s72, 9, 64), (576, 64, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view, chunk_1, view_1, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1, cat_2, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_1_xnumel_0 = 192*s72
            triton_poi_fused_1_xnumel_1 = 576*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_1.run(buf3, arg5_1, arg6_1, buf4, buf5, triton_poi_fused_1_xnumel_0, triton_poi_fused_1_xnumel_1, stream=raw_stream0)
            del arg5_1
            del arg6_1
            buf6 = empty_strided_cuda((s72, 576), (576, 1), torch.bfloat16)
        return (buf4, reinterpret_tensor(buf3, (s72, 3, 64), (960, 64, 1), 768), buf5, reinterpret_tensor(buf6, (s72, 9, 64), (576, 64, 1), 0), buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 128
    arg2_1 = rand_strided((49152, 576), (576, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((960, 576), (576, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
