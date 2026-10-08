# Hugging Face Transformers 모델 연산 모듈 및 공통 커널 분석

> Riley 설계를 위한 기존 구조 분석 문서
> 분석 기준 저장소: [`huggingface/transformers`](https://github.com/huggingface/transformers)  
> 고정 리비전: [`c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc`](https://github.com/huggingface/transformers/commit/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc)  
> 리비전 시각: 2026-08-24 07:24:05 UTC  
> 문서 목적: 구현에 착수하기 전에 Transformers에 등록된 모델 구조를 연산 관점으로 분해하고, 재사용 가치가 높은 공통 모듈과 통합하면 안 되는 경계를 식별한다.

---

## 1. 기술 요약

Hugging Face Transformers에는 텍스트, 비전, 음성, 비디오, 멀티모달, 시계열, MoE, 상태공간모델(SSM), 오디오 코덱 등 매우 다양한 모델 계열이 등록되어 있다. 그러나 모델 클래스와 제품 이름은 많아도, 실제 추론 그래프를 구성하는 **핵심 계산 어휘(computational vocabulary)**는 상대적으로 작다.

가장 중요한 결론은 다음과 같다.

1. **거의 모든 모델 계열의 바닥에는 GEMM/Linear, convolution, normalization, activation, residual, reduction, tensor layout 변환이 있다.** 이들은 범용 런타임의 기반 연산이다.
2. **Transformer 계열은 Q/K/V projection, attention score, mask 또는 bias, softmax, value aggregation, output projection, MLP라는 공통 골격을 공유한다.** 텍스트·비전·음성·비디오에서 입력 어댑터만 달라지고 encoder block의 중심은 상당 부분 재사용된다.
3. **현대 decoder-only LLM에서는 RMSNorm, RoPE, MHA/MQA/GQA, causal KV cache, gated MLP(SwiGLU 계열), pre-norm residual 구조가 반복적으로 등장한다.** `riley`의 초기 범위가 Llama/Qwen 계열이라면 가장 먼저 표준화할 영역이다.
4. **모듈 단위보다 반복되는 subgraph 단위가 더 큰 통합 가치를 가진다.** 예를 들어 `RoPE + KV write`, `residual + RMSNorm`, `gate/up projection + SiLU + multiply`, `logits processing + sampling`이 단일 연산보다 실질적인 fusion 후보다.
5. **Attention만으로 전체 Transformers를 설명할 수는 없다.** Mamba의 selective scan, FNet의 FFT, EnCodec의 residual vector quantization, Autoformer의 시계열 분해는 별도 mixer 또는 domain operator로 남겨야 한다.
6. **공통성을 이유로 의미 차이를 지우면 안 된다.** RoPE scaling 방식, pre/post norm, MHA/MQA/GQA/MLA/DSA, local/sliding/global attention, cache layout, multimodal 3D position, quantization dtype는 IR에서 명시적으로 보존해야 한다.
7. Transformers 자체도 `RMSNorm`, `SwiGLUMLP`, `GeGLUMLP`, activation, `rotary_pos_emb`, MoE, Mamba selective scan 등을 의미 단위 kernel ID로 매핑한다. 이는 `riley`의 primitive-centric 접근이 생태계의 실제 최적화 경계와 잘 맞는다는 강한 근거다.

초기 `riley` 관점에서 우선순위를 요약하면 다음과 같다.

| 우선순위 | 범위 | 핵심 항목 |
|---|---|---|
| P0 | 실행 기반 | tensor/dtype/layout, CUDA stream, allocator, embedding, GEMM dispatch, normalization, activation, residual |
| P1 | 현대 decoder LLM | RoPE family, MHA/MQA/GQA, prefill/decode attention, KV cache, SwiGLU, logits/sampling |
| P2 | 고성능 fusion | residual+RMSNorm, RoPE+KV write, fused sampling, paged decode attention, cache-aware layout |
| P3 | 확장 아키텍처 | MoE routing/dispatch/grouped GEMM, MLA/DSA, Mamba/SSM, multimodal projector/mRoPE |
| P4 | 장기 확장 | vision/audio convolution stack, FFT mixer, vector quantization, detection/time-series operators |

---

## 2. 분석 범위와 방법

### 2.1 리비전 고정

Transformers의 `main`은 빠르게 변하므로 분석 기준을 커밋 `c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc`로 고정했다. 모든 소스 링크는 가능한 한 이 SHA의 permalink를 사용한다.

### 2.2 범위 정의

분석 범위는 다음 네 층으로 구성했다.

1. **모델 레지스트리**
   - [`configuration_auto.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/auto/configuration_auto.py)
   - [`auto_mappings.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/auto/auto_mappings.py)
   - AutoModel 계열 task mapping
2. **공유 실행 인프라**
   - [`modeling_layers.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/modeling_layers.py)
   - [`cache_utils.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/cache_utils.py)
   - [`modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/modeling_rope_utils.py)
   - [`masking_utils.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/masking_utils.py)
   - [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/integrations/hub_kernels.py)
3. **모델 계열별 대표 구현**
   - Llama, BERT, T5, ViT, CLIP, ConvNeXt, TimeSformer
   - Whisper, Wav2Vec2, EnCodec
   - Qwen2-VL
   - Mixtral, DeepSeek-V3
   - Mamba, Jamba, FNet
   - Autoformer, DETR
4. **추론 엔진 관점의 정규화**
   - 클래스 이름이 아니라 연산 의미, shape, state, layout, dtype, 실행 단계(prefill/decode)를 기준으로 모듈을 재분류했다.

### 2.3 이 문서의 정량 해석 방식

이 문서는 거짓 정밀도를 피하기 위해 “Transformers 전체의 정확한 몇 %” 같은 수치를 임의로 만들지 않는다. 대신 다음 공통성 등급을 사용한다.

| 등급 | 의미 | 판단 기준 |
|---|---|---|
| U0 | 거의 보편적 | Transformer 여부와 무관하게 다수 모달리티에서 반복 |
| U1 | Transformer 전반 | encoder, decoder, vision/audio Transformer에 폭넓게 반복 |
| U2 | 현대 생성형 LLM 지배적 | Llama/Qwen/Mistral/Gemma/DeepSeek 계열에서 반복 |
| U3 | 모달리티 또는 계열 반복 | vision, audio, MoE, SSM 등 특정 그룹에서 반복 |
| U4 | 특수 구조 | 소수 모델 또는 특정 task에 종속 |

정확한 파일별 출현 횟수는 향후 별도의 **AST/IR census**로 산출해야 한다. 현재 단계에서는 구현을 시작하지 않는다는 원칙에 따라, 해당 도구의 설계 요건만 마지막에 제시한다.

---

## 3. Transformers 모델 구조의 큰 분류

등록된 모델을 연산 구조로 묶으면 다음과 같이 분류할 수 있다.

| 구조 계열 | 대표 모델 | 중심 연산 | 상태/캐시 | `riley` 관련성 |
|---|---|---|---|---|
| Decoder-only dense Transformer | Llama, Qwen, Mistral, Gemma | causal attention, RoPE, gated MLP | KV cache | 초기 핵심 |
| Encoder-only Transformer | BERT, RoBERTa, ViT | bidirectional attention, GELU MLP | 보통 없음 | 공통 encoder 확장 |
| Encoder-decoder Transformer | T5, Whisper, BART | self-attention + cross-attention | self/cross KV | 중기 확장 |
| Vision Transformer | ViT, CLIP vision | patch embedding + attention | 보통 없음 | 입력 어댑터 재사용 |
| Convolutional vision | ConvNeXt | depthwise Conv2d + pointwise MLP | 없음 | 장기 확장 |
| Video Transformer | TimeSformer | spatial/temporal attention | 없음 | layout 특화 |
| Audio Transformer | Whisper, Wav2Vec2 | Conv1d frontend + attention | 모델별 상이 | 모달리티 확장 |
| Neural audio codec | EnCodec | Conv1d/ConvTranspose1d, LSTM, RVQ | recurrent/codec state | 별도 런타임 영역 |
| Contrastive dual encoder | CLIP | text/vision encoder + projection + normalize + similarity | 없음 | 공통 encoder 조합 |
| Multimodal generative | Qwen2-VL | vision encoder + merger + LLM + mRoPE | text KV | projector/token packing 필요 |
| Sparse MoE Transformer | Mixtral, DeepSeek-V3 | router, top-k, dispatch, expert GEMM | KV + routing metadata | P3 고가치 |
| SSM | Mamba | causal conv + selective scan/update | recurrent state | 별도 mixer |
| Hybrid attention/SSM | Jamba | attention layer + Mamba layer + MoE | KV + recurrent state | IR 검증용 |
| Fourier mixer | FNet | FFT + MLP | 없음 | mixer 추상화 검증 |
| Time-series Transformer | Autoformer | scaling, decomposition, auto-correlation/attention, probabilistic head | seq2seq cache | 낮은 초기 우선순위 |
| Detection/segmentation | DETR | backbone, object query, cross-attention, box/mask head | 없음 | task head 분리 필요 |

이 분류에서 중요한 점은 **모델 family를 하나의 거대한 enum으로 고정하지 않고, Input Adapter + Repeated Block + State + Output Head의 조합으로 표현하는 것**이다.

---

## 4. 공통 연산 모듈 전체 목록

### 4.1 U0: 거의 모든 계열에 나타나는 기반 연산

| 모듈 | 대표 형태 | 등장 이유 | 성능 특성 | 통합 권고 |
|---|---|---|---|---|
| Linear / GEMM | `Y = XWᵀ + b` | projection, MLP, head, router | 대부분의 Transformer FLOPs | 직접 GEMM보다 cuBLASLt/CUTLASS dispatch |
| Batched matmul | QKᵀ, AV, similarity | attention, contrastive, VQ 거리 | shape에 따라 compute/memory bound | attention·specialized backend에 포함 |
| Embedding lookup | token, position, type, codebook | discrete ID → vector | memory bandwidth 중심 | 범용 gather kernel |
| Elementwise add/mul | residual, gate, scale | 모든 block 연결 | launch/memory overhead 비중 큼 | epilogue/fusion 우선 |
| Reduction | mean, sum, variance, max | norm, pooling, softmax, routing | 작은 batch에서 overhead 민감 | norm/softmax 내부 통합 |
| Reshape/view | logical shape change | heads, patches, experts | 보통 zero-copy | metadata-only 여부 보장 |
| Transpose/permute | BSHD↔BHSD 등 | attention·conv layout | materialization 시 큰 비용 | layout planner 필요 |
| Contiguous/copy | backend 요구 layout | kernel ABI 정렬 | 숨은 bandwidth 비용 | 가능한 한 제거/지연 |
| Concatenate/split/chunk | QKV, gate/up, multimodal packing | graph composition | 메모리 복사 가능 | packed projection/aliasing 활용 |
| Gather/scatter/index_add | MoE, cache, beam, token select | irregular data movement | 낮은 locality | 전용 kernel 후보 |
| Cast | FP32 accumulator ↔ BF16/FP16 | norm/softmax 안정성 | 추가 bandwidth | kernel 내부 처리 |
| Masked fill/select | attention, padding, routing | 조건부 값 제거 | elementwise | attention/router fusion |

**판단:** `riley`의 첫 번째 실제 자산은 개별 모델 코드가 아니라 tensor descriptor, stride/layout, dtype conversion, allocator, workspace, stream lifetime을 안전하게 다루는 Rust runtime이어야 한다.

### 4.2 U0/U3: Convolution 계열

| 연산 | 주요 사용처 | 대표 구현 | 비고 |
|---|---|---|---|
| Conv1d | 음성 frontend, positional conv, Mamba causal conv | Wav2Vec2, Whisper, Mamba, EnCodec | 일반 conv와 depthwise/grouped 변형 구분 |
| Conv2d | image patch embedding, CNN backbone | ViT, CLIP, ConvNeXt, DETR | patch embedding은 stride=kernel 패턴이 많음 |
| Conv3d | video/vision-volume patch embedding | Qwen2-VL 일부 vision path | temporal-spatial layout 포함 |
| ConvTranspose1d | 오디오 decoder | EnCodec | causal trim/padding 규칙 중요 |
| Depthwise convolution | ConvNeXt, Mamba/SSM, audio | groups=channels | bandwidth·layout 특화 |
| Pointwise convolution | ConvNeXt | Linear 또는 1×1 Conv | channel-last 변환 비용 주의 |

초기 LLM 범위에서는 Conv를 중심에 둘 필요는 없지만, **Mamba/Jamba 확장 가능성 때문에 depthwise causal Conv1d ABI는 일찍 분리해둘 가치가 있다.**

---

## 5. 정규화 모듈

### 5.1 정규화 종류

| 종류 | 계산 개요 | 대표 계열 | 공통성 | 추론 중요도 |
|---|---|---|---|---|
| RMSNorm | RMS만 정규화, mean subtraction 없음 | Llama, Mixtral, DeepSeek, Jamba, T5 계열 변형 | U2 | 매우 높음 |
| LayerNorm | mean/variance 정규화 | BERT, ViT, CLIP, Whisper, FNet | U1 | 매우 높음 |
| GroupNorm | channel group 단위 | Wav2Vec2, EnCodec, vision/audio | U3 | 중간 |
| BatchNorm/FrozenBatchNorm | running stats | CNN/DETR backbone | U3 | 중간 |
| WeightNorm | weight 재매개변수화 | Wav2Vec2 positional conv, EnCodec | U3 | inference folding 가능성 검토 |
| QK Norm | Q/K head normalization | 일부 최신 LLM/VLM | U3 | attention spec에 포함 |
| Gated RMSNorm | norm + gate | 일부 SSM/linear-attention 계열 | U3 | fused kernel 가치 높음 |

### 5.2 같은 이름처럼 보여도 보존해야 할 파라미터

정규화 IR은 최소한 다음을 가져야 한다.

```rust
pub enum NormKind {
    LayerNorm,
    RmsNorm,
    GroupNorm { groups: u32 },
    FrozenBatchNorm,
    QkNorm,
}

pub struct NormSpec {
    pub kind: NormKind,
    pub eps: f32,
    pub affine: bool,
    pub bias: bool,
    pub accumulator_dtype: DType,
}
```

필수 차이:

- epsilon 값
- weight만 있는지 bias도 있는지
- FP32 accumulation 여부
- normalization axis
- channel-first/channel-last
- residual add 이전 또는 이후인지
- gate와 결합되는지

### 5.3 Fusion 가치

가치가 높은 패턴:

1. `residual + RMSNorm`
2. `residual + LayerNorm`
3. `RMSNorm + QKV projection`의 GEMM prologue
4. `RMSNorm + gate`
5. `bias + residual + LayerNorm`

단, norm과 대형 GEMM 전체를 처음부터 커스텀 CUDA로 합치기보다는 CUTLASS prologue/epilogue 또는 별도 norm kernel과의 비교가 필요하다.

---

## 6. Activation과 MLP 계열

### 6.1 Activation inventory

| Activation | 주요 계열 | 특징 | 통합 판단 |
|---|---|---|---|
| GELU | BERT, ViT, Whisper, FNet | classic Transformer 기본 | P0 |
| GELU tanh approximation | 다수 encoder | 근사식 차이 | variant 보존 |
| QuickGELU | CLIP 계열 | sigmoid 기반 근사 | 별도 variant |
| NewGELU/FastGELU | 일부 모델 | 구현식 차이 | semantic ID 유지 |
| SiLU/Swish | Llama/Qwen/Mistral MLP, Mamba gate | 현대 LLM 지배적 | P0/P1 |
| ReLU | T5 일부, legacy Transformer | 단순 | 범용 |
| ELU | EnCodec | audio codec | P4 |
| Tanh/Sigmoid | pooler, router, recurrent gate | head/router/SSM | 범용 elementwise |
| Softplus | Mamba delta | state-space discretization | SSM 전용 |

Transformers의 kernel mapping도 `FastGELU`, `QuickGELU`, `NewGELU`, `SiLU`, `GeLU`, `GeluTanh`를 서로 다른 의미 ID로 관리한다. `riley` 역시 “GELU 비슷한 것”으로 뭉개지 말고 수식 variant를 보존해야 한다.

### 6.2 MLP 패턴

#### 표준 FFN

```text
hidden
  └─ Linear(hidden → intermediate)
       └─ GELU/ReLU/SiLU
            └─ Linear(intermediate → hidden)
```

대표: BERT, ViT, CLIP, Whisper, FNet.

#### Gated MLP / SwiGLU

```text
                  ┌─ gate_proj ─ activation ─┐
hidden ───────────┤                         × ├─ down_proj
                  └─ up_proj ────────────────┘
```

대표: Llama, Mixtral의 experts, DeepSeek-V3 dense/shared experts.

#### Packed gate/up projection

일부 구현은 gate와 up weight를 하나로 묶어 한 번의 GEMM 후 `chunk(2)`한다.

```text
hidden ─ gate_up GEMM ─ split ─ activation(gate) × up ─ down GEMM
```

이 패턴은 kernel API에서 다음을 지원해야 한다.

- separate weight 또는 packed weight
- activation 종류
- bias 유무
- quantized weight format
- tensor-parallel shard 방식
- expert dimension 유무

### 6.3 추천 모듈 경계

```rust
pub enum MlpKind {
    Dense {
        activation: Activation,
    },
    Gated {
        activation: Activation,
        packed_gate_up: bool,
    },
    Moe(MoeSpec),
}
```

`SwiGLU`를 단일 activation으로만 보지 말고 **두 projection + activation + multiply + down projection의 구조 패턴**으로 보는 편이 정확하다.

---

## 7. Attention 계열의 공통 골격

### 7.1 공통 계산 그래프

대부분의 attention 구현은 다음을 공유한다.

```text
hidden states
  ├─ Q projection
  ├─ K projection
  └─ V projection
        ↓
reshape / transpose to heads
        ↓
position transform or bias
        ↓
KV cache update (generative models)
        ↓
Q × Kᵀ × scale
        ↓
mask / bias
        ↓
softmax
        ↓
probability × V
        ↓
merge heads
        ↓
output projection
```

### 7.2 Attention 종류

| 종류 | Q head 수 | KV head 수 | 대표 사용 | cache 특성 |
|---|---:|---:|---|---|
| MHA | H | H | BERT, ViT, CLIP, legacy LLM | 가장 큰 KV |
| MQA | H | 1 | 일부 decoder | KV 메모리 최소 |
| GQA | H | 1 < K < H | Llama/Qwen/Mistral/Mixtral | 현대 LLM 핵심 |
| Cross-attention | query와 source 분리 | 모델별 | T5, Whisper, DETR | encoder K/V 재사용 |
| Sliding-window | local window | 모델별 | Mistral/Mixtral 등 | bounded KV |
| Window attention | 2D window | Swin 계열 | vision layout | window partition 필요 |
| Spatial/temporal attention | 축 분리 | TimeSformer | video | reshape 비용 큼 |
| Deformable attention | sampled points | Deformable DETR | detection | irregular gather |
| Sparse/DSA | selected tokens | 최신 sparse 모델 | LLM | index cache 추가 |
| MLA | latent KV representation | DeepSeek 계열 | LLM | standard GQA와 cache layout 상이 |
| Linear/recurrent attention | kernelized recurrence | 일부 최신 모델 | long sequence | KV 대신 state |

### 7.3 Prefill과 decode를 분리해야 하는 이유

동일한 attention layer라도 workload가 다르다.

| 구분 | Query 길이 | KV 길이 | 병목 | 권장 실행 전략 |
|---|---:|---:|---|---|
| Prefill | 길다 | 길다 | compute/attention memory | FlashAttention 계열, 큰 tile |
| Decode | 보통 1 또는 소수 | 누적 길다 | KV memory bandwidth, launch | paged/grouped decode kernel |
| Chunked prefill | 중간 chunk | 누적 | scheduling + cache write | continuous batching 연계 |
| Speculative verify | 여러 token | 누적 | shape 변동 | 별도 mode/capture bucket |

따라서 `AttentionSpec`만으로 충분하지 않고 실행 시 `AttentionMode`가 필요하다.

```rust
pub enum AttentionMode {
    Prefill,
    ChunkedPrefill,
    Decode,
    Verify,
}
```

### 7.4 Attention backend interface

Transformers도 `ALL_ATTENTION_FUNCTIONS`를 통해 eager, SDPA, FlashAttention 등의 구현을 교체한다. `riley`는 더 낮은 수준에서 다음 capability 기반 dispatch를 권장한다.

```rust
pub struct AttentionRequest<'a> {
    pub mode: AttentionMode,
    pub q: TensorView<'a>,
    pub k_new: TensorView<'a>,
    pub v_new: TensorView<'a>,
    pub cache: Option<KvCacheView<'a>>,
    pub mask: MaskSpec,
    pub position: PositionSpec,
    pub scale: f32,
    pub output: TensorViewMut<'a>,
}
```

backend 선택 조건:

- GPU compute capability
- dtype/quantization
- head dimension
- MHA/MQA/GQA/MLA
- query length / KV length
- causal/local/sliding 여부
- paged cache 지원
- CUDA Graph capture 가능 여부
- workspace 크기

---

## 8. Position encoding과 attention bias

### 8.1 주요 종류

| 종류 | 적용 지점 | 대표 계열 | 통합 난이도 |
|---|---|---|---|
| Learned absolute embedding | hidden에 add | BERT, ViT, CLIP | 낮음 |
| Sinusoidal absolute | hidden에 add | Whisper encoder 계열, DETR 2D 변형 | 낮음~중간 |
| RoPE | Q/K 회전 | Llama, Mixtral, DeepSeek, Qwen | 중간 |
| Partial RoPE | head 일부만 회전 | 일부 LLM | 중간 |
| Linear scaled RoPE | inverse frequency scale | long context | 중간 |
| Dynamic NTK RoPE | seq length에 따라 재계산 | long context | 높음 |
| YaRN/LongRoPE 등 | frequency/attention scale 변형 | 장문 모델 | 높음 |
| Multimodal RoPE | temporal/height/width section | Qwen2-VL | 높음 |
| Relative position bias | attention logits에 add | T5 | 중간 |
| Relative key/query | score 계산에 추가 | BERT 변형, audio models | 중간 |
| ALiBi | head별 linear bias | 일부 LLM | 중간 |
| 2D sine position | image grid | DETR | 모달리티 특화 |
| Spatial + temporal embedding | video | TimeSformer | layout 특화 |

### 8.2 RoPE는 하나의 kernel이 아니라 family다

`modeling_rope_utils.py`에는 default 외에도 linear scaling, proportional, dynamic NTK, long context 관련 계산이 분리되어 있다. 따라서 IR은 단순 `rope: bool`이 아니라 다음과 비슷해야 한다.

```rust
pub enum RopeKind {
    Default,
    Linear { factor: f32 },
    DynamicNtk { factor: f32, original_max_len: u32 },
    Yarn { factor: f32, attention_factor: f32 },
    LongRope { short_factors: Vec<f32>, long_factors: Vec<f32> },
    Multimodal3d { sections: [u32; 3] },
    Custom,
}

pub struct RopeSpec {
    pub kind: RopeKind,
    pub theta: f32,
    pub rotary_dim: u32,
    pub interleaved: bool,
}
```

### 8.3 가장 유력한 fusion

```text
Q/K projection output
      ↓
head reshape
      ↓
RoPE apply
      ↓
K/V cache write
```

특히 decode에서 `RoPE + K/V write + layout conversion`을 하나로 묶으면 작은 kernel launch와 중간 tensor write를 줄일 수 있다. 다만 mRoPE, partial RoPE, interleaved layout을 capability로 분리해야 한다.

---

## 9. KV cache와 recurrent state

### 9.1 Transformers cache 계층에서 확인되는 동작

`cache_utils.py`는 단순한 K/V tensor 외에도 다음 동작을 다룬다.

- dynamic append
- static preallocation
- sliding window
- crop/rollback
- beam reorder
- batch repeat/select
- CPU offload와 prefetch
- encoder-decoder cache
- sparse attention용 index key
- 모델별 layer type 등록
- recurrent/SSM state 확장

### 9.2 cache 종류를 하나의 tensor로 추상화하면 안 되는 이유

| cache/state | 논리 내용 | shape 특성 | 주요 연산 |
|---|---|---|---|
| Dynamic KV | 계속 증가하는 K/V | `[B, Hkv, T, D]` | append/cat |
| Static KV | 최대 길이 선할당 | 고정 buffer | indexed write |
| Sliding KV | 최근 window 유지 | bounded T | ring/crop |
| Paged KV | block table 기반 | 비연속 block | gather/address translation |
| Cross-attn KV | encoder source 고정 | decode 동안 불변 | 최초 계산 후 재사용 |
| Quantized KV | compressed K/V | scale/zero metadata | quant/dequant |
| Offloaded KV | GPU/CPU/remote | tier 이동 | async prefetch |
| Indexed sparse cache | K/V + index key | 별도 index tensor | search/select |
| MLA latent cache | compressed latent/rope part | 모델 특화 | custom attention |
| SSM recurrent state | conv state + recurrent state | seq dimension 없음/작음 | in-place update |

### 9.3 Riley cache IR 제안

```rust
pub enum StateSpec {
    Kv(KvCacheSpec),
    Recurrent(RecurrentStateSpec),
    Hybrid(Vec<StateSpec>),
    None,
}

pub struct KvCacheSpec {
    pub layout: KvLayout,
    pub allocation: KvAllocation,
    pub window: Option<u32>,
    pub dtype: DType,
    pub quantization: Option<QuantSpec>,
    pub cross_attention: bool,
}
```

Rust의 ownership은 block lifetime, reference count, prefix sharing, cancellation cleanup을 표현하기에 적합하다. 다만 Rust를 쓴다고 fragmentation과 synchronization이 자동으로 해결되는 것은 아니므로 allocator 정책과 lock contention을 별도 benchmark해야 한다.

---

## 10. MoE 공통 모듈

### 10.1 공통 그래프

Mixtral과 DeepSeek-V3 계열에서 반복되는 기본 구조는 다음과 같다.

```text
hidden states
  ↓
router Linear
  ↓
softmax 또는 sigmoid
  ↓
top-k / group top-k
  ↓
weight normalization/scaling
  ↓
token → expert dispatch
  ↓
expert gate/up GEMM
  ↓
activation × up
  ↓
expert down GEMM
  ↓
weighted combine / index_add
  ↓
shared expert add (일부 모델)
```

### 10.2 MoE 내부 모듈

| 모듈 | 연산 | 성능 이슈 | 통합 가치 |
|---|---|---|---|
| Router projection | GEMM | 비교적 작음 | 높음 |
| Router activation | softmax/sigmoid | FP32 안정성 | 높음 |
| Top-k selection | topk/sort | 작은 row, irregular | 높음 |
| Group selection | group reduce + topk | DeepSeek류 | 중간 |
| Dispatch metadata | histogram/prefix sum/sort | CPU 개입 시 병목 | 매우 높음 |
| Expert GEMM | grouped GEMM | load imbalance | 매우 높음 |
| Combine | weighted scatter/index_add | atomics/locality | 매우 높음 |
| Shared expert | dense gated MLP | 병렬 실행 가능 | 중간 |
| Expert parallel communication | all-to-all | multi-GPU | 장기 |

### 10.3 통합 시 보존할 차이

- softmax router와 sigmoid router
- top-k 개수
- group 제한 방식
- router bias/correction bias
- expert capacity/drop 정책
- shared expert 유무
- expert weight packing
- token permutation 안정성
- deterministic requirement
- tensor/expert parallel topology

MoE는 하나의 `MoeKernel`이 아니라 **routing plan + grouped compute + combine plan**으로 나누는 것이 좋다.

---

## 11. SSM, recurrent, hybrid mixer

### 11.1 Mamba 계열

Mamba는 attention 대신 다음 연산을 중심으로 한다.

1. input projection 후 state branch와 gate 분리
2. depthwise causal Conv1d
3. `dt`, `B`, `C` projection
4. `A`와 step을 이용한 discretization
5. selective scan 또는 single-token selective state update
6. skip `D`
7. SiLU gate
8. output projection

핵심 kernel family:

- `causal_conv1d_fn`
- `causal_conv1d_update`
- `selective_scan_fn`
- `selective_state_update`
- fused Mamba inner/chunk scan

Transformers의 kernel mapping도 이 함수들을 별도 최적화 대상으로 등록하고 있다.

### 11.2 Jamba가 보여주는 IR 요구사항

Jamba는 layer마다 attention 또는 Mamba mixer를 선택하고 MoE까지 결합한다. 따라서 모델 전체를 `TransformerModel`과 `MambaModel` 중 하나로 분류하면 부족하다.

```rust
pub enum MixerSpec {
    Attention(AttentionSpec),
    StateSpace(StateSpaceSpec),
    Fourier(FourierSpec),
    Convolution(ConvMixerSpec),
}

pub struct BlockSpec {
    pub input_norm: Option<NormSpec>,
    pub mixer: MixerSpec,
    pub post_mixer_norm: Option<NormSpec>,
    pub channel_mixer: ChannelMixerSpec,
    pub residual: ResidualSpec,
    pub state_slot: Option<StateSlotId>,
}
```

즉 반복 block의 `mixer`를 tagged union으로 표현해야 hybrid model을 자연스럽게 지원할 수 있다.

### 11.3 FNet

FNet은 token mixing에 2D FFT를 사용하고, channel mixing은 BERT형 FFN을 사용한다. 이는 다음 사실을 보여준다.

- attention은 mixer의 한 종류일 뿐이다.
- normalization/residual/MLP는 mixer가 달라도 재사용된다.
- IR에서 “attention block”이 아니라 “mixer + channel mixer + residual topology”를 분리하는 편이 미래 확장에 유리하다.

---

## 12. 모달리티별 입력 어댑터

### 12.1 Text

```text
token IDs
  ↓
Embedding lookup
  + optional token-type embedding
  + learned/sinusoidal position embedding
  ↓
hidden states
```

현대 RoPE decoder는 hidden에 position embedding을 더하지 않고 Q/K 단계에서 회전한다.

### 12.2 Vision

#### ViT/CLIP

```text
image
  ↓
Conv2d(kernel=patch, stride=patch)
  ↓
flatten + transpose
  ↓
CLS token concat
  ↓
position embedding add/interpolate
```

#### ConvNeXt

```text
Conv2d stem
  ↓
depthwise Conv2d
  ↓
channel-last LayerNorm
  ↓
pointwise expansion
  ↓
GELU
  ↓
pointwise contraction
  ↓
layer scale + residual
```

### 12.3 Video

TimeSformer는 frame별 patch embedding 후 spatial/temporal 축을 반복 reshape하여 attention을 수행한다. 여기서는 attention kernel 자체보다 **layout transform 최소화**가 매우 중요하다.

### 12.4 Audio

#### Wav2Vec2

- strided Conv1d feature extractor
- GroupNorm 또는 LayerNorm
- positional Conv1d
- Transformer encoder
- 일부 pretraining path의 quantization/contrastive loss

#### Whisper

- log-mel feature 입력
- Conv1d frontend
- encoder self-attention
- decoder causal self-attention + cross-attention
- self/cross KV cache

#### EnCodec

- causal/asymmetric Conv1d
- residual dilated blocks
- LSTM
- residual vector quantization
- ConvTranspose1d decoder

### 12.5 Multimodal generative

Qwen2-VL의 대표 조합:

```text
image/video
  ↓
Conv3d patch embedding
  ↓
vision Transformer
  ↓
patch merger/projector
  ↓
text token sequence에 삽입
  ↓
LLM decoder
  ↓
multimodal RoPE
```

따라서 멀티모달 지원은 모델 core에 모든 로직을 넣기보다 다음으로 분리하는 것이 좋다.

- modality encoder
- projector/merger
- token packing plan
- position ID builder
- shared text decoder

---

## 13. Task head와 inference core의 분리

Transformers는 하나의 base model 위에 다양한 head를 붙인다.

| head | 연산 | 예시 |
|---|---|---|
| Causal LM | hidden → vocab Linear | Llama/Qwen |
| Masked LM | transform + vocab Linear | BERT/FNet |
| Sequence classification | pooling/select + Linear | BERT/CLIP variants |
| Token classification | per-token Linear | NER |
| Question answering | start/end projection | BERT/T5 variants |
| Contrastive projection | Linear + L2 normalize + similarity | CLIP |
| Object detection | class Linear + box MLP | DETR |
| Segmentation | query-feature interaction + mask head | DETR variants |
| CTC | per-frame vocab projection | Wav2Vec2 |
| Seq2Seq LM | decoder vocab projection | T5/Whisper |
| Time-series distribution | parameter projection + distribution | Autoformer |
| Codec decode | codebook lookup + decoder | EnCodec |

`riley` 초기 버전에서는 `CausalLMHead + Sampling`만 직접 최적화하고, 나머지는 core와 분리된 extension으로 보는 것이 안전하다.

---

## 14. 대표 아키텍처별 연산 매트릭스

| 모델 | 입력 | Norm | Position | Mixer | Channel mixer | State/cache | 특수 연산 |
|---|---|---|---|---|---|---|---|
| Llama | token embedding | RMSNorm, pre-norm | RoPE | causal GQA | SwiGLU | KV | repeat_kv |
| BERT | token+type+position | LayerNorm, post-norm | learned absolute | bidirectional MHA | GELU FFN | 보통 없음 | pooler/head |
| T5 | token embedding | RMS-style pre-norm | relative bias | self/cross MHA | dense/gated FFN | encoder-decoder KV | relative buckets |
| ViT | Conv2d patch | LayerNorm, pre-norm | learned/interpolated | MHA | GELU FFN | 없음 | CLS token |
| CLIP | text embedding + image patch | LayerNorm | learned absolute | MHA | GELU/QuickGELU MLP | 없음 | L2 norm, similarity |
| ConvNeXt | Conv2d stem | LayerNorm | 없음 | depthwise Conv2d | pointwise GELU MLP | 없음 | layer scale, DropPath |
| TimeSformer | frame patch | LayerNorm | spatial+temporal | space/time MHA | GELU FFN | 없음 | 반복 reshape |
| Whisper | audio Conv1d + token embedding | LayerNorm | sinusoidal/learned | encoder self + decoder self/cross | GELU FFN | self/cross KV | audio frontend |
| Wav2Vec2 | raw audio Conv1d | Group/LayerNorm | positional Conv1d | encoder MHA | GELU FFN | 보통 없음 | feature masking/VQ 일부 |
| EnCodec | raw audio Conv1d | Group/weight norm | 없음 | Conv/LSTM | residual Conv | recurrent | RVQ, ConvTranspose |
| Qwen2-VL | Conv3d vision + text | RMSNorm/LayerNorm | mRoPE + vision RoPE | vision MHA + text GQA | vision MLP + SwiGLU | text KV | patch merger, packing |
| Mixtral | token embedding | RMSNorm | RoPE | causal GQA/SWA | sparse expert SwiGLU | KV | top-k router/dispatch |
| DeepSeek-V3 | token embedding | RMSNorm | RoPE/YaRN family | specialized latent attention | dense/shared/sparse expert | latent/custom KV | grouped router |
| Mamba | token embedding | RMSNorm 계열 | explicit RoPE 없음 | selective SSM | gated projection | recurrent state | scan, causal conv |
| Jamba | token embedding | RMSNorm | 별도 RoPE 없음 | Attention + SSM | dense/MoE | KV + recurrent | hybrid layer schedule |
| FNet | token+type+position | LayerNorm, post-norm | learned absolute | FFT | GELU FFN | 없음 | complex FFT |
| Autoformer | numeric/categorical features | LayerNorm 등 | sinusoidal/time features | FFT autocorrelation + delay aggregation | FFN | seq2seq KV | series decomposition, scaling, probabilistic head |
| DETR | CNN backbone | FrozenBN/LayerNorm | 2D sine/learned | encoder + query cross-attn | FFN | 없음 | object queries, box/mask head |

---

## 15. 가장 공통적으로 사용되는 모듈 순위

정확한 전체 AST 출현 횟수가 아니라, **구조 범위와 추론 중요도를 결합한 우선순위**다.

### Tier A — 런타임에 반드시 있어야 하는 범용 primitive

1. Linear/GEMM dispatch
2. tensor view/reshape/transpose/layout conversion
3. elementwise add/multiply/scale/cast
4. reductions
5. embedding/gather
6. residual connection
7. activation family
8. normalization family
9. masking/select
10. allocator/workspace/stream/event 관리

### Tier B — Transformer 전체에서 반복되는 core

1. Q/K/V projection
2. head reshape/merge
3. attention QKᵀ 및 AV
4. mask/bias application
5. softmax
6. output projection
7. two-layer FFN 또는 gated MLP
8. pre-norm/post-norm residual topology
9. dropout/DropPath의 inference no-op 처리

### Tier C — 현대 decoder LLM에서 통합 가치가 가장 큰 모듈

1. RMSNorm
2. RoPE family
3. MHA/MQA/GQA abstraction
4. causal mask
5. dynamic/static/paged/sliding KV cache
6. prefill/decode attention 분기
7. SwiGLU/gated MLP
8. logits projection
9. repetition/frequency penalty
10. top-k/top-p/temperature sampling

### Tier D — 확장 가치가 큰 특화 모듈

1. MoE router/top-k/dispatch/combine/grouped GEMM
2. MLA/DSA 및 specialized cache
3. selective scan/state update
4. causal depthwise Conv1d update
5. multimodal projector/token packing/mRoPE
6. vision patch embedding/interpolation
7. cross-attention cache

### Tier E — 장기 또는 별도 도메인 모듈

1. ConvTranspose/audio codec stack
2. residual vector quantization
3. FFT mixer
4. object detection query/matching/head
5. time-series decomposition/scaler/distribution
6. protein/geometry 전용 연산

---

## 16. 통합 가치 평가 매트릭스

평가 기준:

- **범위:** 얼마나 많은 구조가 재사용하는가
- **비용:** end-to-end 추론 시간에서 비중이 큰가
- **Fusion:** 중간 read/write와 launch를 줄일 수 있는가
- **안정성:** 모델 간 의미가 안정적인가
- **변형 위험:** shape/수식/layout 차이가 큰가

| 후보 | 범위 | 비용 | Fusion | 의미 안정성 | 변형 위험 | 권고 |
|---|---|---|---|---|---|---|
| GEMM dispatch | 매우 높음 | 매우 높음 | epilogue 중심 | 높음 | dtype/quant 큼 | P0, vendor backend |
| Embedding/gather | 높음 | 중간 | 제한적 | 높음 | vocab/layout | P0 |
| RMSNorm | LLM 높음 | 중간 | 높음 | 높음 | eps/dtype | P0/P1 |
| LayerNorm | Transformer 높음 | 중간 | 높음 | 높음 | axis/bias | P0 |
| GELU/SiLU | 높음 | 낮음~중간 | 매우 높음 | variant별 높음 | 근사식 | P0 |
| residual+norm | 높음 | 중간 | 매우 높음 | 높음 | topology | P1 |
| RoPE | LLM 높음 | 중간 | 매우 높음 | family 내부 중간 | variant 큼 | P1 |
| RoPE+KV write | LLM 높음 | 중간 | 매우 높음 | 중간 | cache/layout | P2 |
| Prefill attention | Transformer 높음 | 매우 높음 | backend 내부 | 중간 | mask/head/layout | P1 |
| Decode attention | LLM 높음 | 매우 높음 | 매우 높음 | 중간 | cache 종류 | P1/P2 |
| SwiGLU epilogue | LLM 높음 | 높음 | 매우 높음 | 높음 | packing/quant | P1 |
| Sampling | 생성 모델 높음 | batch 작을 때 중요 | 매우 높음 | 중간 | 정책 조합 | P1/P2 |
| KV allocator | LLM 높음 | 간접적으로 매우 높음 | N/A | 엔진별 | policy 큼 | P0/P1 |
| MoE routing | MoE 중간 | 높음 | 높음 | 중간 | router 차이 | P3 |
| Expert grouped GEMM | MoE 중간 | 매우 높음 | backend 내부 | 높음 | imbalance | P3 |
| Selective scan | SSM 제한적 | 매우 높음 | 매우 높음 | family별 | 세대 차이 | P3 |
| Patch embedding | vision 높음 | 중간 | 제한적 | 높음 | 2D/3D | P4 |
| Conv audio frontend | audio 높음 | 중간 | 중간 | 중간 | padding/norm | P4 |
| RVQ | codec 제한적 | 높음 | 중간 | codec별 | codebook 차이 | P4 |
| FFT mixer | 제한적 | 높음 | 제한적 | 높음 | complex dtype | P4 |

---

## 17. 추천 fused subgraph 목록

### 17.1 초기 후보

| 패턴 | 기대 효과 | 난이도 | 주의점 |
|---|---|---:|---|
| residual + RMSNorm | read/write 및 launch 감소 | 낮음~중간 | FP32 reduction |
| residual + LayerNorm | encoder 재사용 | 중간 | mean/variance |
| bias + activation | GEMM epilogue | 낮음 | activation variant |
| gate/up output + SiLU + multiply | SwiGLU 중간 tensor 제거 | 중간 | packed/separate weight |
| RoPE + K/V cache write | decode launch/layout 감소 | 중간~높음 | RoPE/cache variants |
| QKV split + reshape + RoPE | metadata/copy 제거 | 높음 | projection output layout |
| logits penalties + temperature | sampling 전처리 통합 | 중간 | per-request params |
| top-k + top-p + RNG sample | CPU round-trip 제거 | 높음 | determinism |
| cache block gather + decode attention | paged cache 최적화 | 매우 높음 | address translation |

### 17.2 중기 후보

| 패턴 | 대상 |
|---|---|
| router logits + activation + top-k | MoE |
| dispatch histogram + prefix sum + permutation | MoE |
| grouped expert GEMM + combine | MoE |
| causal Conv1d update + state update | Mamba/Jamba |
| multimodal position build + token packing | VLM |
| cross-attention K/V projection + persistent cache | seq2seq |

### 17.3 처음부터 직접 만들지 않을 영역

- 일반 dense GEMM 전체
- 모든 shape의 universal FlashAttention 대체품
- NCCL 대체 collective
- 모든 quantization format
- 모든 GPU 세대용 최적 kernel

이들은 기존 vendor/library 구현을 호출하고, `riley`는 **dispatch, layout, state management, fusion boundary**에 집중하는 것이 합리적이다.

---

## 18. Transformers의 현재 kernel 경계가 주는 시사점

`hub_kernels.py`의 기본 mapping에는 다음과 같은 의미 단위가 이미 나타난다.

- `RMSNorm`
- `RMSNormGated`
- `SwiGLUMLP`
- `GeGLUMLP`
- `Linear`
- `FastGELU`, `QuickGELU`, `NewGELU`, `SiLU`, `GeLU`, `GeluTanh`
- `rotary_pos_emb`
- `MegaBlocksMoeMLP`
- `causal_conv1d_fn`, `causal_conv1d_update`
- `selective_scan_fn`, `selective_state_update`
- fused Mamba scan variants
- attention kernel repository loading
- Causal LM loss kernel

이 목록은 중요한 검증 결과다.

1. **모델 이름이 아니라 연산 의미 이름으로 kernel을 교체한다.**
2. **동일 layer도 device와 training/inference mode에 따라 구현이 달라진다.**
3. **RMSNorm, gated MLP, RoPE, SSM, MoE가 실제 최적화 경계로 인정된다.**
4. **attention은 하나의 고정 구현이 아니라 repository/backend dispatch 대상이다.**
5. **`riley`도 Kernel ID + Capability + Mode + Shape signature를 핵심 레지스트리 키로 가져야 한다.**

추천 key 예시:

```rust
pub struct KernelKey {
    pub op: OpId,
    pub device: DeviceArch,
    pub mode: ExecutionMode,
    pub dtype: DType,
    pub quant: Option<QuantFormat>,
    pub shape_class: ShapeClass,
    pub layout: LayoutId,
}
```

---

## 19. Canonical IR 제안

### 19.1 최상위 구성

```rust
pub struct ModelGraph {
    pub inputs: Vec<InputAdapterSpec>,
    pub embeddings: Vec<EmbeddingSpec>,
    pub blocks: Vec<BlockSpec>,
    pub final_norm: Option<NormSpec>,
    pub heads: Vec<HeadSpec>,
    pub tied_weights: Vec<WeightTie>,
    pub state: StateLayoutSpec,
}
```

### 19.2 Input adapter

```rust
pub enum InputAdapterSpec {
    Token(TokenInputSpec),
    ImagePatch(Patch2dSpec),
    VideoPatch(Patch3dSpec),
    AudioConv(AudioFrontendSpec),
    ContinuousValue(ValueProjectionSpec),
    Composite(Vec<InputAdapterSpec>),
}
```

### 19.3 Block

```rust
pub struct BlockSpec {
    pub pre_mixer_norm: Option<NormSpec>,
    pub mixer: MixerSpec,
    pub post_mixer_norm: Option<NormSpec>,
    pub channel_mixer: ChannelMixerSpec,
    pub residual_topology: ResidualTopology,
    pub layer_scale: Option<f32>,
    pub state_slot: Option<StateSlotId>,
}
```

### 19.4 Attention

```rust
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub qk_norm: Option<NormSpec>,
    pub position: PositionSpec,
    pub mask: MaskSpec,
    pub window: Option<u32>,
    pub projection_layout: ProjectionLayout,
    pub bias: AttentionBiasSpec,
    pub cache: Option<KvCacheSpec>,
}
```

### 19.5 Channel mixer

```rust
pub enum ChannelMixerSpec {
    DenseMlp(DenseMlpSpec),
    GatedMlp(GatedMlpSpec),
    Moe(MoeSpec),
    Convolution(ConvMixerSpec),
    None,
}
```

### 19.6 Position

```rust
pub enum PositionSpec {
    None,
    LearnedAbsolute(LearnedPositionSpec),
    Sinusoidal(SinusoidalSpec),
    Rope(RopeSpec),
    RelativeBias(RelativeBiasSpec),
    Alibi(AlibiSpec),
    Spatial2d(SpatialPositionSpec),
    Multimodal(MultimodalPositionSpec),
}
```

### 19.7 설계 원칙

- 모델 클래스 이름을 실행 그래프에 박지 않는다.
- shape와 semantic parameter를 잃지 않는다.
- inference-only IR과 checkpoint-import metadata를 분리한다.
- training-only dropout/loss/augmentation은 production graph에서 제거 가능해야 한다.
- 동일 op라도 prefill/decode/verify mode를 dispatch key에 포함한다.
- graph rewrite는 명시적 legality check 후 수행한다.

---

## 20. 통합하면 안 되는 경계

### 20.1 Norm placement

- pre-norm
- post-norm
- sandwich norm
- parallel residual

동일한 RMSNorm/LayerNorm kernel을 재사용할 수는 있지만 graph 순서를 합치면 안 된다.

### 20.2 QKV projection layout

- separate Q/K/V Linear
- packed QKV Linear
- packed Q + KV
- latent projection(MLA)
- bias 유무
- tensor-parallel shard 축

### 20.3 Position semantics

- hidden add와 Q/K rotation은 다르다.
- partial/interleaved RoPE는 full non-interleaved RoPE와 다르다.
- mRoPE는 1D RoPE로 단순 대체할 수 없다.
- relative bias는 cache position과 mask offset에 영향을 준다.

### 20.4 Attention mask

- causal
- bidirectional
- padding
- sliding window
- local/global hybrid
- arbitrary dense bias
- block sparse
- recurrent attention mask

Flash backend가 지원하지 않는 mask도 있으므로 capability check가 필요하다.

### 20.5 Cache semantics

- logical sequence length와 physical storage length가 다를 수 있다.
- sliding cache의 cumulative position은 남아 있어야 한다.
- cross-attention cache는 self cache와 update 정책이 다르다.
- sparse index cache나 MLA cache는 표준 K/V 두 tensor가 아니다.

### 20.6 Numerical behavior

- RMSNorm/softmax/router는 FP32 accumulation을 사용하는 경우가 많다.
- activation 근사식 차이는 autoregressive 결과를 바꿀 수 있다.
- top-k tie-breaking과 RNG는 재현성에 영향을 준다.
- fused kernel은 reference와 tolerance뿐 아니라 generated token parity도 확인해야 한다.

### 20.7 Training-only graph

다음은 추론 runtime의 초기 core에서 제외하거나 no-op 제거할 수 있다.

- dropout
- stochastic depth/DropPath
- SpecAugment
- gradient checkpointing
- auxiliary loss
- contrastive negative sampling
- training router jitter
- backward용 saved tensor

---

## 21. 소스 수준에서 확인된 반복 패턴

### 21.1 Llama

[`modeling_llama.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/llama/modeling_llama.py)

- RMSNorm
- RoPE
- separate Q/K/V/O projection
- GQA용 `repeat_kv`
- causal attention
- KV cache update
- gate/up/down SwiGLU MLP
- pre-norm residual

### 21.2 BERT

[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/bert/modeling_bert.py)

- token/position/type embedding
- LayerNorm + dropout
- bidirectional self-attention
- optional cross-attention/decoder mode
- post-norm residual
- GELU FFN
- 다양한 task head

### 21.3 T5

[`modeling_t5.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/t5/modeling_t5.py)

- RMS-style norm
- relative position bucket/bias
- encoder/decoder self-attention
- cross-attention
- dense 또는 gated FFN
- encoder-decoder cache

### 21.4 ViT와 CLIP

- [`ViT`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/vit/modeling_vit.py)
- [`CLIP`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/clip/modeling_clip.py)

공통:

- Conv2d patch embedding
- flatten/transpose
- learned position + interpolation
- LayerNorm pre-norm
- MHA + standard MLP
- residual

CLIP 추가:

- text/vision dual encoder
- projection
- L2 normalization
- image-text similarity matmul

### 21.5 ConvNeXt

[`modeling_convnext.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/convnext/modeling_convnext.py)

- depthwise Conv2d
- channel layout permutation
- LayerNorm
- pointwise expansion/contraction
- GELU
- layer scale
- residual + DropPath

### 21.6 Whisper와 Wav2Vec2

- [`Whisper`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/whisper/modeling_whisper.py)
- [`Wav2Vec2`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/wav2vec2/modeling_wav2vec2.py)

공통:

- Conv1d audio frontend
- normalization/activation
- Transformer attention/MLP

차이:

- Whisper는 encoder-decoder와 cross KV cache가 중요하다.
- Wav2Vec2는 raw waveform feature extraction, positional Conv, pretraining quantization/contrastive path가 있다.

### 21.7 EnCodec

[`modeling_encodec.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/encodec/modeling_encodec.py)

- causal/asymmetric Conv1d padding
- ConvTranspose1d
- dilated residual block
- LSTM
- Euclidean codebook search
- residual vector quantizer

LLM 중심 runtime과는 계산 구조가 크게 다르므로 plugin/extension 경계가 적절하다.

### 21.8 Mixtral과 DeepSeek-V3

- [`Mixtral`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/mixtral/modeling_mixtral.py)
- [`DeepSeek-V3`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)

공통:

- router projection
- top-k selection
- expert gated MLP
- weighted combine
- RMSNorm/RoPE 기반 decoder

DeepSeek 계열 추가:

- grouped routing
- shared experts
- specialized latent attention/cache

### 21.9 Mamba와 Jamba

- [`Mamba`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/mamba/modeling_mamba.py)
- [`Jamba`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/jamba/modeling_jamba.py)

- causal Conv1d/update
- selective scan/state update
- SiLU gating
- recurrent state
- Jamba는 attention, SSM, MoE를 layer schedule로 혼합

### 21.10 Qwen2-VL

[`modeling_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)

- LLM primitive 상당 부분 재사용
- Conv3d patch embedding
- vision attention/MLP
- patch merger
- temporal/height/width multimodal RoPE
- text/vision sequence 통합

### 21.11 FNet, Autoformer, DETR, TimeSformer

- [`FNet`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/fnet/modeling_fnet.py): FFT mixer + BERT형 MLP
- [`Autoformer`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/autoformer/modeling_autoformer.py): scaler, moving-average series decomposition, FFT autocorrelation, top-k delay aggregation, probabilistic output
- [`DETR`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/detr/modeling_detr.py): CNN backbone, 2D position, object query, cross-attention, detection heads
- [`TimeSformer`](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/timesformer/modeling_timesformer.py): frame patch, temporal/spatial attention, heavy reshape

이들은 “Transformer 외 long tail”을 IR이 어떻게 수용해야 하는지 확인하는 대표 사례다.

---

## 22. Riley 초기 분석 결론

### 22.1 첫 지원 범위

현재 프로젝트 방향과 가장 잘 맞는 초기 범위:

```text
Model families
- Llama-compatible
- Qwen-compatible

Execution
- NVIDIA single GPU
- BF16 first
- decoder-only causal generation
- small-batch interactive serving

Core
- RMSNorm
- RoPE variants needed by target models
- MHA/MQA/GQA
- causal KV cache
- SwiGLU
- LM head
- greedy/top-k/top-p sampling
```

### 22.2 첫 번째 공통 모듈 registry

```text
Embedding
Linear / GEMM dispatch
RMSNorm
LayerNorm
ResidualAdd
SiLU
GELU variants
GatedMLP
RoPE
AttentionPrefill
AttentionDecode
KVWrite
KVGather
CausalMask
LogitsProcessor
Sampler
```

### 22.3 첫 fusion 후보

```text
Residual + RMSNorm
Gate/Up + SiLU + Multiply
RoPE + KV Write
Paged KV Gather + Decode Attention
Logits Penalty + Temperature + Sampling
```

### 22.4 아직 구현하지 않고 더 분석해야 할 영역

1. vLLM, SGLang, TensorRT-LLM의 동일 primitive 경계 비교
2. 각 엔진의 KV layout와 block table 비교
3. FlashAttention/FlashInfer/CUTLASS/cuBLASLt의 지원 shape/ABI 비교
4. Qwen/Llama 계열의 checkpoint weight packing 차이
5. CUDA Graph capture와 dynamic batching의 충돌 지점
6. quantization format별 weight/cache metadata
7. Hugging Face kernel registry와 실제 backend fallback 조건
8. numerical parity 기준과 token-level differential test 설계

---

## 23. 향후 전체 AST census 설계안

현재 문서는 source-level stratified analysis다. 정확한 빈도표를 만들 때는 다음 pipeline이 적절하다.

```text
Transformers pinned revision
        ↓
model registry extraction
        ↓
modeling_*.py + modular_*.py AST
        ↓
class inheritance / copied-from / generated-file relation
        ↓
operator semantic normalization
        ↓
per-model canonical graph
        ↓
primitive frequency + subgraph frequency
        ↓
manual validation sample
```

### 23.1 반드시 해결할 중복 문제

단순 문자열 횟수는 다음 때문에 왜곡된다.

- generated `modeling_*.py`와 원본 `modular_*.py` 중복
- `Copied from` 코드 복제
- task head 클래스가 base model을 여러 번 참조
- TensorFlow/Flax/legacy 구현 혼재
- training-only code 포함
- helper 함수 정의와 실제 사용 차이
- 한 모델 family에 여러 config/variant 존재

### 23.2 권장 집계 단위

| 집계 단위 | 의미 |
|---|---|
| Model type | AutoConfig의 `model_type` |
| Base architecture | base `PreTrainedModel` graph |
| Task variant | LM/classification/QA 등 head |
| Block type | repeated layer의 canonical signature |
| Primitive | semantic op |
| Subgraph | fusion 가능한 연속 pattern |
| State | KV/recurrent/cache type |

### 23.3 출력 schema 예시

```json
{
  "revision": "c7cf04b1...",
  "model_type": "llama",
  "modalities": ["text"],
  "architecture": "decoder_only",
  "block_signature": {
    "norm": "rms_norm",
    "mixer": "gqa_attention",
    "position": "rope",
    "channel_mixer": "swiglu",
    "residual": "pre_norm_serial"
  },
  "state": "kv_cache",
  "primitives": [
    "embedding",
    "rms_norm",
    "linear",
    "rope",
    "attention",
    "silu",
    "mul",
    "residual_add"
  ]
}
```

### 23.4 필요한 검증

- AST 결과와 config-based graph가 일치하는가
- dynamic dispatch 때문에 놓친 branch가 없는가
- generated/modular 파일 중 하나만 집계했는가
- training-only op를 inference 빈도에서 제외했는가
- model family와 checkpoint variant를 혼동하지 않았는가

---

## 24. Benchmark와 검증 계획

### 24.1 Primitive correctness

- PyTorch/Transformers reference와 output 비교
- FP32, BF16, FP16 별 tolerance
- edge shape와 odd head dimension
- contiguous/non-contiguous input
- dynamic sequence length
- NaN/Inf behavior

### 24.2 Autoregressive correctness

- logit max absolute/relative error
- top-k set 일치
- greedy token exact match
- multi-token generation exact match 또는 divergence 시점
- KV cache on/off 결과 일치
- prefill 전체 실행과 chunked prefill 일치

### 24.3 Microbenchmark

- latency distribution, 최소/중앙/p95
- achieved bandwidth/FLOPs
- kernel launch 수
- temporary allocation 수/bytes
- CUDA Graph capture/replay 가능 여부

### 24.4 End-to-end matrix

| 차원 | 값 |
|---|---|
| Engine | riley, vLLM, SGLang, TensorRT-LLM |
| Concurrency | 1, 2, 4, 8, 16 |
| Prompt length | 128, 1K, 4K, 8K |
| Output length | 32, 128, 512 |
| Metrics | TTFT, median TPOT, p95/p99 ITL, E2E, tokens/s, GPU/CPU utilization, VRAM, scheduler time, launch count |

초기 성공 기준은 모든 조건에서 승리하는 것이 아니라, **명확히 정의된 small-batch interactive workload에서 host overhead 또는 latency 우위가 재현되는 것**이다.

---

## 25. 제한 사항과 불확실성

1. 이 문서는 registry를 통해 전체 scope를 확인하고, 주요 구조군을 층화해 source를 검토한 분석이다. 모든 `modeling_*.py`를 자동 AST로 완전 집계한 통계 보고서는 아니다.
2. 클래스/파일 출현 빈도는 실제 사용 checkpoint 수나 시장 사용량과 다르다.
3. 한 모델 구현 안에도 config에 따라 attention, activation, norm, cache branch가 달라질 수 있다.
4. Transformers eager reference의 모듈 경계가 최적 inference kernel 경계와 항상 같지는 않다.
5. 최신 모델의 generated `modeling_*.py`와 `modular_*.py` 관계를 중복 처리하지 않으면 빈도 분석이 왜곡된다.
6. GPU별 최적 kernel은 architecture, shape, dtype, batch, sequence length에 따라 달라진다.
7. fusion은 microbenchmark가 빨라도 register pressure, occupancy, graph capture, workspace 때문에 end-to-end에서 손해일 수 있다.
8. MoE/SSM/multimodal은 빠르게 변화하는 영역이므로 IR 확장성을 우선하고 초기 구현 범위는 좁게 유지해야 한다.

---

## 26. 최종 결론

Transformers의 모델 수가 많다는 사실은 `riley`가 모델마다 별도 실행 엔진을 만들어야 한다는 뜻이 아니다. 소스 구조를 연산 의미로 정규화하면 다음과 같은 작은 중심부가 드러난다.

```text
Common runtime
├─ Tensor / layout / dtype / memory
├─ Embedding
├─ GEMM dispatch
├─ Normalization
├─ Activation / gating
├─ Residual
├─ Attention interface
├─ Position interface
├─ State/cache interface
└─ Head / sampling interface
```

현대 decoder LLM의 중심은 더 좁다.

```text
Embedding
→ RMSNorm
→ QKV
→ RoPE
→ GQA/MQA/MHA
→ KV cache
→ Output projection
→ Residual
→ RMSNorm
→ SwiGLU
→ Residual
→ LM head
→ Sampling
```

따라서 초기 전략은 다음이 가장 합리적이다.

1. **모델 이름이 아니라 canonical primitive와 subgraph를 정의한다.**
2. **GEMM은 vendor stack을 활용하고, runtime/layout/cache/fusion에 집중한다.**
3. **RMSNorm, RoPE, GQA decode, KV cache, SwiGLU, sampling을 첫 공통 최적화 축으로 삼는다.**
4. **MoE, SSM, multimodal은 별도 extension이 아니라 처음부터 표현 가능한 IR variant로 설계하되, 구현은 뒤로 미룬다.**
5. **정확한 frequency는 이후 AST census로 검증하고, 그 전에는 공통성에 과도한 숫자를 붙이지 않는다.**
6. **성능 판단은 kernel 단독이 아니라 TTFT, TPOT, p99, GPU idle, CPU overhead, VRAM, launch count의 end-to-end 조합으로 한다.**

이 방향이라면 `riley`는 “Transformers 모델을 Rust로 일일이 다시 작성하는 프로젝트”가 아니라, **Transformers 생태계의 반복 계산 구조를 정규화하고 가장 가치 있는 실행 경로를 Rust+CUDA로 최적화하는 inference runtime**으로 정의할 수 있다.

---

## 27. 주요 소스

### Registry 및 공통 인프라

- [Transformers repository](https://github.com/huggingface/transformers/tree/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc)
- [Auto configuration mapping](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/auto/configuration_auto.py)
- [Auto mappings](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/auto/auto_mappings.py)
- [Shared modeling layers](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/modeling_layers.py)
- [Cache utilities](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/cache_utils.py)
- [RoPE utilities](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/modeling_rope_utils.py)
- [Hub kernel integration](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/integrations/hub_kernels.py)

### 대표 모델 구현

- [Llama](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/llama/modeling_llama.py)
- [BERT](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/bert/modeling_bert.py)
- [T5](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/t5/modeling_t5.py)
- [ViT](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/vit/modeling_vit.py)
- [CLIP](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/clip/modeling_clip.py)
- [ConvNeXt](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/convnext/modeling_convnext.py)
- [TimeSformer](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/timesformer/modeling_timesformer.py)
- [Whisper](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/whisper/modeling_whisper.py)
- [Wav2Vec2](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/wav2vec2/modeling_wav2vec2.py)
- [EnCodec](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/encodec/modeling_encodec.py)
- [Qwen2-VL](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
- [Mixtral](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/mixtral/modeling_mixtral.py)
- [DeepSeek-V3](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)
- [Mamba](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/mamba/modeling_mamba.py)
- [Jamba](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/jamba/modeling_jamba.py)
- [FNet](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/fnet/modeling_fnet.py)
- [Autoformer](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/autoformer/modeling_autoformer.py)
- [DETR](https://github.com/huggingface/transformers/blob/c7cf04b1e3b1d497dbb1473c2e65e75ee69e12dc/src/transformers/models/detr/modeling_detr.py)
