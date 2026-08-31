# C02-P0 — Effective Runtime Configuration Receipt

**목적:** C02 candidate qualification 전에, cold prepare 이후 실제로 선택된
runtime 구성을 canonical 증적으로 노출한다.

**순서:** `C01 → C02-P0 → C02 → C03`. C02-P0가 merge/검증되기 전에는 RC3
freeze나 C03 작업을 시작하지 않는다.

## 문제와 계약 정정

기존 C02 초안은 live `/v1/config`와 startup artifact에 `freeze_sha256`와
Gate-E report SHA를 직접 넣도록 요구했다. 두 값은 freeze 이후 또는 Gate-E
이후에야 생기므로, exact argv/environment hash가 freeze에 포함되는 구조와
결합하면 self-reference가 된다.

따라서 live 증적은 아래 다섯 top-level field만 가진다.

```text
schema_version
candidate_id
runtime_identity { configuration_profile, configuration_sha256 }
effective_config
effective_config_sha256
```

`freeze_sha256`와 `base_release_candidate_report_sha256`는 live endpoint나
startup artifact에 넣지 않는다. 둘은 raw capture 뒤에 C02 semantic checker가
freeze와 replayed Gate E report에 결합한다. 이 순서는 다음과 같이 acyclic이다.

```text
frozen argv/env + candidate/profile
  → cold prepare → canonical endpoint + create-only artifact
  → Gate E replay + C02 semantic report → RC3 decision
```

server는 외부에서 받은 configuration SHA를 echo하지 않고, sanitize된 exact
launch argv/environment으로부터 직접 계산해야 한다. `freeze SHA`, `Gate-E SHA`,
또는 자기 configuration SHA를 같은 argv/environment에 주입하면 안 된다.

## Launch identity

C02-P0 mode는 candidate ID, configuration profile, startup-artifact absolute path를
all-or-none launch inputs로 받는다. 이 값들은 freeze 전에 정해져 있으므로 frozen
argv/environment에 포함될 수 있다. 반면 configuration SHA는 input이 아니라 server가
실제 launch에서 재계산한 output이다. 구현은 executable을 제외하고 `serve`를 포함한
argv와, `env -i`로 명시한 UTF-8 environment map의 canonical JSON을 동일한 방식으로
hash해야 한다. unknown/non-UTF-8/중복 환경 또는 partial C02-P0 identity는 fail closed
한다.

## 범위

- `GET /v1/config`을 추가한다. cold prepare가 성공한 경우에만 미리 만든
  canonical bytes를 반환하고, 증적이 없으면 fabricated default 대신 `503`을
  반환한다. `POST /v1/config`은 `405`이다.
- `CompletionBackend` 경계에 immutable effective-config body accessor를 둔다.
  CPU mock은 기본 `None`으로 유지한다.
- prepared executor/engine에서 실제 선택값을 snapshot한다. CLI echo가 아니라
  prepared shape bucket, KV layout, attention/fallback selection, GEMM aggregate,
  sampling/metadata/completion policy를 사용한다.
- startup에서 같은 endpoint bytes를 포함한 canonical artifact를 `O_EXCL`로
  create-only 기록한다.
- endpoint와 artifact는 raw C02 evidence이고, C02 semantic report만
  candidate/freeze/replayed-Gate-E 바인딩을 가진다.

## 비범위

- CUDA Graph, kernel fusion, scheduler 의미 변경, default 성능 변경
- Gate-E threshold 완화 또는 기존 evidence 재사용
- C03 fuzz 또는 이후 roadmap 구현

## 검증

- CPU route: canonical body byte-identical 반복 GET, unavailable `503`, POST `405`.
- CPU identity: parsed launch argv/environment의 SHA가 endpoint identity와 같고,
  forbidden self-referential attestation values는 거부.
- CPU prepared-fact: fixed-max bucket collapse, KV layout, attention fallback,
  GEMM aggregate가 prepared runtime state와 정확히 일치.
- CPU artifact: embedded endpoint bytes/digest, `O_EXCL`, replacement 거부.
- Remote: 두 arm을 독립 launch하여 endpoint/artifact capture 후 C02 checker의
  dual-arm semantic replay를 통과.

## 롤백

C02-P0는 observability/evidence surface만 추가한다. rollback은 endpoint와
startup-artifact emission을 함께 제거하고, C02 candidate는 새 freeze로 처음부터
재생성한다. partial RC3 evidence는 promotion 근거로 재사용하지 않는다.
