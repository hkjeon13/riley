# PR 01 correctness evidence

작은 oracle comparison report는 review와 임계값 provenance를 위해 이
디렉터리에 exact bytes로 보존한다. 대형 tensor sidecar와 manifest는 Git에
넣지 않으며 report의 `inputs` SHA-256과 아래 raw bundle 위치로 결합한다.

## V1 calibration basis

- report: `smollm2-fp32-bf16-native-e0-v1-failed-oracle-report.json`
- SHA-256: `ca13c033af2ddce5cfbf280fc1f4d2f95d0cba0e242bda8c59f2592946cec726`
- result: 31 cases 중 12 failures, semantic self-check pass
- source revision: `8ab7490bfdf9efd1d7c7d831204b8e67c0c7c5b9`

V1 report는 v2 threshold의 사전 calibration 근거일 뿐 activation evidence가
아니다. Active gate와 contract validator가 이 파일의 path, size, SHA-256,
identity, source revision과 aggregate metrics를 직접 교차검증한다.

## V2 activation replay

- report: `smollm2-fp32-bf16-native-e0-v2-passing-oracle-report.json`
- SHA-256: `1fd064d780868ed76202b9adbd773f2ef76cc54a35551a92145f882d779871ea`
- result: 31/31 pass, numeric pass, semantic self-check pass
- `e0_candidate_evidence`: `false`
- source revision: `2d22ca061f601389fad7f45708497daad14d9297`
- matrix SHA-256: `a979659ef9d7b3c5a7a85e423347eb6f06ccbd3ae5a370056bd056d3137c7e87`
- v2 gate SHA-256: `eb97b2011bd77e6b2bfdb039c846484e281b35108ba6b357cdd1aba7033479e9`

Raw replay bundle은
`server-4096:/home/psyche/riley-artifacts/pr01/2d22ca061f601389fad7f45708497daad14d9297/oracles/`
에 보존한다.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `fp32-manifest.json` | 63,399 | `5def01c42a0bc54a06b3936638dcd0ddba8c2c68d58b1a022d5c683af0454f27` |
| `fp32.safetensors` | 130,136,288 | `95fac531e9a08b5f8c184441078829ce2c4bea2a286525035b95205a4bca891a` |
| `bf16-manifest.json` | 125,589 | `40ad7ae1734b02b0c5533ba7bbaba48f6a5629b0c4b66a3f39178ceb5ca2592f` |
| `bf16.safetensors` | 68,121,432 | `598bf7255c5aab1ff6701784992b22e55ee4d85e4247dd8fe2cafa49f1a49b66` |
| `oracle-calibration-report.json` | 48,801 | `1fd064d780868ed76202b9adbd773f2ef76cc54a35551a92145f882d779871ea` |

이 raw bundle은 PR 01 release evidence가 별도 장기 보관소로 이관될 때까지
삭제하거나 덮어쓰지 않는다. 재검증 명령은
`tools/python/reference/CALIBRATION.md`의 `calibrate-validate-manifest`와
`calibrate-validate-oracles` 절차를 따른다.

V2 gate, matrix, manifests, sidecars와 report는 v3로 재작성하지 않는 frozen
oracle lineage다. PR16의 별도 `smollm2-fp32-bf16-native-e0-v3` release gate는
이 exact v2 oracle만 허용하고 `canonical-v1` candidate 하나를 검증한다.
Fixed37 결과는 optimizer regression diagnostic으로 남지만 첫 release candidate
승인 evidence가 아니다.
