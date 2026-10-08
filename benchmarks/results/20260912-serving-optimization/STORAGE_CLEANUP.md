# 로컬 원자료 용량 정리 — 2026-09-16

폴더의 파일 크기 합계는 약 **10.27GiB → 3.77GiB**로 줄었으며,
약 **6.50GiB**를 확보했다. 응답 원자료 아카이브는 약 **678MiB**다.

사용자 요청에 따라 `raw/`의 `*-retained-rows.json` 1,113개를
[`retained-rows.tar.gz`](raw/storage-cleanup-20260916/retained-rows.tar.gz)에 무손실 압축했다.
아카이브를 다시 읽어 모든 원문 SHA256과 크기를 확인한 뒤 해당 원본만 제거했다.
대응하는 `.nsys-rep`가 있는 재생성 가능한 `.sqlite` 55개도 제거했다.
결과 문서, 집계·분석 JSON, 소스, 검증 receipt, Nsight 원본 보고서와 기존 아카이브는 보존한다.

[`inventory.json`](raw/storage-cleanup-20260916/inventory.json)은 원래 경로·크기·SHA256,
압축 아카이브의 SHA256, 제거한 SQLite와 원본 보고서의 대응 관계를 기록한다.
기존 실험 manifest와 `LOCAL_ARTIFACTS_20260913.jsonl`은 당시 원문을 가리키는
역사적 기록으로 유지한다. 압축으로 이동한 응답 원자료의 현재 위치는 이 inventory를 따른다.
기존 분석 도구에서 원래 JSON 경로가 필요하면 먼저 아래 도구로 복원한다.

## 확인과 복원

캠페인 디렉터리에서 실행한다. 복원은 추가 측정이나 서버 실행을 하지 않는다.

```sh
python3 restore_cleaned_artifacts.py --list
python3 restore_cleaned_artifacts.py --verify
python3 restore_cleaned_artifacts.py --restore 'raw/serving-round62/*'
```

원래 위치 대신 별도 디렉터리에 일부 원자료를 복원할 수도 있다.
동일한 내용이 이미 있으면 유지하고, 다른 내용이 있으면 덮어쓰지 않는다.

```sh
python3 restore_cleaned_artifacts.py --restore 'raw/serving-round62/*' --destination /tmp/riley-evidence
```

전체 원자료를 복원하려면 `--restore '*'`를 사용한다. 복원 후에는 기존 manifest의
원문 경로·크기·SHA256으로 검증할 수 있다. 다시 압축 원본을 삭제하기 전에는
inventory와 아카이브 검증을 수행해야 한다.

SQLite는 Nsight Systems가 설치된 환경에서 보존한 보고서를 export하여 재생성한다.
아래 `path/to/profile`은 inventory의 `source_report`에서 `.nsys-rep`를 뺀 경로다.

```sh
nsys export --type sqlite --output path/to/profile.sqlite path/to/profile.nsys-rep
```

재생성된 SQLite 파일 자체의 체크섬은 Nsight 버전에 따라 달라질 수 있다.
원본 `.nsys-rep`의 SHA256을 inventory와 먼저 비교한다.

Git에서 추적하던 원본 JSON은 삭제 상태로 표시된다. 아카이브는 기존
`.tar.gz` 제외 규칙을 따르는 로컬 파일이며 Git에 업로드되지 않는다.
커밋·푸시하지 않았고 `.git`의 과거 이력도 정리하지 않았다.
따라서 이 아카이브와 inventory를 함께 보존해야 한다.
