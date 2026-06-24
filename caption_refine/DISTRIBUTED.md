# caption_refine 분산 실행 가이드

다른 서버에서 GPU 10장에 분산해 `caption_refine`을 실행하는 방법을 설명합니다.

---

## 아키텍처 개요

```
[다른 서버]

  GPU 0  →  vLLM :8000  ←  worker shard 0  (클립 0, 10, 20, ...)
  GPU 1  →  vLLM :8001  ←  worker shard 1  (클립 1, 11, 21, ...)
  GPU 2  →  vLLM :8002  ←  worker shard 2  (클립 2, 12, 22, ...)
  ...
  GPU 9  →  vLLM :8009  ←  worker shard 9  (클립 9, 19, 29, ...)
```

- GPU 1장당 vLLM 서버 1개 (포트 8000~8009)
- 클립 목록을 **스트라이드 방식**으로 분할 — 각 워커가 균등하게 담당
- 진행 파일이 샤드별 독립 저장 → 워커가 개별 중단/재시작 가능

---

## 환경변수 전체 목록

### 데이터 경로

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `CR_DATA_ROOT` | `/Data1/home/bskang/cds-data` | 데이터 루트. **이것 하나만 바꾸면 하위 경로 자동 변경** |
| `CR_VIDEOS_DIR` | `$CR_DATA_ROOT/front_camera_videos` | 비디오 파일 폴더 |
| `CR_CAPTIONS_DIR` | `$CR_DATA_ROOT/captions` | 원본 캡션 폴더 |
| `CR_OUTPUT_ROOT` | `$CR_DATA_ROOT/caption_v2` | 출력 루트 |
| `CR_ODD_OUT_DIR` | `$CR_OUTPUT_ROOT/odd` | ODD JSON 출력 |
| `CR_CAPTION_OUT_DIR` | `$CR_OUTPUT_ROOT/captions` | 정제 캡션 출력 |
| `CR_DIFF_OUT_DIR` | `$CR_OUTPUT_ROOT/diff` | diff JSON 출력 |
| `CR_INDEX_DIR` | `/Data1/home/bskang/AVdata-distirbution/data/index` | clip_ids.json 폴더 |
| `CR_SANFLOW_GAP_PATH` | `.../sanflow_gaps.json` | gap 클립 목록 JSON |

### vLLM 연결

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `CR_VLLM_URL` | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `CR_VLLM_MODEL` | `nvidia/Cosmos-Reason2-2B` | 모델 이름 |
| `CR_VLLM_APIKEY` | `EMPTY` | API 키 |

### 처리 파라미터

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `CR_CONCURRENT` | `2` | 샤드당 동시 처리 클립 수 |
| `CR_VIDEO_MODE` | `frames` | `frames`(16장 이미지) 또는 `video`(MP4 통째) |
| `CR_NUM_FRAMES` | `16` | 추출 프레임 수 |
| `CR_CONF_THRESHOLD` | `0.7` | Stage 3 재검증 confidence 기준 |

---

## 실행 방법

### 방법 1: `run_distributed.sh` 스크립트 (권장)

```bash
cd /path/to/cosmos-reason2

# vLLM이 이미 실행 중인 경우
CR_DATA_ROOT=/mnt/cds-data \
CR_INDEX_DIR=/mnt/avdata/index \
CR_SANFLOW_GAP_PATH=/mnt/avdata/experiments/EXP-002/results/sanflow_gaps.json \
SOURCE=all CONCURRENT=4 \
bash caption_refine/run_distributed.sh

# vLLM 서버도 함께 기동하는 경우
VLLM_MODEL=nvidia/Cosmos-Reason2-7B \
bash caption_refine/run_distributed.sh --start-vllm
```

GPU 수, 포트, 동시 처리 수 조정:

```bash
TOTAL_GPUS=4 VLLM_BASE_PORT=9000 CONCURRENT=6 SOURCE=longtail \
bash caption_refine/run_distributed.sh
```

### 방법 2: 수동 실행

#### Step 1 — vLLM 서버 10개 기동

```bash
mkdir -p logs

VIDEOS_DIR="${CR_DATA_ROOT:-/Data1/home/bskang/cds-data}/front_camera_videos"

for i in $(seq 0 9); do
    CUDA_VISIBLE_DEVICES=$i vllm serve nvidia/Cosmos-Reason2-2B \
        --port $((8000+i)) \
        --dtype auto \
        --max-model-len 16384 \
        --allowed-local-media-path "$VIDEOS_DIR" \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --reasoning-parser qwen3 \
        > logs/vllm_gpu${i}.log 2>&1 &
    echo "GPU $i → port $((8000+i)) (PID $!)"
done

# 서버 기동 대기
sleep 60
```

#### Step 2 — 워커 10개 실행

```bash
export CR_DATA_ROOT=/mnt/cds-data
export CR_INDEX_DIR=/mnt/avdata/index
export CR_SANFLOW_GAP_PATH=/mnt/avdata/experiments/EXP-002/results/sanflow_gaps.json

for i in $(seq 0 9); do
    CR_VLLM_URL="http://localhost:$((8000+i))/v1" \
    uv run python -m caption_refine.batch_runner \
        --source all \
        --shard-index $i \
        --total-shards 10 \
        --concurrent 4 \
        > logs/worker_shard${i}.log 2>&1 &
    echo "Shard $i → port $((8000+i)) (PID $!)"
done

wait && echo "All done"
```

---

## `batch_runner` 주요 인자

```
--source {gap,longtail,all}   클립 소스 (default: gap)
--ids-file PATH               clip_id 목록 JSON 파일 (--source 무시)
--limit N                     처리할 최대 클립 수
--concurrent N                동시 처리 클립 수 (default: CR_CONCURRENT=2)
--shard-index N               이 워커의 샤드 번호 (0-based)
--total-shards N              전체 워커 수 (default: 1 = 분산 없음)
--vllm-url URL                vLLM 서버 URL (CR_VLLM_URL 환경변수보다 우선)
--reset                       진행 기록 초기화 후 처음부터 재시작
```

---

## 진행 상황 확인

### 로그 실시간 확인

```bash
# 특정 샤드 로그
tail -f logs/worker_shard0.log

# 모든 샤드 진행률 한눈에
grep -h "Progress:" logs/worker_shard*.log | tail -n 20
```

### 진행 파일 확인

샤드별 진행 파일: `$CR_OUTPUT_ROOT/progress_shard00of10.json`, `progress_shard01of10.json`, ...

```bash
# 전체 완료/에러 수 요약
for f in /mnt/cds-data/caption_v2/progress_shard*.json; do
    done_count=$(python3 -c "import json; d=json.load(open('$f')); print(len(d['done']))")
    err_count=$(python3 -c  "import json; d=json.load(open('$f')); print(len(d['error']))")
    echo "$(basename $f): done=$done_count  error=$err_count"
done
```

---

## 중단 후 재시작

진행 파일(`progress_shard*.json`)을 기반으로 완료된 클립은 자동 스킵됩니다.  
중단된 워커만 다시 실행하면 됩니다.

```bash
# shard 3만 재시작
CR_DATA_ROOT=/mnt/cds-data \
CR_VLLM_URL="http://localhost:8003/v1" \
uv run python -m caption_refine.batch_runner \
    --source all \
    --shard-index 3 \
    --total-shards 10 \
    --concurrent 4
```

처음부터 다시 처리하려면 `--reset` 플래그 추가 (주의: 해당 샤드 진행 기록 삭제):

```bash
CR_VLLM_URL="http://localhost:8003/v1" \
uv run python -m caption_refine.batch_runner \
    --source all --shard-index 3 --total-shards 10 --reset
```

---

## 출력 파일 구조

```
$CR_OUTPUT_ROOT/
├── captions/{clip_id}.camera_front_wide_120fov.txt   # 정제 캡션
├── odd/{clip_id}.json                                 # 구조화 ODD
├── diff/{clip_id}_diff.json                           # 변경 내역
├── progress_shard00of10.json                          # shard 0 진행
├── progress_shard01of10.json                          # shard 1 진행
└── ...
```

---

## 단일 서버 실행 (기존 방식)

분산 없이 단일 vLLM 서버 + 단일 워커로 실행할 때는 기존과 동일합니다.

```bash
export CR_VLLM_URL="http://localhost:8000/v1"
uv run python -m caption_refine.batch_runner --source gap --concurrent 4
```

`--shard-index`, `--total-shards` 생략 시 분산 없이 동작합니다.
