# AV-Caption: 자율주행 영상 캡션 생성 파이프라인

자율주행(AV) 전방 카메라 영상 클립에 대해 고품질 자연어 캡션을 생성하는 파이프라인입니다.
[NVIDIA Cosmos-Reason2](https://huggingface.co/collections/nvidia/cosmos-reason2) VLM을 백엔드로 사용합니다.

______________________________________________________________________

**Table of Contents**

- [Overview](#overview)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [가상환경 설치](#가상환경-설치)
  - [NVIDIA 라이브러리 추가 설치](#nvidia-라이브러리-추가-설치)
- [실행 방법](#실행-방법)
  - [1. vLLM 서버 시작](#1-vllm-서버-시작)
  - [2. 단일 영상 추론 (빠른 테스트)](#2-단일-영상-추론-빠른-테스트)
  - [3. 배치 캡션 생성 — Online 모드](#3-배치-캡션-생성--online-모드)
  - [4. 배치 캡션 생성 — Offline 모드](#4-배치-캡션-생성--offline-모드)
  - [5. Shard 파일 생성 (분산 처리)](#5-shard-파일-생성-분산-처리)
  - [6. Caption Refinement v1 (4-stage)](#6-caption-refinement-v1-4-stage)
  - [7. Caption Refinement v2 (Motion-Aware, 5-stage)](#7-caption-refinement-v2-motion-aware-5-stage)
  - [8. 분산 실행](#8-분산-실행)
- [환경변수 레퍼런스](#환경변수-레퍼런스)
- [파이프라인 상세](#파이프라인-상세)
  - [caption_refine v1](#caption_refine-v1)
  - [caption_refine v2](#caption_refine-v2)
- [디렉토리 구조](#디렉토리-구조)

______________________________________________________________________

## Overview

두 가지 캡션 생성·정제 파이프라인을 제공합니다.

| 파이프라인 | 스테이지 수 | 특징 |
|---|---|---|
| `caption_refine` v1 | 4-stage | 기존 캡션 기반 Hallucination 제거 + ODD 추출 + 정제 |
| `caption_refine_v2` | 5-stage | 센서(egomotion) 선확정 + Motion-Aware ODD + Cross-Validation |

## Setup

### Prerequisites

- Python **3.10** 이상
- CUDA **12.8** 또는 **13.0**
- NVIDIA GPU (추천: H100 24GB 이상)
- `uv` 패키지 매니저

### 가상환경 설치

**1) 시스템 패키지 설치**

일반 환경 (`sudo` 사용):

```shell
sudo apt-get install curl ffmpeg git git-lfs unzip --fix-missing
```

Clunix 컨테이너 환경 (`gcsudo` 사용):

```shell
gcsudo apt-get install curl ffmpeg git git-lfs unzip --fix-missing
```

**2) uv 설치**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 PATH 적용 방법은 환경에 따라 다릅니다:

- **일반 환경**: 설치 스크립트가 자동으로 `~/.bashrc`에 추가하므로 아래 명령으로 즉시 적용합니다.
  ```shell
  source $HOME/.local/bin/env
  ```
- **Clunix 컨테이너 환경**: `env` 파일이 생성되지 않으므로 PATH를 직접 추가합니다.
  ```shell
  export PATH="$HOME/.local/bin:$PATH"
  ```

**3) 저장소 클론**

```shell
git clone https://github.com/bskang8/AV-Caption.git
cd AV-Caption
```

**4) Python 가상환경 생성 및 패키지 설치**

CUDA 12.8 (H100, A100 등):

```shell
uv sync --extra cu128
```

CUDA 13.0 (DGX Spark, Jetson AGX Thor):

```shell
uv sync --extra cu130
```

**5) 가상환경 활성화**

```shell
source .venv/bin/activate
```

> 이후 모든 명령은 가상환경이 활성화된 상태에서 실행하거나, `uv run python ...` 형태로 실행합니다.

### NVIDIA 라이브러리 추가 설치

vLLM 실행에 필요한 NVIDIA 라이브러리입니다.

**CUDA 12.8** — `uv sync --extra cu128` 후 별도 설치 필요:

```shell
uv pip install nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12
```

**CUDA 13.0** — `uv sync --extra cu130` 실행 시 자동 설치되므로 별도 설치 불필요.

> `start_vllm.sh`가 누락된 라이브러리를 자동으로 감지하고 안내합니다.

### Hugging Face 모델 접근 설정

```shell
uvx hf auth login
```

> 또는 환경변수: `export HF_TOKEN="hf_..."`

______________________________________________________________________

## 실행 방법

### 1. vLLM 서버 시작

```shell
bash start_vllm.sh [GPU번호] [포트]
```

예시:

```shell
bash start_vllm.sh 0 8000   # GPU 0, 포트 8000
bash start_vllm.sh 1 8001   # GPU 1, 포트 8001
```

환경변수 오버라이드:

```shell
VLLM_MODEL=nvidia/Cosmos-Reason2-8B \
VIDEOS_DIR=/path/to/videos \
bash start_vllm.sh 0 8000
```

서버 준비 확인:

```shell
watch -n5 'curl -sf http://localhost:8000/health && echo OK'
```

서버 종료:

```shell
kill $(pgrep -f "vllm serve")
```

---

### 2. 단일 영상 추론 (빠른 테스트)

```shell
cosmos-reason2-inference online --port 8000 \
    -i prompts/caption_detail.yaml \
    --videos assets/sample.mp4 --fps 4
```

---

### 3. 배치 캡션 생성 — Online 모드

vLLM 서버를 미리 띄워 놓고 여러 영상을 순차 처리합니다.

**디렉토리 전체 처리:**

```shell
python batch_caption.py \
    --input-dir /Data1/home/bskang/cds-data/front_camera_videos \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8000 --fps 4 --skip-existing
```

**Shard 파일 지정 (분산 처리):**

```shell
# GPU 0 — 서버 포트 8000
python batch_caption.py \
    --video-list shards/shard_00.txt \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8000 --skip-existing

# GPU 1 — 서버 포트 8001 (별도 터미널)
python batch_caption.py \
    --video-list shards/shard_01.txt \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8001 --skip-existing
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input-dir` | `/Data1/home/bskang/cds-data/front_camera_videos` | 영상 디렉토리 |
| `--output-dir` | (필수) | 캡션 .txt 저장 디렉토리 |
| `--video-list` | — | Shard 파일 경로 (있으면 input-dir 무시) |
| `--prompt-file` | `prompts/caption_detail.yaml` | 프롬프트 파일 |
| `--port` | `8000` | vLLM 서버 포트 |
| `--fps` | `4` | 초당 샘플링 프레임 수 |
| `--skip-existing` | off | 이미 완료된 파일 건너뜀 |

---

### 4. 배치 캡션 생성 — Offline 모드

외부 서버 등 vLLM HTTP 서버 없이 모델을 직접 메모리에 올려 처리합니다.

```shell
CUDA_VISIBLE_DEVICES=0 python batch_caption_offline.py \
    --video-list shards/shard_00.txt \
    --output-dir /local/captions \
    --skip-existing
```

외부 서버로 rsync한 shard를 로컬 경로로 재매핑:

```shell
CUDA_VISIBLE_DEVICES=0 python batch_caption_offline.py \
    --video-list shards/shard_02.txt \
    --input-dir /mnt/nas/front_camera_videos \
    --output-dir /local/captions \
    --skip-existing
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `nvidia/Cosmos-Reason2-2B` | 모델명 또는 로컬 경로 |
| `--video-list` | — | Shard 파일 (없으면 input-dir 전체) |
| `--input-dir` | 자동 탐색 | 영상 디렉토리 (경로 재매핑 시 사용) |
| `--output-dir` | (필수) | 캡션 .txt 저장 디렉토리 |
| `--fps` | `4` | 초당 샘플링 프레임 수 |
| `--max-model-len` | `16384` | 최대 컨텍스트 길이 |
| `--max-tokens` | `4096` | 최대 생성 토큰 수 |
| `--skip-existing` | off | 이미 완료된 파일 건너뜀 |

---

### 5. Shard 파일 생성 (분산 처리)

여러 GPU / 서버에 영상을 나눠 처리하기 위한 shard 파일을 생성합니다.

```shell
python generate_shards.py --num-shards 10 --output-dir shards/
```

미리보기만 (파일 미생성):

```shell
python generate_shards.py --num-shards 10 --dry-run
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input-dir` | `/Data1/home/bskang/cds-data/front_camera_videos` | 영상 디렉토리 |
| `--captions-dir` | `/Data1/home/bskang/cds-data/captions` | 이미 완료된 캡션 디렉토리 |
| `--num-shards` | `10` | 분할 수 |
| `--output-dir` | `shards/` | shard 파일 저장 위치 |

---

### 6. Caption Refinement v1 (4-stage)

기존 캡션을 기반으로 Hallucination을 제거하고 ODD를 추출하여 정제합니다.

```shell
# gap 클립 처리 (기본)
uv run python -m caption_refine.batch_runner --source gap

# 전체 클립 처리 (최대 100개)
uv run python -m caption_refine.batch_runner --source all --limit 100

# 특정 클립 ID 목록 처리
uv run python -m caption_refine.batch_runner --ids-file my_clips.json --concurrent 4
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `gap` | `gap` / `longtail` / `all` |
| `--ids-file` | — | clip_id 목록 JSON 파일 |
| `--limit` | — | 처리 클립 수 제한 |
| `--concurrent` | `2` | 동시 처리 수 |
| `--reset` | off | 진행 상태 초기화 후 재처리 |

---

### 7. Caption Refinement v2 (Motion-Aware, 5-stage)

센서(egomotion) 데이터를 선확정하여 더 정확한 ODD와 캡션을 생성합니다.

```shell
# gap 클립 처리 (기본)
uv run python -m caption_refine_v2.batch_runner --source gap

# 전체 처리, 동시 4개
uv run python -m caption_refine_v2.batch_runner --source all --concurrent 4

# 특정 클립 ID 목록
uv run python -m caption_refine_v2.batch_runner \
    --ids-file my_clips.json --concurrent 4
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `gap` | `gap` / `longtail` / `all` |
| `--ids-file` | — | clip_id 목록 JSON 파일 |
| `--limit` | — | 처리 클립 수 제한 |
| `--concurrent` | `2` | 동시 처리 수 |
| `--shard-index` | `0` | 분산 샤드 인덱스 |
| `--total-shards` | `1` | 분산 샤드 총 수 |
| `--vllm-url` | `$CR_VLLM_URL` | vLLM 엔드포인트 URL |
| `--reset` | off | 진행 상태 초기화 후 재처리 |

---

### 8. 분산 실행

**vLLM 서버가 이미 실행 중인 경우 (워커만 실행):**

```shell
bash caption_refine/run_distributed.sh
```

**vLLM 서버도 함께 시작:**

```shell
bash caption_refine/run_distributed.sh --start-vllm
```

**GPU 수, 소스, 동시 처리 수 오버라이드:**

```shell
TOTAL_GPUS=4 SOURCE=longtail CONCURRENT=6 \
bash caption_refine/run_distributed.sh
```

**v2 수동 분산 실행 예시 (GPU 4개):**

```shell
for i in 0 1 2 3; do
    CR_VLLM_URL="http://localhost:$((8000 + i))/v1" \
    uv run python -m caption_refine_v2.batch_runner \
        --source all \
        --shard-index $i --total-shards 4 \
        --concurrent 4 \
        > logs/worker_shard${i}.log 2>&1 &
done
```

______________________________________________________________________

## 환경변수 레퍼런스

### 데이터 경로

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CR_DATA_ROOT` | `/Data1/home/bskang/cds-data` | 데이터셋 루트 디렉토리 |
| `CR_VIDEOS_DIR` | `$CR_DATA_ROOT/front_camera_videos` | 전방 카메라 영상 디렉토리 |
| `CR_OUTPUT_ROOT` | `$CR_DATA_ROOT/caption_v3` | 출력 루트 디렉토리 |
| `CR_META_DIR` | `$CR_DATA_ROOT/clip_meta` | 클립 센서 메타데이터 디렉토리 |
| `CR_EGOMOTION_DIR` | `$CR_DATA_ROOT/egomotion_offline` | Egomotion parquet 디렉토리 |
| `CR_INDEX_DIR` | (AVdata 경로) | clip_id 인덱스 JSON 디렉토리 |

### vLLM / 모델

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CR_VLLM_URL` | `http://localhost:8000/v1` | vLLM 서버 엔드포인트 |
| `CR_VLLM_MODEL` | `nvidia/Cosmos-Reason2-2B` | 모델명 |
| `CR_VLLM_APIKEY` | `EMPTY` | API 키 (로컬 vLLM은 `EMPTY`) |
| `VLLM_MODEL` | `nvidia/Cosmos-Reason2-2B` | `start_vllm.sh` 모델 오버라이드 |
| `VIDEOS_DIR` | `$CR_VIDEOS_DIR` | `start_vllm.sh` 미디어 경로 |
| `MAX_MODEL_LEN` | `16384` | `start_vllm.sh` 최대 컨텍스트 길이 |

### 파이프라인 튜닝

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CR_CONCURRENT` | `2` | 동시 처리 클립 수 |
| `CR_NUM_FRAMES` | `12` | 최종 캡션 스테이지 프레임 수 |
| `CR_NUM_FRAMES_STAGE1` | `8` | Stage 1 프레임 수 |
| `CR_NUM_FRAMES_STAGE2` | `4` | Stage 2 (ODD) 프레임 수 |
| `CR_VIDEO_FPS` | `4` | 비디오 샘플링 FPS |

______________________________________________________________________

## 파이프라인 상세

### caption_refine v1

기존 캡션을 입력으로 받아 4단계로 정제합니다.

```
[입력] 원본 캡션 + 영상
   ↓
Stage 1 (Ground)   — VLM으로 Hallucination / 누락 항목 확인
   ↓
Stage 2 (Extract)  — 구조화된 ODD(Operational Design Domain) JSON 추출
   ↓
Stage 3 (Verify)   — 낮은 confidence 필드 재확인
   ↓
Stage 4 (Refine)   — Hallucination 제거 + ODD 반영한 최종 캡션 생성
   ↓
[출력] refined_caption.txt + odd.json + diff.json
```

### caption_refine v2

센서 데이터(egomotion)를 먼저 확정하고 VLM을 보조로 사용하는 Motion-Aware 파이프라인입니다.

```
[입력] 영상 + egomotion parquet (선택)
   ↓
Stage 0 (Motion)   — 곡률·IMU·속도·타임스탬프 → Phase 1 센서 확정값
   ↓
Stage 1 (Caption)  — NL 시나리오 캡션 (4섹션)
   ↓
Stage 2 (ODD)      — 4차원 ODD JSON (road_structure / environment / dynamic_elements / scene_complexity)
   ↓
Stage 3 (Verify)   — unknown 필드 재확인
   ↓
Stage 4 (CrossVal) — Cross-Validation consistency score 산출
   ↓
Stage 5 (Refine)   — 센서 + ODD + CrossVal 통합 최종 캡션
   ↓
[출력] caption.txt + odd.json + crossval.json
```

**출력 디렉토리 구조 (`CR_OUTPUT_ROOT`):**

```
caption_v3/
├── captions/       # 최종 정제 캡션 (.txt)
├── odd/            # ODD 구조화 결과 (.json)
├── crossval/       # Cross-Validation 결과 (.json)
└── progress.json   # 진행 상태 (재시작 지원)
```

______________________________________________________________________

## 디렉토리 구조

```
.
├── batch_caption.py          # 배치 캡션 생성 — Online 모드
├── batch_caption_offline.py  # 배치 캡션 생성 — Offline 모드
├── generate_shards.py        # 분산 처리용 Shard 파일 생성
├── start_vllm.sh             # vLLM 서버 시작 스크립트
│
├── caption_refine/           # Caption Refinement v1 (4-stage)
│   ├── batch_runner.py       # 배치 실행 진입점
│   ├── pipeline.py           # 파이프라인 오케스트레이션
│   ├── config.py             # 환경변수 설정
│   ├── run_distributed.sh    # 분산 실행 스크립트
│   └── stages/
│       ├── stage1_ground.py
│       ├── stage2_extract.py
│       ├── stage3_verify.py
│       └── stage4_refine.py
│
├── caption_refine_v2/        # Caption Refinement v2 (Motion-Aware, 5-stage)
│   ├── batch_runner.py       # 배치 실행 진입점
│   ├── pipeline.py           # 파이프라인 오케스트레이션
│   ├── config.py             # 환경변수 설정
│   └── stages/
│       ├── stage0_motion.py
│       ├── stage1_caption.py
│       ├── stage2_odd.py
│       ├── stage3_verify.py
│       ├── stage4_crossval.py
│       └── stage5_refine.py
│
├── prompts/                  # VLM 프롬프트 YAML 파일
│   ├── caption_detail.yaml
│   ├── caption_lane.yaml
│   ├── av_cot.yaml
│   └── ...
│
├── shards/                   # generate_shards.py가 생성하는 shard 파일
└── scripts/
    └── inference_sample.py   # 단일 추론 최소 예제
```
