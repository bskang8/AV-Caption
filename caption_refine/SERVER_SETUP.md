# 다른 서버 환경 설정 가이드

caption_refine을 새 서버에서 실행하기 위한 환경 설정 방법입니다.

> caption_refine 워커는 vLLM에 HTTP로 요청만 하므로 **워커 자체는 GPU 불필요**합니다.  
> GPU는 vLLM 서버에만 필요합니다.

---

## 필요한 것 요약

```
[다른 서버에서 필요한 것]

vLLM 서버 (GPU 필요)
├── CUDA 12.x + GPU 드라이버
├── Python 3.10+
├── vLLM 패키지
└── nvidia/Cosmos-Reason2-7B 모델 다운로드

caption_refine 워커 (GPU 불필요)
├── Python 3.10+
├── uv (패키지 매니저)
├── openai, opencv-python-headless, numpy
└── cosmos-reason2 코드 (이 repo)

데이터 접근
├── 비디오 파일 (front_camera_videos/)
├── 원본 캡션 파일 (captions/)
└── clip_ids 인덱스 파일
```

---

## Step 1 — 시스템 패키지 설치

```bash
sudo apt-get update && sudo apt-get install -y \
    python3 python3-pip \
    curl git \
    ffmpeg \
    libgl1          # opencv 의존성
```

---

## Step 2 — uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version  # 확인
```

---

## Step 3 — 코드 복사

```bash
# 현재 서버에서 다른 서버로 전송
rsync -av /Data1/home/bskang/cosmos-reason2/ user@다른서버:/workspace/cosmos-reason2/

# 또는 다른 서버에서 git clone (repo가 있는 경우)
git clone <repo-url> /workspace/cosmos-reason2
```

---

## Step 4 — Python 의존성 설치

```bash
cd /workspace/cosmos-reason2

# caption_refine 패키지만 설치 (GPU/CUDA 불필요)
uv pip install openai "opencv-python-headless>=4.0" "numpy>=1.21"

# 또는 uv workspace 전체 설치 (cosmos-reason2 전체가 필요한 경우)
uv sync
```

---

## Step 5 — 데이터 접근 설정

### 방법 A: NFS 마운트 (권장)

```bash
# 현재 서버의 데이터를 NFS로 마운트
sudo mount -t nfs 현재서버IP:/Data1/home/bskang/cds-data /mnt/cds-data

# 환경변수로 경로 지정
export CR_DATA_ROOT=/mnt/cds-data
```

### 방법 B: rsync로 복사

```bash
rsync -av --progress \
    user@현재서버:/Data1/home/bskang/cds-data/front_camera_videos/ \
    /local/cds-data/front_camera_videos/

rsync -av --progress \
    user@현재서버:/Data1/home/bskang/cds-data/captions/ \
    /local/cds-data/captions/

# 인덱스 파일
rsync -av \
    user@현재서버:/Data1/home/bskang/AVdata-distirbution/ \
    /local/avdata/

export CR_DATA_ROOT=/local/cds-data
export CR_INDEX_DIR=/local/avdata/data/index
export CR_SANFLOW_GAP_PATH=/local/avdata/experiments/EXP-002/results/sanflow_gaps.json
```

---

## Step 6 — vLLM 설치 및 모델 서빙

```bash
# vLLM 설치
pip install vllm

# 모델 다운로드 (HuggingFace)
# 2B: GPU 메모리 ~8GB, 7B: ~20GB
huggingface-cli download nvidia/Cosmos-Reason2-2B
# huggingface-cli download nvidia/Cosmos-Reason2-7B  # 더 높은 품질

# 비디오 경로 설정 (NFS 마운트 경로에 맞게 조정)
VIDEOS_DIR="/mnt/cds-data/front_camera_videos"  # 실제 마운트 경로로 변경

# GPU 10장에 vLLM 10개 기동
mkdir -p logs
for i in $(seq 0 9); do
    CUDA_VISIBLE_DEVICES=$i vllm serve nvidia/Cosmos-Reason2-2B \
        --port $((8000+i)) \
        --dtype auto \
        --max-model-len 16384 \
        --allowed-local-media-path "$VIDEOS_DIR" \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --reasoning-parser qwen3 \
        > logs/vllm_gpu${i}.log 2>&1 &
    echo "GPU $i → port $((8000+i)) PID=$!"
done

# 기동 확인 (모델 로딩 1~3분 소요)
sleep 90
for i in $(seq 0 9); do
    curl -sf http://localhost:$((8000+i))/health && echo "GPU $i OK" || echo "GPU $i NG"
done
```

> **`--allowed-local-media-path`**: 서버가 직접 파일을 읽을 수 있는 경로.  
> caption_refine은 `video` 모드에서 `file://` URL을 전송하므로 이 경로가 반드시 유효해야 합니다.  
> `--limit-mm-per-prompt`는 `video` 모드(video_url 1개)에서는 불필요합니다.

---

## Step 7 — 워커 실행

```bash
cd /workspace/cosmos-reason2

export CR_DATA_ROOT=/mnt/cds-data
export CR_INDEX_DIR=/local/avdata/data/index
export CR_SANFLOW_GAP_PATH=/local/avdata/experiments/EXP-002/results/sanflow_gaps.json

# run_distributed.sh로 한번에 실행
SOURCE=all CONCURRENT=4 \
bash caption_refine/run_distributed.sh

# 또는 개별 실행
for i in $(seq 0 9); do
    CR_VLLM_URL="http://localhost:$((8000+i))/v1" \
    uv run python -m caption_refine.batch_runner \
        --source all \
        --shard-index $i \
        --total-shards 10 \
        --concurrent 4 \
        > logs/worker_shard${i}.log 2>&1 &
done
```

---

## Docker로 실행하는 경우 (더 간편)

이 repo의 Dockerfile을 사용하면 환경 설정이 대폭 단순해집니다.

```bash
# 현재 서버에서 이미지 빌드 후 다른 서버로 전송
docker build -t cosmos-reason2 .
docker save cosmos-reason2 | ssh user@다른서버 docker load

# 다른 서버에서 컨테이너 실행
docker run -d \
    --gpus all \
    -v /mnt/cds-data:/Data1/home/bskang/cds-data \
    -v /workspace/cosmos-reason2:/workspace \
    -p 8000-8009:8000-8009 \
    cosmos-reason2
```

---

## 체크리스트

```
[ ] Python 3.10+ 설치
[ ] uv 설치
[ ] cosmos-reason2 코드 복사
[ ] openai / opencv-python-headless / numpy 설치
[ ] 데이터 경로 마운트 또는 복사 (front_camera_videos/, captions/, index/)
[ ] CR_DATA_ROOT / CR_INDEX_DIR / CR_SANFLOW_GAP_PATH 환경변수 설정
[ ] vLLM 설치 + nvidia/Cosmos-Reason2-7B 모델 다운로드
[ ] GPU당 vLLM 서버 기동 (포트 8000~8009)
[ ] vLLM 헬스체크 확인 (curl localhost:8000/health)
[ ] run_distributed.sh 실행
```

---

## 관련 문서

- [DISTRIBUTED.md](DISTRIBUTED.md) — 분산 실행 상세 가이드 (샤드 분할, 진행 확인, 재시작)
- [PIPELINE.md](PIPELINE.md) — 4-Stage 파이프라인 상세 설명
