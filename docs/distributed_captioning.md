# 분산 캡션 추출 가이드

30만 개 이상의 영상에서 캡션을 추출하는 분산 처리 방법을 설명합니다.  
현재 서버의 GPU 2장(Online 방식)과 외부 서버의 GPU 8장(Offline 방식)을 합쳐 총 10개 GPU에 작업을 분배합니다.

---

## 전체 구조

```
현재 서버 (GPU 2장)                    외부 서버 (GPU 8장)
─────────────────────────────          ─────────────────────────────
vLLM HTTP 서버 × 2 (port 8000/8001)   vLLM.LLM 직접 로드 × 8
         ↑                                       ↑
batch_caption.py (online)              batch_caption_offline.py (offline)
  --video-list shard_00.txt              --video-list shard_02.txt
  --video-list shard_01.txt              --video-list shard_03.txt
                                         ...
                                         --video-list shard_09.txt
         ↓                                       ↓
/Data1/home/bskang/cds-data/captions/   /local/captions/
                                                 ↓
                                          rsync → 현재 서버로 병합
```

### Online vs Offline 비교

| 항목 | Online (현재 서버) | Offline (외부 서버) |
|------|-------------------|---------------------|
| 모델 로드 방식 | vLLM HTTP 서버 (상시 가동) | 프로세스 내 `vllm.LLM` 직접 로드 |
| 호출 방식 | OpenAI API → HTTP 요청 | Python API → 직접 추론 |
| GPU 할당 | `CUDA_VISIBLE_DEVICES` + 포트 분리 | `CUDA_VISIBLE_DEVICES` 로 프로세스 고정 |
| 장점 | 서버 재사용, 모니터링 용이 | 서버 설정 없음, 독립 실행 |
| 주의 | 배치 스크립트 실행 전 서버 기동 필요 | GPU 메모리 독점 (프로세스당 1 GPU) |

---

## 관련 스크립트

| 파일 | 역할 |
|------|------|
| `generate_shards.py` | 미처리 영상 목록을 N개 shard 파일로 분할 (실행 전 1회) |
| `batch_caption.py` | Online 방식 배치 처리 (현재 서버) |
| `batch_caption_offline.py` | Offline 방식 배치 처리 (외부 서버) |
| `shards/shard_NN.txt` | 각 worker가 처리할 영상 경로 목록 |

---

## 영상 분배 방식 (Modulo Sharding)

```
pending(미처리) 영상 정렬 후:

  pending[0]  → shard_00.txt  (현재 서버 GPU 0)
  pending[1]  → shard_01.txt  (현재 서버 GPU 1)
  pending[2]  → shard_02.txt  (외부 서버 GPU 0)
  ...
  pending[9]  → shard_09.txt  (외부 서버 GPU 7)
  pending[10] → shard_00.txt  (다시 현재 서버 GPU 0)
  pending[11] → shard_01.txt  ...
```

**핵심 원칙**
- `generate_shards.py` 실행 시점에 이미 완료된 영상(`/Data1/home/bskang/cds-data/captions/*.txt` 존재 여부)을 제외하고 shard를 생성합니다.
- 따라서 외부 서버는 현재 서버의 출력 디렉터리를 마운트하지 않아도 **중복 처리가 발생하지 않습니다**.
- `--skip-existing`은 worker 중간 실패 후 **재시작 안전망**으로만 사용됩니다.

---

## 실행 순서

### Step 1 — Shard 파일 생성 (현재 서버에서 1회 실행)

```bash
cd /Data1/home/bskang/cosmos-reason2

# dry-run으로 먼저 확인
python generate_shards.py --num-shards 10 --dry-run

# 실제 생성
python generate_shards.py \
    --input-dir /Data1/home/bskang/cds-data/front_camera_videos \
    --captions-dir /Data1/home/bskang/cds-data/captions \
    --num-shards 10 \
    --output-dir shards/
```

출력 예시:
```
Scanning videos : /Data1/home/bskang/cds-data/front_camera_videos
  Total videos   : 306152
  Already done   : 12400
  Pending        : 293752
  Written: shards/shard_00.txt  (29376 videos)
  Written: shards/shard_01.txt  (29376 videos)
  ...
  Written: shards/shard_09.txt  (29375 videos)
```

### Step 2 — 외부 서버에 Shard 파일 전달

```bash
# 현재 서버에서 실행
rsync -av shards/shard_0{2..9}.txt  외부서버:/path/to/cosmos-reason2/shards/

# 또는 scp
scp shards/shard_{02..09}.txt  user@외부서버:/path/to/cosmos-reason2/shards/
```

### Step 3 — 현재 서버 실행 (Online, GPU 2장)

**3-1. vLLM 서버 기동**

vllm은 호스트에 없고 `.venv`에 설치되어 있으므로 먼저 활성화합니다.

```bash
source /Data1/home/bskang/cosmos-reason2/.venv/bin/activate
```

백그라운드 실행 방법은 세 가지입니다. **tmux를 권장**합니다.

**방법 A — tmux (권장)**

```bash
# GPU 0 — port 8000
tmux new-session -d -s vllm_0 \
    "source /Data1/home/bskang/cosmos-reason2/.venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=0 vllm serve nvidia/Cosmos-Reason2-2B \
         --port 8000 --max-model-len 16384 \
         --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
         --media-io-kwargs '{\"video\": {\"num_frames\": -1}}' \
         --reasoning-parser qwen3"

# GPU 1 — port 8001
tmux new-session -d -s vllm_1 \
    "source /Data1/home/bskang/cosmos-reason2/.venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=1 vllm serve nvidia/Cosmos-Reason2-2B \
         --port 8001 --max-model-len 16384 \
         --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
         --media-io-kwargs '{\"video\": {\"num_frames\": -1}}' \
         --reasoning-parser qwen3"

# 로그 확인 (붙기/떼기: Ctrl+B, D)
tmux attach -t vllm_0

# 종료
tmux kill-session -t vllm_0
tmux kill-session -t vllm_1
```

**방법 B — nohup**

```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Cosmos-Reason2-2B \
    --port 8000 --max-model-len 16384 \
    --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --reasoning-parser qwen3 \
    > vllm_8000.log 2>&1 &
echo "GPU0 PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup vllm serve nvidia/Cosmos-Reason2-2B \
    --port 8001 --max-model-len 16384 \
    --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --reasoning-parser qwen3 \
    > vllm_8001.log 2>&1 &
echo "GPU1 PID: $!"

# 로그 확인
tail -f vllm_8000.log
```

**방법 C — Docker 백그라운드**

```bash
docker run -d --rm \
    --name vllm_server_0 \
    --gpus '"device=0"' \
    -v /Data1/home/bskang/cosmos-reason2:/workspace \
    -v /root/.cache:/root/.cache \
    -v /Data1/home/bskang/cds-data:/Data1/home/bskang/cds-data \
    -p 8000:8000 \
    5a0bd46102db \
    /workspace/.venv/bin/vllm serve nvidia/Cosmos-Reason2-2B \
        --port 8000 --max-model-len 16384 \
        --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --reasoning-parser qwen3

docker run -d --rm \
    --name vllm_server_1 \
    --gpus '"device=1"' \
    -v /Data1/home/bskang/cosmos-reason2:/workspace \
    -v /root/.cache:/root/.cache \
    -v /Data1/home/bskang/cds-data:/Data1/home/bskang/cds-data \
    -p 8001:8001 \
    5a0bd46102db \
    /workspace/.venv/bin/vllm serve nvidia/Cosmos-Reason2-2B \
        --port 8001 --max-model-len 16384 \
        --allowed-local-media-path /Data1/home/bskang/cds-data/front_camera_videos \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --reasoning-parser qwen3

# 로그 확인
docker logs -f vllm_server_0

# 종료
docker stop vllm_server_0 vllm_server_1
```

| | tmux | nohup | Docker |
|---|---|---|---|
| 간편함 | ✅ | ✅ | 보통 |
| 로그 실시간 확인 | 터미널 직접 | 파일로만 | `docker logs` |
| SSH 끊겨도 유지 | ✅ | ✅ | ✅ |
| 종료 방법 | `tmux kill-session` | `kill PID` | `docker stop` |

**3-2. 서버 Ready 확인**

```bash
curl -s http://localhost:8000/health && echo "GPU0 OK"
curl -s http://localhost:8001/health && echo "GPU1 OK"
```

**3-3. 배치 스크립트 실행**

nohup으로 백그라운드 실행합니다 (SSH 끊겨도 계속 동작).

```bash
# shard_00 처리 (GPU 0)
nohup python batch_caption.py \
    --video-list shards/shard_00.txt \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8000 --fps 4 --skip-existing \
    > caption_shard00.log 2>&1 &
echo "PID: $!"

# shard_01 처리 (GPU 1)
nohup python batch_caption.py \
    --video-list shards/shard_01.txt \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8001 --fps 4 --skip-existing \
    > caption_shard01.log 2>&1 &
echo "PID: $!"




python batch_caption.py \
    --video-list shards/shard_00.txt \
    --output-dir /Data1/home/bskang/cds-data/lane_captions \
    --prompt-file prompts/caption_lane.yaml \
    --port 8000 --fps 4 --skip-existing




# 로그 확인
tail -f caption_shard00.log
tail -f caption_shard01.log
```

### Step 4 — 외부 서버 실행 (Offline, GPU 8장)

```bash
# 외부 서버에서 실행
# GPU 인덱스 0~7 → shard_id 2~9

for GPU_ID in 0 1 2 3 4 5 6 7; do
    SHARD_ID=$((GPU_ID + 2))
    SHARD_FILE=$(printf "shards/shard_%02d.txt" $SHARD_ID)

    CUDA_VISIBLE_DEVICES=$GPU_ID python batch_caption_offline.py \
        --video-list $SHARD_FILE \
        --output-dir /local/captions \
        --model nvidia/Cosmos-Reason2-7B \
        --fps 4 \
        --max-model-len 16384 \
        --skip-existing &
done

wait
echo "All offline workers done."
```

### Step 5 — 외부 서버 결과 병합

```bash
# 외부 서버 작업 완료 후, 현재 서버로 수집
rsync -av user@외부서버:/local/captions/*.txt \
    /Data1/home/bskang/cds-data/captions/
```

---

## 재시작 / 장애 복구

### 중간에 worker가 죽은 경우

`--skip-existing` 옵션 덕에 **같은 명령을 그대로 재실행**하면 됩니다.  
이미 저장된 `.txt` 파일은 건너뛰고 미완료 영상부터 이어서 처리합니다.

```bash
# 그대로 재실행
python batch_caption.py \
    --video-list shards/shard_00.txt \
    --output-dir /Data1/home/bskang/cds-data/captions \
    --port 8000 --fps 4 --skip-existing
```

### 2차 전체 실행 (새 영상 추가 등)

`generate_shards.py`를 다시 실행하면 그 시점의 완료분이 자동으로 제외된 새 shard 파일이 생성됩니다.

```bash
python generate_shards.py --num-shards 10 --output-dir shards/
```

---

## 진행 상황 확인

```bash
# 완료된 캡션 파일 수
ls /Data1/home/bskang/cds-data/captions/*.txt | wc -l

# 전체 영상 수 대비 진행률
TOTAL=$(ls /Data1/home/bskang/cds-data/front_camera_videos/*.mp4 | wc -l)
DONE=$(ls /Data1/home/bskang/cds-data/captions/*.txt 2>/dev/null | wc -l)
echo "Done: $DONE / $TOTAL ($(( DONE * 100 / TOTAL ))%)"

# 특정 shard 파일의 미완료 영상 수 확인
python - <<'EOF'
from pathlib import Path
shard = Path("shards/shard_00.txt")
captions_dir = Path("/Data1/home/bskang/cds-data/captions")
videos = [Path(l.strip()) for l in shard.read_text().splitlines() if l.strip()]
done = sum(1 for v in videos if (captions_dir / (v.stem + ".txt")).exists())
print(f"shard_00: {done}/{len(videos)} done")
EOF
```

---

## 경로 정리

| 항목 | 경로 |
|------|------|
| 입력 영상 | `/Data1/home/bskang/cds-data/front_camera_videos/` |
| 출력 캡션 (현재 서버) | `/Data1/home/bskang/cds-data/captions/` |
| 출력 캡션 (외부 서버) | `/local/captions/` (외부 서버 로컬) |
| Shard 파일 | `/Data1/home/bskang/cosmos-reason2/shards/` |
| 프롬프트 파일 | `/Data1/home/bskang/cosmos-reason2/prompts/caption_detail.yaml` |
