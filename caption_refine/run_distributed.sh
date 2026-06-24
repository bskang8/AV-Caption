#!/usr/bin/env bash
# =============================================================================
# caption_refine 분산 실행 스크립트 (GPU 10장 기준)
#
# 사전 조건:
#   - 각 GPU에 vLLM 서버가 이미 실행 중이거나, --start-vllm 옵션 사용
#   - 데이터 경로가 이 서버에서 접근 가능 (NFS 마운트 또는 로컬 복사)
#
# 사용법:
#   # 1) vLLM 서버가 이미 떠 있을 때 (worker만 실행)
#   bash run_distributed.sh
#
#   # 2) vLLM 서버도 함께 기동
#   bash run_distributed.sh --start-vllm
#
#   # 3) 전체 설정 오버라이드
#   TOTAL_GPUS=4 SOURCE=longtail CONCURRENT=6 bash run_distributed.sh
# =============================================================================
set -euo pipefail

# ── 설정 ───────────────────────────────────────────────────────────────────────
TOTAL_GPUS="${TOTAL_GPUS:-10}"           # GPU 수 = 샤드 수
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}" # GPU 0 → 포트 8000, GPU 1 → 8001, ...
VLLM_MODEL="${VLLM_MODEL:-nvidia/Cosmos-Reason2-7B}"
VLLM_START_WAIT="${VLLM_START_WAIT:-60}" # vLLM 기동 대기 시간 (초)

SOURCE="${SOURCE:-all}"                  # gap | longtail | all
CONCURRENT="${CONCURRENT:-4}"            # 샤드당 동시 처리 클립 수
LIMIT="${LIMIT:-}"                       # 전체 클립 수 제한 (빈 문자열 = 무제한)

# 데이터/출력 경로 (환경변수로 오버라이드)
export CR_DATA_ROOT="${CR_DATA_ROOT:-/Data1/home/bskang/cds-data}"
export CR_INDEX_DIR="${CR_INDEX_DIR:-/Data1/home/bskang/AVdata-distirbution/data/index}"
export CR_SANFLOW_GAP_PATH="${CR_SANFLOW_GAP_PATH:-/Data1/home/bskang/AVdata-distirbution/experiments/EXP-002/results/sanflow_gaps.json}"

LOG_DIR="./logs/distributed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

START_VLLM=false
for arg in "$@"; do
    [[ "$arg" == "--start-vllm" ]] && START_VLLM=true
done

# ── vLLM 서버 기동 (--start-vllm 옵션) ────────────────────────────────────────
if $START_VLLM; then
    echo "[INFO] Starting $TOTAL_GPUS vLLM servers..."
    VIDEOS_DIR="${CR_VIDEOS_DIR:-${CR_DATA_ROOT}/front_camera_videos}"
    for i in $(seq 0 $((TOTAL_GPUS - 1))); do
        PORT=$((VLLM_BASE_PORT + i))
        LOG_FILE="$LOG_DIR/vllm_gpu${i}.log"
        echo "[INFO]   GPU $i → port $PORT (log: $LOG_FILE)"
        CUDA_VISIBLE_DEVICES=$i vllm serve "$VLLM_MODEL" \
            --port "$PORT" \
            --dtype auto \
            --max-model-len 16384 \
            --allowed-local-media-path "$VIDEOS_DIR" \
            --media-io-kwargs '{"video": {"num_frames": -1}}' \
            --reasoning-parser qwen3 \
            > "$LOG_FILE" 2>&1 &
    done
    echo "[INFO] Waiting ${VLLM_START_WAIT}s for vLLM servers to initialize..."
    sleep "$VLLM_START_WAIT"
fi

# ── vLLM 헬스체크 ─────────────────────────────────────────────────────────────
echo "[INFO] Checking vLLM server health..."
HEALTHY=0
for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    PORT=$((VLLM_BASE_PORT + i))
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[OK]   GPU $i (port $PORT) is ready"
        HEALTHY=$((HEALTHY + 1))
    else
        echo "[WARN] GPU $i (port $PORT) is NOT responding"
    fi
done

if [[ $HEALTHY -eq 0 ]]; then
    echo "[ERROR] No vLLM servers are reachable. Start vLLM first or use --start-vllm."
    exit 1
fi
echo "[INFO] $HEALTHY/$TOTAL_GPUS vLLM servers healthy"

# ── batch_runner 워커 실행 ────────────────────────────────────────────────────
echo "[INFO] Launching $TOTAL_GPUS batch_runner workers..."
PIDS=()

# --limit 인자 동적 구성
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"

for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    PORT=$((VLLM_BASE_PORT + i))
    # 해당 포트가 응답하지 않으면 스킵
    if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[SKIP] Shard $i — GPU $i (port $PORT) not ready, skipping"
        continue
    fi

    LOG_FILE="$LOG_DIR/worker_shard${i}.log"
    echo "[INFO]   Shard $i/$((TOTAL_GPUS-1)) → port $PORT (log: $LOG_FILE)"

    CR_VLLM_URL="http://localhost:${PORT}/v1" \
    uv run python -m caption_refine.batch_runner \
        --source "$SOURCE" \
        --shard-index "$i" \
        --total-shards "$TOTAL_GPUS" \
        --concurrent "$CONCURRENT" \
        $LIMIT_ARG \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[INFO] All workers launched (PIDs: ${PIDS[*]})"
echo "[INFO] Logs: $LOG_DIR/"
echo "[INFO] Waiting for completion..."
echo ""

# ── 진행 상황 모니터링 ────────────────────────────────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[DONE] PID $pid completed successfully"
    else
        echo "[FAIL] PID $pid exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "══════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    echo "[SUCCESS] All ${#PIDS[@]} workers completed."
else
    echo "[WARN] $FAILED worker(s) failed. Check logs in $LOG_DIR/"
fi
echo "Progress files: ${CR_DATA_ROOT}/caption_v2/progress_shard*.json"
echo "══════════════════════════════════════════════════"
