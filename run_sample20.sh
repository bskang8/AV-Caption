#!/usr/bin/env bash
# caption_refine_v2 시험 실행 — cds-data 샘플 20개, Cosmos-Reason2-8B
#
# 사용법:
#   bash run_sample20.sh            # vLLM 서버 자동 시작 + 파이프라인 실행
#   bash run_sample20.sh --no-vllm  # 이미 실행 중인 vLLM 서버 사용
#
# 전제 조건:
#   - .env 파일이 프로젝트 루트에 있어야 함
#   - caption_refine_v2/.env 에 HF_TOKEN 이 있어야 함
#   - uv 로 의존성 설치 완료
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 환경변수 로드 (KEY=VALUE / KEY = VALUE 모두 지원) ─────────────────────────
_load_env() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 주석·빈 줄 건너뜀
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        # KEY = VALUE → KEY=VALUE 정규화
        local key value
        key=$(echo "$line" | cut -d'=' -f1 | tr -d ' ')
        value=$(echo "$line" | cut -d'=' -f2- | sed 's/^[[:space:]]*//')
        [[ -z "$key" ]] && continue
        export "${key}=${value}"
    done < "$file"
}

_load_env "${SCRIPT_DIR}/caption_refine_v2/.env"   # HF_TOKEN
_load_env "${SCRIPT_DIR}/.env"                       # CR_ 설정

# ── 변수 확인 ──────────────────────────────────────────────────────────────────
MODEL="${CR_VLLM_MODEL:-nvidia/Cosmos-Reason2-8B}"
VLLM_URL="${CR_VLLM_URL:-http://localhost:8000/v1}"
VIDEOS_DIR="${CR_VIDEOS_DIR:-}"
SAMPLE_FILE="${SCRIPT_DIR}/sample_20.json"
GPU="${VLLM_GPU:-0}"
PORT="${VLLM_PORT:-8000}"
START_VLLM=true

for arg in "$@"; do
    [[ "${arg}" == "--no-vllm" ]] && START_VLLM=false
done

echo "=========================================="
echo " caption_refine_v2 — 시험 실행 (20 클립)"
echo "=========================================="
echo "  모델      : ${MODEL}"
echo "  vLLM URL  : ${VLLM_URL}"
echo "  영상 경로 : ${VIDEOS_DIR}"
echo "  샘플 파일 : ${SAMPLE_FILE}"
echo "  HF_TOKEN  : ${HF_TOKEN:+설정됨 (${#HF_TOKEN}자)}"
echo ""

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN 이 설정되지 않았습니다."
    echo "        caption_refine_v2/.env 파일에 HF_TOKEN=hf_... 를 추가하세요."
    exit 1
fi

if [[ ! -f "${SAMPLE_FILE}" ]]; then
    echo "[ERROR] sample_20.json 을 찾을 수 없습니다: ${SAMPLE_FILE}"
    exit 1
fi

# ── vLLM 서버 시작 ─────────────────────────────────────────────────────────────
if [[ "${START_VLLM}" == true ]]; then
    echo "[INFO] vLLM 서버 시작 중 (GPU=${GPU}, PORT=${PORT}, MODEL=${MODEL})..."

    # 이미 실행 중인지 확인
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] vLLM 서버가 이미 실행 중입니다 (localhost:${PORT}). --no-vllm 모드로 진행합니다."
        START_VLLM=false
    else
        # vllm 프로세스에 전달할 환경변수 명시 export
        export VLLM_MODEL="${MODEL}"
        export CR_VIDEOS_DIR="${VIDEOS_DIR}"
        export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
        export HF_TOKEN="${HF_TOKEN}"
        # torch/nvidia 라이브러리 경로 설정 (nohup 프로세스 상속용)
        _VENV_NVIDIA="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia"
        _NVIDIA_LIBS=$(find "${_VENV_NVIDIA}" -name "lib" -type d 2>/dev/null | tr '\n' ':')
        _TORCH_LIBS="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/torch/lib"
        export LD_LIBRARY_PATH="${_TORCH_LIBS}:${_NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
        # /tmp 이 noexec 로 마운트돼 있어 triton 캐시를 실행할 수 없으므로 /var/tmp 사용
        export TMPDIR="/var/tmp"
        export TRITON_CACHE_DIR="/var/tmp/triton_${USER:-korea_sdv}"
        export TORCHINDUCTOR_CACHE_DIR="/var/tmp/inductor_${USER:-korea_sdv}"
        mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
        bash "${SCRIPT_DIR}/start_vllm.sh" "${GPU}" "${PORT}"

        echo "[INFO] vLLM 서버 준비 대기 중 (최대 5분)..."
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
                echo "[INFO] vLLM 서버 준비 완료 (${i}×5초 경과)"
                break
            fi
            [[ ${i} -eq 60 ]] && { echo "[ERROR] vLLM 서버 시작 시간 초과"; exit 1; }
            sleep 5
        done
    fi
fi

# ── 파이프라인 실행 ────────────────────────────────────────────────────────────
echo ""
echo "[INFO] caption_refine_v2 파이프라인 실행 시작..."
echo "[INFO] 샘플 20개, vLLM: ${VLLM_URL}"
echo ""

uv run python -m caption_refine_v2.batch_runner \
    --ids-file "${SAMPLE_FILE}" \
    --concurrent "${CR_CONCURRENT:-2}" \
    --vllm-url "${VLLM_URL}"

echo ""
echo "[INFO] 완료! 결과 위치: ${CR_OUTPUT_ROOT:-${CR_DATA_ROOT}/caption_v3}"
