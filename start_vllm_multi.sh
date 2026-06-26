#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Byungsu Kang. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 단일 GPU에서 vLLM 서버를 여러 포트로 기동 (검증된 설정 기본값 포함)
#
# 사용법:
#   bash start_vllm_multi.sh                    # 기본: 포트 8000~8001, GPU 0, 2개 인스턴스
#   bash start_vllm_multi.sh --instances 3      # 인스턴스 수 변경
#   bash start_vllm_multi.sh --base-port 8010   # 시작 포트 변경
#   bash start_vllm_multi.sh --gpu 1            # GPU 번호 변경
#   bash start_vllm_multi.sh --kill-all         # 기존 인스턴스 종료 후 재기동
#
# 환경변수 오버라이드:
#   VLLM_MODEL           모델명
#   GPU_MEMORY_UTIL      인스턴스당 GPU 메모리 비율 (vLLM 내부 체크용)
#   GPU_BLOCKS_OVERRIDE  인스턴스당 KV 캐시 블록 수
#   VLLM_ENFORCE_EAGER   1이면 CUDA 그래프 비활성화 (메모리 절약, 속도 10~20% 감소)
#   MAX_MODEL_LEN        최대 컨텍스트 길이
#
# 검증된 설정 (B200 183 GiB, Cosmos-Reason2-8B):
#   인스턴스 2개: GPU_MEMORY_UTIL=0.45, GPU_BLOCKS_OVERRIDE=8000, MAX_MODEL_LEN=32768
#                enforce_eager 없음 → 인스턴스당 ~43.7 GiB, 합계 ~87.5 GiB
#   인스턴스 3개: 현재 설정으로는 불가 (87.5 GiB + 17 GiB 로딩 > 0.45×179 GiB)
#                enforce_eager=1 또는 max_model_len=16384 으로 낮춰야 가능
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 환경변수 로드 (KEY=VALUE / KEY = VALUE 모두 지원) ─────────────────────────
_load_env() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        local key value
        key=$(echo "$line" | cut -d'=' -f1 | tr -d ' ')
        value=$(echo "$line" | cut -d'=' -f2- | sed 's/^[[:space:]]*//')
        [[ -z "$key" ]] && continue
        export "${key}=${value}"
    done < "$file"
}

_load_env "${SCRIPT_DIR}/caption_refine_v2/.env"   # HF_TOKEN
_load_env "${SCRIPT_DIR}/.env"                       # CR_ 설정

# ── NVIDIA / Torch 라이브러리 경로 설정 ──────────────────────────────────────
_VENV_NVIDIA="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia"
_NVIDIA_LIBS=$(find "${_VENV_NVIDIA}" -name "lib" -type d 2>/dev/null | tr '\n' ':')
_TORCH_LIBS="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${_TORCH_LIBS}:${_NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

# /tmp 이 noexec 마운트인 경우 triton/inductor 캐시를 /var/tmp 으로 우회
export TMPDIR="${TMPDIR:-/var/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/var/tmp/triton_${USER:-korea_sdv}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/var/tmp/inductor_${USER:-korea_sdv}}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# HF 토큰 (허깅페이스 모델 다운로드용)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
INSTANCES="${INSTANCES:-2}"
BASE_PORT="${BASE_PORT:-8000}"
GPU="${GPU:-0}"
KILL_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instances)  INSTANCES="$2"; shift 2 ;;
        --base-port)  BASE_PORT="$2"; shift 2 ;;
        --gpu)        GPU="$2";       shift 2 ;;
        --kill-all)   KILL_ALL=true;  shift   ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ── 검증된 기본값 (B200 + Cosmos-Reason2-8B, 2인스턴스 기준) ──────────────────
# enforce_eager=0: CUDA 그래프 활성화 → 최대 성능
# GPU_MEMORY_UTIL=0.45: 3번째 인스턴스 시작 체크 통과를 위한 값 (실제 할당은 GPU_BLOCKS_OVERRIDE 로 제한)
# GPU_BLOCKS_OVERRIDE=8000: 인스턴스당 KV 캐시 블록 수 (8000×16=128K 토큰 KV 공간)
# MAX_MODEL_LEN=16384: vllm 0.12 multimodal 직렬화 버그 회피 (32768 이상 시 serial_utils.py 버그 발생)
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.45}"
GPU_BLOCKS_OVERRIDE="${GPU_BLOCKS_OVERRIDE:-8000}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export GPU_MEMORY_UTIL GPU_BLOCKS_OVERRIDE VLLM_ENFORCE_EAGER MAX_MODEL_LEN

MODEL="${VLLM_MODEL:-${CR_VLLM_MODEL:-nvidia/Cosmos-Reason2-8B}}"
VIDEOS_DIR="${VIDEOS_DIR:-${CR_VIDEOS_DIR:-}}"

echo "=========================================="
echo " vLLM 멀티 인스턴스 기동"
echo "=========================================="
echo "  GPU              : ${GPU}"
echo "  인스턴스 수       : ${INSTANCES}"
echo "  포트              : ${BASE_PORT} ~ $((BASE_PORT + INSTANCES - 1))"
echo "  모델              : ${MODEL}"
echo "  max_model_len     : ${MAX_MODEL_LEN}"
echo "  GPU 메모리 비율   : ${GPU_MEMORY_UTIL}"
echo "  KV 캐시 블록      : ${GPU_BLOCKS_OVERRIDE} 블록/인스턴스"
echo "  Enforce-Eager     : ${VLLM_ENFORCE_EAGER} (0=비활성화, CUDA 그래프 사용)"
echo "  HF_TOKEN          : ${HF_TOKEN:+설정됨 (${#HF_TOKEN}자)}"
echo ""

# ── 기존 프로세스 정리 (--kill-all 시) ────────────────────────────────────────
if [[ "${KILL_ALL}" == true ]]; then
    echo "[INFO] 기존 vLLM 프로세스 종료 중..."
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
        pkill -f "vllm.entrypoints.openai.api_server" || true
        echo "[INFO] 종료 신호 전송 완료. GPU 메모리 해제 대기 (15초)..."
        sleep 15
    else
        echo "[INFO] 실행 중인 vLLM 프로세스 없음"
    fi
fi

# ── 인스턴스별 순차 기동 ───────────────────────────────────────────────────────
# 이전 인스턴스가 완전히 준비된 후 다음을 기동해야 GPU 메모리 체크가 정확함
for i in $(seq 0 $((INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[SKIP] 포트 ${PORT}: 이미 실행 중 (--kill-all 로 재기동하려면 해당 옵션 추가)"
        continue
    fi

    echo ""
    echo "[INFO] ── 인스턴스 $((i+1))/${INSTANCES}: 포트 ${PORT} 기동 ──────────────"
    VLLM_MODEL="${MODEL}" \
    VIDEOS_DIR="${VIDEOS_DIR}" \
        bash "${SCRIPT_DIR}/start_vllm.sh" "${GPU}" "${PORT}"

    echo "[INFO] 포트 ${PORT} 준비 대기 중 (최대 10분)..."
    READY=false
    for attempt in $(seq 1 120); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[OK]  포트 ${PORT} 준비 완료 ($((attempt * 5))초 경과)"
            READY=true
            break
        fi
        if grep -q "ValueError\|RuntimeError\|CUDA out of memory" \
           "${SCRIPT_DIR}/vllm_${PORT}.log" 2>/dev/null; then
            echo "[FAIL] 포트 ${PORT}: 오류 감지"
            grep -E "ValueError|RuntimeError|CUDA out" \
                "${SCRIPT_DIR}/vllm_${PORT}.log" | tail -3
            break
        fi
        sleep 5
    done

    if [[ "${READY}" == false ]]; then
        echo "[WARN] 포트 ${PORT}: 준비 실패 — 다음 인스턴스로 계속"
    fi
done

echo ""
echo "=========================================="
echo " 기동 결과 — 엔드포인트 목록"
echo "=========================================="
RUNNING=0
for i in $(seq 0 $((INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  http://localhost:${PORT}/v1  [OK]"
        RUNNING=$((RUNNING + 1))
    else
        echo "  http://localhost:${PORT}/v1  [X  — vllm_${PORT}.log 확인]"
    fi
done
echo ""
echo "  총 ${RUNNING}/${INSTANCES}개 정상 기동"
echo "  GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""
echo "로그: ${SCRIPT_DIR}/vllm_800*.log"
echo "종료: pkill -f 'vllm.entrypoints.openai.api_server'"
