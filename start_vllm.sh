#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Byungsu Kang. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# vLLM 서버 시작 스크립트 — 어떤 서버에서도 동작
#
# 사용법:
#   bash start_vllm.sh [GPU번호] [포트]
#   bash start_vllm.sh 0 8000        # GPU 0, 포트 8000 (기본값)
#   bash start_vllm.sh 1 8001        # GPU 1, 포트 8001
#
# 환경변수로 오버라이드 가능:
#   VLLM_MODEL      모델명 (기본: nvidia/Cosmos-Reason2-2B)
#   VIDEOS_DIR      --allowed-local-media-path 경로
#   MAX_MODEL_LEN   최대 컨텍스트 길이 (기본: 16384)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
PORT="${2:-8000}"

VLLM_MODEL="${VLLM_MODEL:-nvidia/Cosmos-Reason2-2B}"
VIDEOS_DIR="${VIDEOS_DIR:-${CR_VIDEOS_DIR:-${CR_DATA_ROOT:-/Data1/home/bskang/cds-data}/front_camera_videos}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

echo "[INFO] GPU=$GPU  PORT=$PORT  MODEL=$VLLM_MODEL"
echo "[INFO] VIDEOS_DIR=$VIDEOS_DIR"

# ── 1) NVIDIA 라이브러리 경로 — 프로젝트 venv에서 자동 탐색 ─────────────────
# CUDA 12.8: uv sync --extra cu128 + uv pip install nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12
# CUDA 13.0: uv sync --extra cu130 (cu13 라이브러리가 의존성으로 자동 설치됨)

VENV_NVIDIA="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS=$(find "${VENV_NVIDIA}" -name "lib" -type d 2>/dev/null | tr '\n' ':')

# 누락된 핵심 라이브러리 감지
MISSING=()
for lib in libcudnn.so.9 libcusparseLt.so.0 libnccl.so.2; do
    if ! find "${VENV_NVIDIA}" -name "${lib}" 2>/dev/null | grep -q .; then
        MISSING+=("$lib")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "[ERROR] 누락된 NVIDIA 라이브러리: ${MISSING[*]}"
    echo "        CUDA 12.8 환경이라면 다음 명령어로 설치하세요:"
    echo "        uv pip install nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12"
    echo "        CUDA 13.0 환경이라면 다음 명령어로 설치하세요:"
    echo "        uv pip install nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13"
    exit 1
fi

TORCH_LIBS="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

# ── 2) ptxas 경로 — CUDA 툴킷에서 자동 탐색 ────────────────────────────────
if [ -z "${TRITON_PTXAS_PATH:-}" ]; then
    PTXAS=$(find /usr/local/cuda*/bin /usr/bin -name "ptxas" 2>/dev/null | sort -V | tail -1)
    if [ -n "${PTXAS}" ]; then
        export TRITON_PTXAS_PATH="${PTXAS}"
        echo "[INFO] ptxas: ${PTXAS}"
    else
        echo "[WARN] ptxas를 찾을 수 없습니다. Triton 커널 컴파일이 실패할 수 있습니다."
    fi
fi

# ── 3) vLLM 서버 기동 ────────────────────────────────────────────────────────
LOG_FILE="${SCRIPT_DIR}/vllm_${PORT}.log"
echo "[INFO] 로그: ${LOG_FILE}"
echo "[DEBUG] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:0:120}" | tee -a "${LOG_FILE}"

nohup env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-}" \
    TMPDIR="${TMPDIR:-/var/tmp}" \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/var/tmp/triton_${USER:-korea_sdv}}" \
    TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/var/tmp/inductor_${USER:-korea_sdv}}" \
    "${SCRIPT_DIR}/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --allowed-local-media-path "${VIDEOS_DIR}" \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --reasoning-parser qwen3 \
    > "${LOG_FILE}" 2>&1 &

echo "[INFO] vLLM 시작 완료 — PID=$!"
echo "[INFO] 준비 확인: watch -n5 'curl -sf http://localhost:${PORT}/health && echo OK'"