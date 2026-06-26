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

"""
caption_refine_v2 전역 설정.
환경변수로 오버라이드 가능.

v2 변경사항:
  - 기존 caption 미사용 (CAPTIONS_DIR 제거)
  - ODD: 4차원 스키마 (road_structure / environment / dynamic_elements / scene_complexity)
  - confidence/evidence 제거 → unknown 방식 + consistency_score
  - Phase 1: 센서 확정 속성 (곡률·IMU·속도·타임스탬프)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 .env 로드 (caption_refine_v2/.env → 루트 .env 순으로 적용)
# 이미 셸에서 설정된 환경변수는 덮어쓰지 않음 (override=False)
_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env", override=False)
load_dotenv(_HERE.parent / ".env", override=False)

# ── vLLM 서버 ─────────────────────────────────────────────────────────────────
VLLM_BASE_URL  = os.getenv("CR_VLLM_URL",   "http://localhost:8000/v1")
VLLM_MODEL     = os.getenv("CR_VLLM_MODEL",  "nvidia/Cosmos-Reason2-2B")
VLLM_API_KEY   = os.getenv("CR_VLLM_APIKEY", "EMPTY")

VIDEO_INPUT_MODE = os.getenv("CR_VIDEO_MODE", "frames")
VIDEO_FPS        = float(os.getenv("CR_VIDEO_FPS", "4"))

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
DATA_ROOT  = Path(os.getenv("CR_DATA_ROOT",  "/Data1/home/bskang/cds-data"))
VIDEOS_DIR = Path(os.getenv("CR_VIDEOS_DIR", str(DATA_ROOT / "front_camera_videos")))
VIDEO_SUFFIX   = ".camera_front_wide_120fov.mp4"
CAPTION_SUFFIX = ".camera_front_wide_120fov.txt"

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
OUTPUT_ROOT      = Path(os.getenv("CR_OUTPUT_ROOT",      str(DATA_ROOT / "caption_v3")))
ODD_OUT_DIR      = Path(os.getenv("CR_ODD_OUT_DIR",      str(OUTPUT_ROOT / "odd")))
CAPTION_OUT_DIR  = Path(os.getenv("CR_CAPTION_OUT_DIR",  str(OUTPUT_ROOT / "captions")))
CROSSVAL_OUT_DIR = Path(os.getenv("CR_CROSSVAL_OUT_DIR", str(OUTPUT_ROOT / "crossval")))
PROGRESS_FILE    = OUTPUT_ROOT / "progress.json"

# 클립별 센서 메타데이터 디렉토리 ({clip_id}.meta.json 형식)
META_DIR = Path(os.getenv("CR_META_DIR", str(DATA_ROOT / "clip_meta")))

# egomotion.offline 디렉토리 ({clip_id}.egomotion.offline.parquet 형식)
EGOMOTION_DIR = Path(os.getenv("CR_EGOMOTION_DIR", str(DATA_ROOT / "egomotion_offline")))

for _d in (ODD_OUT_DIR, CAPTION_OUT_DIR, CROSSVAL_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 프레임 샘플링 ─────────────────────────────────────────────────────────────
NUM_FRAMES        = int(os.getenv("CR_NUM_FRAMES",        "12"))
NUM_FRAMES_STAGE1 = int(os.getenv("CR_NUM_FRAMES_STAGE1", "8"))   # NL 시나리오: 적은 프레임으로 충분
NUM_FRAMES_STAGE2 = int(os.getenv("CR_NUM_FRAMES_STAGE2", "4"))   # ODD JSON: 토큰 절약
NUM_FRAMES_STAGE3 = int(os.getenv("CR_NUM_FRAMES_STAGE3", "8"))   # unknown 재확인
NUM_FRAMES_STAGE5 = int(os.getenv("CR_NUM_FRAMES_STAGE5", "12"))  # 최종 캡션: 풍부한 프레임
FRAME_QUALITY     = 85
MAX_FRAME_W       = 854
MAX_FRAME_H       = 480

# ── API / 재시도 ──────────────────────────────────────────────────────────────
MAX_RETRIES      = 3
RETRY_DELAY_BASE = 2.0
REQUEST_TIMEOUT  = 180

MAX_TOKENS_STAGE1 = 512    # NL 시나리오 (4섹션 × 1~2문장)
MAX_TOKENS_STAGE2 = 2048   # ODD JSON (16 필드, confidence 없어서 v1보다 작음)
MAX_TOKENS_STAGE3 = 512    # unknown 재확인 (부분 필드)
MAX_TOKENS_STAGE4 = 512    # NL→implied ODD JSON
MAX_TOKENS_STAGE5 = 1024   # 최종 캡션

# ── Phase 1 센서 임계값 ───────────────────────────────────────────────────────
CURVATURE_STRAIGHT_M = float(os.getenv("CR_CURVE_STRAIGHT", "500"))  # m
CURVATURE_GENTLE_M   = float(os.getenv("CR_CURVE_GENTLE",   "200"))  # m
GRADIENT_FLAT_DEG    = float(os.getenv("CR_GRAD_FLAT",      "1.5"))  # degrees
SPEED_LOW_KPH        = float(os.getenv("CR_SPEED_LOW",      "50"))
SPEED_MID_KPH        = float(os.getenv("CR_SPEED_MID",      "100"))
SUNRISE_HOUR         = int(os.getenv("CR_SUNRISE", "6"))    # KST 기준
SUNSET_HOUR          = int(os.getenv("CR_SUNSET",  "19"))   # KST 기준
DUSK_DAWN_WINDOW     = 0.5  # ±30분

# ── ODD QA 임계값 ─────────────────────────────────────────────────────────────
UNKNOWN_RATIO_WARN     = 0.3   # unknown > 30% → 프롬프트 재검토
UNKNOWN_RATIO_CRITICAL = 0.5   # unknown > 50% → 영상 품질 점검
CONSISTENCY_GOOD       = 0.9   # 자동 확정
CONSISTENCY_WARN       = 0.7   # 갈등 항목 검토 필요

# ── 배치 처리 ─────────────────────────────────────────────────────────────────
DEFAULT_CONCURRENT = int(os.getenv("CR_CONCURRENT", "2"))

# ── 인덱스 경로 (batch_runner용) ──────────────────────────────────────────────
INDEX_DIR = Path(os.getenv(
    "CR_INDEX_DIR",
    "/Data1/home/bskang/AVdata-distirbution/data/index",
))
SANFLOW_GAP_PATH = Path(os.getenv(
    "CR_SANFLOW_GAP_PATH",
    "/Data1/home/bskang/AVdata-distirbution/experiments/EXP-002/results/sanflow_gaps.json",
))