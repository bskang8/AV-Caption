"""
caption_refine 패키지 전역 설정.
환경변수로 오버라이드 가능.
"""
import os
from pathlib import Path

# ── vLLM 서버 ─────────────────────────────────────────────────────────────────
VLLM_BASE_URL  = os.getenv("CR_VLLM_URL",   "http://localhost:8000/v1")
VLLM_MODEL     = os.getenv("CR_VLLM_MODEL",  "nvidia/Cosmos-Reason2-2B")
VLLM_API_KEY   = os.getenv("CR_VLLM_APIKEY", "EMPTY")

# VIDEO_INPUT_MODE:
#   "frames" → 균등 샘플링한 JPEG 프레임을 image_url 배열로 전달 (범용)
#   "video"  → MP4를 base64로 직렬화해 video_url 한 개로 전달 (vLLM >= 0.5 필요)
VIDEO_INPUT_MODE = os.getenv("CR_VIDEO_MODE", "frames")
VIDEO_FPS        = float(os.getenv("CR_VIDEO_FPS", "4"))

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
# CR_DATA_ROOT만 설정하면 하위 경로들이 자동으로 결정됨
DATA_ROOT      = Path(os.getenv("CR_DATA_ROOT",     "/Data1/home/bskang/cds-data"))
VIDEOS_DIR     = Path(os.getenv("CR_VIDEOS_DIR",    str(DATA_ROOT / "front_camera_videos")))
CAPTIONS_DIR   = Path(os.getenv("CR_CAPTIONS_DIR",  str(DATA_ROOT / "captions")))
VIDEO_SUFFIX   = ".camera_front_wide_120fov.mp4"
CAPTION_SUFFIX = ".camera_front_wide_120fov.txt"

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
OUTPUT_ROOT     = Path(os.getenv("CR_OUTPUT_ROOT",     str(DATA_ROOT / "caption_v2")))
ODD_OUT_DIR     = Path(os.getenv("CR_ODD_OUT_DIR",     str(OUTPUT_ROOT / "odd")))
CAPTION_OUT_DIR = Path(os.getenv("CR_CAPTION_OUT_DIR", str(OUTPUT_ROOT / "captions")))
DIFF_OUT_DIR    = Path(os.getenv("CR_DIFF_OUT_DIR",    str(OUTPUT_ROOT / "diff")))
PROGRESS_FILE   = OUTPUT_ROOT / "progress.json"

for _d in (ODD_OUT_DIR, CAPTION_OUT_DIR, DIFF_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 프레임 샘플링 ─────────────────────────────────────────────────────────────
NUM_FRAMES       = int(os.getenv("CR_NUM_FRAMES",       "12"))
NUM_FRAMES_STAGE1 = int(os.getenv("CR_NUM_FRAMES_STAGE1", "16"))  # Stage 1은 더 많은 프레임으로 할루시네이션 탐지
FRAME_QUALITY  = 85              # JPEG 품질 (0-100)
MAX_FRAME_W    = 854             # 리사이즈 상한 (px)
MAX_FRAME_H    = 480

# ── API / 재시도 ──────────────────────────────────────────────────────────────
MAX_RETRIES        = 3
RETRY_DELAY_BASE   = 2.0    # 지수 백오프 기준 (초)
REQUEST_TIMEOUT    = 180    # API 호출당 타임아웃 (초)
MAX_TOKENS_STAGE1  = 2048   # Stage 1 (hallucination check)
MAX_TOKENS_STAGE12 = 2048   # Stage 1·2 최대 응답 토큰 (하위 호환)
MAX_TOKENS_STAGE2  = 4096   # Stage 2 (ODD JSON — 15+ 필드로 길어짐)
MAX_TOKENS_STAGE3  = 1024   # Stage 3 (부분 재검증)
MAX_TOKENS_STAGE4  = 1024   # Stage 4 (caption)

# ── 검증 기준 ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CR_CONF_THRESHOLD", "0.7"))

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
