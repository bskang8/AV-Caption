"""
8B 모델 (port 8001) 으로 5클립 caption_refine 파이프라인 테스트.
"""
import asyncio
import logging
import os
import sys

# 환경변수 설정 (import 전)
os.environ["CR_VLLM_URL"]      = "http://localhost:8001/v1"
os.environ["CR_VLLM_MODEL"]    = "nvidia/Cosmos-Reason2-8B"
os.environ["CR_VIDEOS_DIR"]    = "/Data1/home/bskang/cds-data/front_camera_videos"  # allowed-local-media-path 하위
os.environ["CR_CAPTIONS_DIR"]  = "/Data1/home/bskang/cds-data/caption_v2/captions"
os.environ["CR_OUTPUT_ROOT"]   = "/Data1/home/bskang/cds-data/caption_v2_8b_video"
os.environ["CR_VIDEO_MODE"]    = "video"   # file:// URL로 MP4 직접 전송
os.environ["CR_VIDEO_FPS"]     = "1"       # 20s × 1fps = 20프레임 → ~28,540 tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from caption_refine.cosmos_client import CosmosClient
from caption_refine.pipeline import process_clip

CLIP_IDS = [
    "47302f29-4895-4ed0-8053-11601eec80e1",
    "6c4c8acb-2179-491d-9bca-c88242a5c4ef",
    "e73225e5-9341-40e6-90e4-706c5ffa2b09",
    "e8770620-5491-4b43-ba35-8f4efcc8d660",
    "ef742bb7-c767-4848-a15b-7d39c565b45e",
]


async def main():
    client = CosmosClient()
    for clip_id in CLIP_IDS:
        print(f"\n{'='*60}")
        print(f"Clip: {clip_id[:8]}")
        result = await process_clip(clip_id, client)
        print(f"Status:      {result.status}")
        if result.status == "ok":
            diff = result.diff or {}
            print(f"Hallucinated ({len(diff.get('hallucinated', []))}):")
            for h in diff.get("hallucinated", []):
                print(f"  - {h[:120]}")
            print(f"Missed ({len(diff.get('missed', []))}):")
            for m in diff.get("missed", []):
                print(f"  + {m[:120]}")
            print(f"\nOriginal:\n  {result.original_caption[:300]}")
            print(f"\nRefined:\n  {result.refined_caption[:400]}")
        else:
            print(f"Error: {result.error}")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
