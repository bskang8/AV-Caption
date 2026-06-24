"""
Stage 5 — 최종 캡션 합성

입력:
  - video (최종 확인용 프레임)
  - nl_scenario: Stage 1의 4섹션 NL 시나리오
  - odd_final: Stage 4에서 충돌 해소된 최종 ODD
  - sensor_confirmed: Phase 1 센서 확정값
  - crossval: Stage 4 교차검증 결과

실패 시 nl_scenario를 fallback으로 반환.
"""
from __future__ import annotations

import logging
from pathlib import Path

from caption_refine_v2.config import MAX_TOKENS_STAGE5, NUM_FRAMES_STAGE5
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.prompts import stage5_final_caption
from caption_refine_v2.stages.stage4_crossval import CrossValResult

log = logging.getLogger(__name__)


async def run(
    client: CosmosClient,
    video_path: str | Path,
    nl_scenario: str,
    odd_final: dict,
    sensor_confirmed: dict,
    crossval: CrossValResult,
    motion_narrative: str = "",
) -> str:
    """최종 캡션 생성. 실패 시 nl_scenario 반환."""
    if not nl_scenario and not odd_final:
        log.error("Stage 5: no NL scenario and no ODD — cannot synthesize.")
        return ""

    nl_challenges = crossval.nl_implied.get("implied_challenges", [])
    if not isinstance(nl_challenges, list):
        nl_challenges = []

    prompt = stage5_final_caption(
        nl_scenario=nl_scenario or "(no scenario available)",
        odd_final=odd_final,
        sensor_confirmed=sensor_confirmed,
        conflicts=crossval.conflicts,
        consistency_score=crossval.consistency_score,
        nl_challenges=nl_challenges,
        motion_narrative=motion_narrative,
    )

    try:
        result = await client.chat_text(
            video_path,
            prompt,
            max_tokens=MAX_TOKENS_STAGE5,
            force_frames=True,
            n_frames=NUM_FRAMES_STAGE5,
        )
        result = result.strip()
        if not result:
            raise ValueError("empty response")
        log.info("Stage 5 done — final caption %d chars", len(result))
        return result

    except Exception as exc:
        log.error("Stage 5 failed: %s — falling back to NL scenario", exc)
        return nl_scenario
