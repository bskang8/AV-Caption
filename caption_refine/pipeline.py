"""
단일 클립에 대한 4-Stage 파이프라인 오케스트레이션.

ClipResult를 반환하고, 각 출력 파일을 OUTPUT_ROOT에 저장한다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from caption_refine.config import (
    CAPTION_OUT_DIR,
    CAPTION_SUFFIX,
    CAPTIONS_DIR,
    DIFF_OUT_DIR,
    ODD_OUT_DIR,
    VIDEO_SUFFIX,
    VIDEOS_DIR,
)
from caption_refine.cosmos_client import CosmosClient
from caption_refine.stages import stage1_ground, stage2_extract, stage3_verify, stage4_refine
from caption_refine.stages.stage2_extract import ExtractionResult

log = logging.getLogger(__name__)


@dataclass
class ClipResult:
    clip_id:          str
    status:           str          # "ok" | "no_video" | "no_caption" | "error"
    original_caption: str = ""
    refined_caption:  str = ""
    odd_structured:   dict | None = None
    diff:             dict | None = None
    error:            str = ""


def _video_path(clip_id: str) -> Path:
    return VIDEOS_DIR / (clip_id + VIDEO_SUFFIX)


def _caption_path(clip_id: str) -> Path:
    return CAPTIONS_DIR / (clip_id + CAPTION_SUFFIX)


def _save_outputs(result: ClipResult) -> None:
    if result.odd_structured:
        (ODD_OUT_DIR / f"{result.clip_id}.json").write_text(
            json.dumps(result.odd_structured, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if result.refined_caption:
        (CAPTION_OUT_DIR / (result.clip_id + CAPTION_SUFFIX)).write_text(
            result.refined_caption, encoding="utf-8"
        )
    if result.diff:
        (DIFF_OUT_DIR / f"{result.clip_id}_diff.json").write_text(
            json.dumps(result.diff, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


async def process_clip(clip_id: str, client: CosmosClient) -> ClipResult:
    vid  = _video_path(clip_id)
    cap  = _caption_path(clip_id)

    if not vid.exists():
        return ClipResult(clip_id=clip_id, status="no_video")
    if not cap.exists():
        return ClipResult(clip_id=clip_id, status="no_caption")

    original_caption = cap.read_text(encoding="utf-8", errors="replace")

    try:
        # ── Stage 1: hallucination check ─────────────────────────────────────
        log.info("[%s] Stage 1: grounding check", clip_id[:8])
        grounding = await stage1_ground.run(client, vid, original_caption)

        # ── Stage 2: structured ODD extraction ───────────────────────────────
        log.info("[%s] Stage 2: ODD extraction", clip_id[:8])
        extraction: ExtractionResult = await stage2_extract.run(client, vid)

        # ── Stage 3: self-verification (low-confidence fields only) ──────────
        log.info("[%s] Stage 3: self-verify (%d low-conf fields)",
                 clip_id[:8], len(extraction.low_confidence))
        verified_odd = await stage3_verify.run(client, vid, extraction)

        # ── Stage 4: caption refinement ───────────────────────────────────────
        log.info("[%s] Stage 4: caption refine", clip_id[:8])
        refined = await stage4_refine.run(
            client, vid, original_caption, grounding, verified_odd
        )

        # 저장용 구조화 ODD (기존 스키마 호환 + 확장 필드)
        odd_out = {
            "clip_id":       clip_id,
            "odd_compat":    extraction.to_odd_dict(),   # 기존 odd_tags 스키마
            "odd_extended":  verified_odd,               # confidence + evidence 포함
        }

        diff_out = {
            "clip_id":     clip_id,
            "grounded":    grounding.grounded,
            "hallucinated": grounding.hallucinated,
            "missed":      grounding.missed,
            "low_conf_fields": list(extraction.low_confidence.keys()),
        }

        result = ClipResult(
            clip_id=clip_id,
            status="ok",
            original_caption=original_caption,
            refined_caption=refined,
            odd_structured=odd_out,
            diff=diff_out,
        )
        _save_outputs(result)
        log.info("[%s] Done — hal=%d missed=%d low_conf=%d",
                 clip_id[:8],
                 len(grounding.hallucinated),
                 len(grounding.missed),
                 len(extraction.low_confidence))
        return result

    except Exception as exc:
        log.exception("[%s] Pipeline error: %s", clip_id[:8], exc)
        return ClipResult(clip_id=clip_id, status="error", error=str(exc))
