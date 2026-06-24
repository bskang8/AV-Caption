"""
Stage 1 — Caption Grounding Check

기존 caption이 영상에 없는 내용(hallucination)을 담고 있는지 검증.
반환: {"grounded": [...], "hallucinated": [...], "missed": [...]}

신뢰성 향상을 위해 동일 프롬프트를 2회 병렬 실행하고 hallucinated 합집합을 사용.
bitsandbytes INT8 양자화의 비결정적 특성으로 단일 실행은 탐지를 놓칠 수 있음.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from caption_refine.cosmos_client import CosmosClient
from caption_refine.config import NUM_FRAMES_STAGE1
from caption_refine.prompts import stage1_grounding

log = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Caption 텍스트를 개별 문장으로 분리."""
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [s.strip() for s in sentences if s.strip()]


_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'and', 'or', 'but', 'not',
    'it', 'its', 'this', 'that', 'these', 'those', 'also', 'which', 'while',
})


def _word_set(text: str) -> set[str]:
    return set(re.findall(r'[a-z]+', text.lower()))


def _content_words(text: str) -> set[str]:
    return _word_set(text) - _STOPWORDS


def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _coverage(item: str, sentence: str) -> float:
    """item의 내용어 중 sentence에 포함된 비율."""
    iw = _content_words(item)
    if not iw:
        return 0.0
    return len(iw & _content_words(sentence)) / len(iw)


@dataclass
class GroundingResult:
    grounded:     list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    missed:       list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "grounded":     self.grounded,
            "hallucinated": self.hallucinated,
            "missed":       self.missed,
        }


_GENERIC_PHRASES = (
    "road is clean", "road is part of", "driver's vehicle", "driver remains",
    "driver follows", "driver's actions", "road is wide", "road is well",
    "road surface shows", "ego-vehicle maintains", "driver's attention",
)


def _parse_hal_indices(data: dict | list, sentences: list[str]) -> tuple[set[int], dict[int, str]]:
    """응답 데이터에서 hallucinated sentence 인덱스와 이유를 파싱."""
    if isinstance(data, list):
        data = {"hallucinated": data, "missed": []}
    if not isinstance(data, dict):
        return set(), {}

    raw_hal = data.get("hallucinated", [])
    if not isinstance(raw_hal, list):
        return set(), {}

    indices: set[int] = set()
    reasons: dict[int, str] = {}

    for item in raw_hal:
        if isinstance(item, dict):
            try:
                i = int(item.get("sentence_num", 0)) - 1
                if not (0 <= i < len(sentences)):
                    continue
                reason = str(item.get("reason", "")).strip()
                sentence = sentences[i]
                if _jaccard(reason, sentence) > 0.5:
                    continue
                indices.add(i)
                reasons[i] = reason
            except (TypeError, ValueError):
                pass
        elif isinstance(item, (int, float)):
            i = int(item) - 1
            if 0 <= i < len(sentences):
                indices.add(i)
    return indices, reasons


def _parse_missed(data: dict | list, sentences: list[str]) -> list[str]:
    """응답 데이터에서 missed 항목을 파싱하고 필터링."""
    if isinstance(data, list):
        raw_missed = []
    elif isinstance(data, dict):
        raw_missed = data.get("missed", [])
        if not isinstance(raw_missed, list):
            raw_missed = []
    else:
        return []

    seen: set[str] = set()
    missed: list[str] = []
    for item in raw_missed:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item:
            continue
        key = item.lower()
        if "specific thing clearly visible" in key:
            continue
        if key in seen:
            continue
        if any(_jaccard(item, s) > 0.35 or _coverage(item, s) > 0.55 for s in sentences):
            continue
        if any(g in key for g in _GENERIC_PHRASES):
            continue
        seen.add(key)
        missed.append(item)

    return sorted(missed, key=len, reverse=True)[:5]


_STAGE1_TEMPERATURE = 0.3   # 탐지 다양성 확보 (temperature=0 이면 두 패스가 동일한 결과를 냄)
_STAGE1_PASSES     = 3      # 병렬 패스 수 — 합집합으로 재현성 확보


async def _single_pass(
    client: CosmosClient,
    video_path: str | Path,
    prompt: str,
) -> dict | list | None:
    """단일 Stage 1 API 호출. 실패 시 None 반환."""
    try:
        data = await client.chat_json(video_path, prompt, force_frames=True,
                                      n_frames=NUM_FRAMES_STAGE1,
                                      temperature=_STAGE1_TEMPERATURE)
        return data
    except Exception as exc:
        log.warning("Stage 1 pass failed: %s", exc)
        return None


async def run(
    client: CosmosClient,
    video_path: str | Path,
    existing_caption: str,
) -> GroundingResult:
    sentences = _split_sentences(existing_caption)
    if not sentences:
        log.warning("Stage 1: no sentences extracted from caption")
        return GroundingResult()

    log.debug("Stage 1: %d sentences — running %d parallel passes (temp=%.1f)",
              len(sentences), _STAGE1_PASSES, _STAGE1_TEMPERATURE)
    prompt = stage1_grounding(sentences)

    # N회 병렬 실행 → hallucinated 합집합 (temperature>0 으로 패스마다 다른 결과 유도)
    results = await asyncio.gather(
        *[_single_pass(client, video_path, prompt) for _ in range(_STAGE1_PASSES)]
    )

    merged_indices: set[int] = set()
    merged_reasons: dict[int, str] = {}
    all_missed_raw: list[str] = []

    for data in results:
        if data is None:
            continue
        indices, reasons = _parse_hal_indices(data, sentences)
        merged_indices |= indices
        for i, r in reasons.items():
            if i not in merged_reasons:
                merged_reasons[i] = r
        missed_items = _parse_missed(data, sentences)
        all_missed_raw.extend(missed_items)

    if all(r is None for r in results):
        log.error("Stage 1: all %d passes failed", _STAGE1_PASSES)
        return GroundingResult()

    pass_sets = [
        _parse_hal_indices(r, sentences)[0] if r else set()
        for r in results
    ]
    log.debug("Stage 1: passes=%s → union=%s", pass_sets, merged_indices)

    hallucinated = [
        f"{sentences[i]} [{merged_reasons[i]}]" if i in merged_reasons else sentences[i]
        for i in sorted(merged_indices)
    ]
    grounded = [s for i, s in enumerate(sentences) if i not in merged_indices]

    # missed 합집합 (중복 제거, 최대 5개)
    seen: set[str] = set()
    missed: list[str] = []
    for item in sorted(all_missed_raw, key=len, reverse=True):
        key = item.lower()
        if key not in seen:
            seen.add(key)
            missed.append(item)
        if len(missed) >= 5:
            break

    log.info("Stage 1 done — hal=%d missed=%d (%d-pass union, temp=%.1f)",
             len(hallucinated), len(missed), _STAGE1_PASSES, _STAGE1_TEMPERATURE)
    return GroundingResult(
        grounded=grounded,
        hallucinated=hallucinated,
        missed=missed,
    )
