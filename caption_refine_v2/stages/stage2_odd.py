"""
Stage 2 — 구조화 ODD 추출 (Phase 2-A)

video → 4차원 ODD JSON.
confidence/evidence 없음. 불확실하면 "unknown" 사용.

4차원:
  road_structure   : road_type, lanes_ego_direction, lanes_opposite, road_divider,
                     lane_marking_quality, junction_proximity
  environment      : lighting_condition, precipitation, fog, road_surface, backlight
  dynamic_elements : traffic_density, road_user_types, construction_zone, special_event
  scene_complexity : occlusion_level, visibility_range, scene_ambiguity, unexpected_element

Phase 1 센서 확정값(road_curvature, road_gradient, speed_range, time_of_day)은
pipeline.py의 extract_phase1()에서 별도 처리 후 sensor_confirmed로 병합.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from caption_refine_v2.config import (
    MAX_TOKENS_STAGE2,
    NUM_FRAMES_STAGE2,
    UNKNOWN_RATIO_WARN,
    UNKNOWN_RATIO_CRITICAL,
)
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.prompts import stage2_odd_extract

log = logging.getLogger(__name__)

# VLM이 추출하는 모든 필드 (group → [fields])
_ODD_SCHEMA: dict[str, list[str]] = {
    "road_structure":   ["road_type", "lanes_ego_direction", "lanes_opposite", "road_divider",
                         "lane_marking_quality", "junction_proximity"],
    "environment":      ["lighting_condition", "precipitation", "fog", "road_surface", "backlight"],
    "dynamic_elements": ["traffic_density", "road_user_types", "construction_zone", "special_event"],
    "scene_complexity": ["occlusion_level", "visibility_range", "scene_ambiguity", "unexpected_element"],
}


@dataclass
class ExtractionResult:
    road_structure:   dict = field(default_factory=dict)
    environment:      dict = field(default_factory=dict)
    dynamic_elements: dict = field(default_factory=dict)
    scene_complexity: dict = field(default_factory=dict)
    unknown_fields:   list[str] = field(default_factory=list)  # "group.field" 경로
    unknown_ratio:    float = 0.0

    def to_grouped_dict(self) -> dict:
        return {
            "road_structure":   self.road_structure,
            "environment":      self.environment,
            "dynamic_elements": self.dynamic_elements,
            "scene_complexity": self.scene_complexity,
        }

    def to_odd_compat(self) -> dict:
        """Stage 2 원본 ExtractionResult → 호환 레코드 (Stage 3 이전 값)."""
        return odd_compat_from_grouped({
            "road_structure":   self.road_structure,
            "environment":      self.environment,
            "dynamic_elements": self.dynamic_elements,
            "scene_complexity": self.scene_complexity,
        })


def odd_compat_from_grouped(grouped: dict) -> dict:
    """
    odd_final 구조 딕셔너리(grouped) → 호환 레코드.
    Stage 3까지 반영된 최종값을 pipeline.py에서 직접 변환할 때 사용.
    """
    road  = grouped.get("road_structure", {})
    env   = grouped.get("environment", {})
    dyn   = grouped.get("dynamic_elements", {})
    scene = grouped.get("scene_complexity", {})

    precip  = env.get("precipitation", "unknown")
    weather = precip if precip in ("rain", "snow") else ("clear" if precip == "none" else "unknown")

    ego = road.get("lanes_ego_direction", "unknown")
    opp = road.get("lanes_opposite", "unknown")
    lane_summary = f"ego:{ego} opp:{opp}" if ego != "unknown" or opp != "unknown" else "unknown"

    return {
        "road_type":           road.get("road_type", "unknown"),
        "weather":             weather,
        "traffic_density":     dyn.get("traffic_density", "unknown"),
        "agent_type":          dyn.get("road_user_types", "unknown"),
        "lanes_ego_direction": ego,
        "lanes_opposite":      opp,
        "lane_summary":        lane_summary,
        "road_divider":        road.get("road_divider", "unknown"),
        "lighting":            env.get("lighting_condition", "unknown"),
        "scene_ambiguity":     scene.get("scene_ambiguity", "unknown"),
    }


def _parse_grouped(data: dict) -> ExtractionResult:
    """VLM 응답 dict를 ExtractionResult로 파싱. 스키마 외 키 무시."""
    groups: dict[str, dict] = {}
    unknown_fields: list[str] = []
    total = 0

    for group_key, field_names in _ODD_SCHEMA.items():
        raw_group = data.get(group_key, {})
        parsed: dict = {}
        for fname in field_names:
            val = raw_group.get(fname, "unknown") if isinstance(raw_group, dict) else "unknown"
            # 빈 값·None → unknown 정규화
            if not val or not isinstance(val, str):
                val = "unknown"
            parsed[fname] = val
            total += 1
            if val == "unknown":
                unknown_fields.append(f"{group_key}.{fname}")
        groups[group_key] = parsed

    unknown_ratio = len(unknown_fields) / total if total > 0 else 0.0

    return ExtractionResult(
        road_structure=groups.get("road_structure", {}),
        environment=groups.get("environment", {}),
        dynamic_elements=groups.get("dynamic_elements", {}),
        scene_complexity=groups.get("scene_complexity", {}),
        unknown_fields=unknown_fields,
        unknown_ratio=unknown_ratio,
    )


async def run(client: CosmosClient, video_path: str | Path) -> ExtractionResult:
    prompt = stage2_odd_extract()
    try:
        data = await client.chat_json(
            video_path,
            prompt,
            max_tokens=MAX_TOKENS_STAGE2,
            n_frames=NUM_FRAMES_STAGE2,
        )
    except Exception as exc:
        log.error("Stage 2 failed: %s", exc)
        return ExtractionResult()

    if not isinstance(data, dict):
        log.warning("Stage 2: unexpected response type %s", type(data))
        return ExtractionResult()

    result = _parse_grouped(data)

    # unknown 비율 경고
    if result.unknown_ratio > UNKNOWN_RATIO_CRITICAL:
        log.warning("Stage 2: unknown_ratio=%.2f > %.2f — check video quality",
                    result.unknown_ratio, UNKNOWN_RATIO_CRITICAL)
    elif result.unknown_ratio > UNKNOWN_RATIO_WARN:
        log.warning("Stage 2: unknown_ratio=%.2f > %.2f — consider prompt review",
                    result.unknown_ratio, UNKNOWN_RATIO_WARN)

    log.info("Stage 2 done — %d unknown fields (ratio=%.2f)",
             len(result.unknown_fields), result.unknown_ratio)
    return result
