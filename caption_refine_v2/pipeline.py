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
단일 클립에 대한 5-Stage + Phase 1 파이프라인 오케스트레이션.

실행 순서:
  Phase 1 (선택):  센서 메타데이터 → 확정 속성 (곡률·IMU·속도·타임스탬프)
  Stage 1+2 (병렬): video → NL 시나리오 + ODD JSON (독립 실행)
  Stage 3:          unknown 필드 재확인
  Stage 4:          NL ↔ ODD 교차검증 → 충돌 해소 + consistency_score
  Stage 5:          video + 모든 결과 → 최종 캡션

출력 파일:
  odd/{clip_id}.json          — odd_raw / odd_final / odd_compat / sensor_confirmed / _meta
  captions/{clip_id}.txt      — 최종 캡션
  crossval/{clip_id}_crossval.json — nl_implied / conflicts / consistency_score
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from caption_refine_v2.config import (
    CAPTION_OUT_DIR,
    CAPTION_SUFFIX,
    CROSSVAL_OUT_DIR,
    CURVATURE_GENTLE_M,
    CURVATURE_STRAIGHT_M,
    DUSK_DAWN_WINDOW,
    EGOMOTION_DIR,
    GRADIENT_FLAT_DEG,
    META_DIR,
    ODD_OUT_DIR,
    SPEED_LOW_KPH,
    SPEED_MID_KPH,
    SUNRISE_HOUR,
    SUNSET_HOUR,
    VIDEO_SUFFIX,
    VIDEOS_DIR,
    VLLM_MODEL,
)
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.stages import (
    stage0_motion,
    stage1_caption,
    stage2_odd,
    stage3_verify,
    stage4_crossval,
    stage5_refine,
)
from caption_refine_v2.stages.stage2_odd import ExtractionResult, odd_compat_from_grouped
from caption_refine_v2.stages.stage4_crossval import CrossValResult

log = logging.getLogger(__name__)


# ── Phase 1: 센서 확정 속성 ───────────────────────────────────────────────────

@dataclass
class ClipMeta:
    """클립 센서 메타데이터. 없는 필드는 None."""
    curvature_radius_m:  float | None = None  # 주행 경로 곡률 반경
    imu_pitch_mean_deg:  float | None = None  # IMU pitch 평균 (경사도)
    speed_mean_kph:      float | None = None  # 평균 속도
    timestamp_hour:      int   | None = None  # 타임스탬프 시각 (KST, 0~23)


def extract_phase1(meta: ClipMeta) -> dict:
    """
    센서 데이터로 확정 가능한 ODD 속성 계산.
    반환: 확정된 속성만 포함한 dict (누락 가능).
    """
    result: dict = {}

    if meta.curvature_radius_m is not None:
        r = meta.curvature_radius_m
        result["road_curvature"] = (
            "straight" if r > CURVATURE_STRAIGHT_M
            else ("gentle" if r > CURVATURE_GENTLE_M else "sharp")
        )

    if meta.imu_pitch_mean_deg is not None:
        p = meta.imu_pitch_mean_deg
        result["road_gradient"] = (
            "flat" if abs(p) < GRADIENT_FLAT_DEG
            else ("uphill" if p > 0 else "downhill")
        )

    if meta.speed_mean_kph is not None:
        v = meta.speed_mean_kph
        result["speed_range"] = (
            "low" if v < SPEED_LOW_KPH
            else ("mid" if v < SPEED_MID_KPH else "high")
        )

    if meta.timestamp_hour is not None:
        h = meta.timestamp_hour
        if SUNRISE_HOUR + DUSK_DAWN_WINDOW < h < SUNSET_HOUR - DUSK_DAWN_WINDOW:
            result["time_of_day"] = "day"
        elif (abs(h - SUNRISE_HOUR) <= DUSK_DAWN_WINDOW
              or abs(h - SUNSET_HOUR) <= DUSK_DAWN_WINDOW):
            result["time_of_day"] = "dusk_dawn"
        else:
            result["time_of_day"] = "night"

    return result


def _load_clip_meta(clip_id: str) -> ClipMeta | None:
    """{clip_id}.meta.json 파일에서 센서 메타데이터 로드. 없으면 None."""
    meta_file = META_DIR / f"{clip_id}.meta.json"
    if not meta_file.exists():
        return None
    try:
        data = json.loads(meta_file.read_text())
        return ClipMeta(
            curvature_radius_m = data.get("curvature_radius_m"),
            imu_pitch_mean_deg = data.get("imu_pitch_mean_deg"),
            speed_mean_kph     = data.get("speed_mean_kph"),
            timestamp_hour     = data.get("timestamp_hour"),
        )
    except Exception as exc:
        log.warning("Failed to load clip meta for %s: %s", clip_id[:8], exc)
        return None


# ── ClipResult ────────────────────────────────────────────────────────────────

@dataclass
class ClipResult:
    clip_id:          str
    status:           str           # "ok" | "no_video" | "error"
    nl_scenario:      str = ""      # Stage 1 NL 시나리오
    final_caption:    str = ""      # Stage 5 최종 캡션
    odd_structured:   dict | None = None
    crossval:         dict | None = None
    error:            str = ""


# ── 경로 헬퍼 ─────────────────────────────────────────────────────────────────

def _video_path(clip_id: str) -> Path:
    return VIDEOS_DIR / (clip_id + VIDEO_SUFFIX)


def _save_outputs(result: ClipResult) -> None:
    if result.odd_structured:
        (ODD_OUT_DIR / f"{result.clip_id}.json").write_text(
            json.dumps(result.odd_structured, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if result.final_caption:
        (CAPTION_OUT_DIR / (result.clip_id + CAPTION_SUFFIX)).write_text(
            result.final_caption, encoding="utf-8"
        )
    if result.crossval:
        (CROSSVAL_OUT_DIR / f"{result.clip_id}_crossval.json").write_text(
            json.dumps(result.crossval, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────

async def process_clip(
    clip_id: str,
    client: CosmosClient,
    clip_meta: ClipMeta | None = None,
) -> ClipResult:
    vid = _video_path(clip_id)
    if not vid.exists():
        return ClipResult(clip_id=clip_id, status="no_video")

    try:
        # ── Phase 1: Stage 0 (Motion Pre-processing) + 센서 확정 ─────────────────
        motion_summary = stage0_motion.load_motion_summary(
            clip_id, EGOMOTION_DIR, timestamp_hour=None
        )
        # data_collection의 hour_of_day를 timestamp_hour로 활용 가능하나
        # 현재는 egomotion만으로 처리. 외부 주입은 배치러너에서 확장 가능.

        if not motion_summary.valid:
            # fallback: 기존 meta.json 방식
            if clip_meta is None:
                clip_meta = _load_clip_meta(clip_id)
            if clip_meta is not None:
                sensor_confirmed = extract_phase1(clip_meta)
            else:
                sensor_confirmed = {}
        else:
            # Stage 0 결과로 Phase 1 확정값 생성
            meta_dict = motion_summary.to_phase1_dict()
            clip_meta = ClipMeta(
                curvature_radius_m = meta_dict["curvature_radius_m"],
                imu_pitch_mean_deg = meta_dict["imu_pitch_mean_deg"],
                speed_mean_kph     = meta_dict["speed_mean_kph"],
                timestamp_hour     = meta_dict["timestamp_hour"],
            )
            sensor_confirmed = extract_phase1(clip_meta)

            # sensor_confirmed 수치 보강: category 레이블만으로는 Stage 5가 세부 정보를 놓침
            if motion_summary.segments:
                spd_s = round(motion_summary.segments[0].speed_start_kph)
                spd_e = round(motion_summary.segments[-1].speed_end_kph)
                if abs(spd_e - spd_s) >= 20 and "speed_range" in sensor_confirmed:
                    # 20 km/h 이상 속도 변화가 있을 때 실제 범위 병기
                    sensor_confirmed["speed_range"] += f" ({spd_s}→{spd_e} km/h)"
                elif "speed_range" in sensor_confirmed:
                    sensor_confirmed["speed_range"] += f" (avg {round(motion_summary.speed_mean_kph)} km/h)"

            if motion_summary.imu_pitch_mean_deg is not None and "road_gradient" in sensor_confirmed:
                p = motion_summary.imu_pitch_mean_deg
                sensor_confirmed["road_gradient"] += f" (mean {p:+.1f}°)"

            if sensor_confirmed:
                log.info("[%s] Phase 1 (Stage 0): confirmed %s", clip_id[:8], list(sensor_confirmed.keys()))

        # ── Stage 1+2: 병렬 실행 (독립, 서로 참조 안 함) ─────────────────────
        log.info("[%s] Stage 1+2: NL scenario & ODD extraction (parallel)", clip_id[:8])
        nl_scenario, extraction = await asyncio.gather(
            stage1_caption.run(client, vid, motion_summary.narrative if motion_summary.valid else ""),
            stage2_odd.run(client, vid),
        )

        # ── Stage 3: unknown 필드 재확인 ─────────────────────────────────────
        log.info("[%s] Stage 3: resolving %d unknown fields",
                 clip_id[:8], len(extraction.unknown_fields))
        odd_grouped = await stage3_verify.run(client, vid, extraction, sensor_confirmed=sensor_confirmed)

        # ── Stage 4: 교차검증 (NL → implied ODD + 규칙 기반 충돌 감지) ───────
        log.info("[%s] Stage 4: cross-validation (text-only)", clip_id[:8])
        crossval: CrossValResult = await stage4_crossval.run(
            client, nl_scenario, odd_grouped
        )

        # ── Stage 5: 최종 캡션 합성 ──────────────────────────────────────────
        log.info("[%s] Stage 5: final caption synthesis", clip_id[:8])
        final_caption = await stage5_refine.run(
            client, vid, nl_scenario, crossval.odd_final,
            sensor_confirmed, crossval,
            motion_narrative=motion_summary.narrative if motion_summary.valid else "",
        )

        # ── 출력 구조 조립 ─────────────────────────────────────────────────
        odd_out = {
            "clip_id":          clip_id,
            "sensor_confirmed": sensor_confirmed,
            "odd_raw":          extraction.to_grouped_dict(),
            "odd_final":        crossval.odd_final,
            "odd_compat":       odd_compat_from_grouped(crossval.odd_final),
            "_meta": {
                "unknown_ratio":     round(extraction.unknown_ratio, 3),
                "consistency_score": crossval.consistency_score,
                "model":             VLLM_MODEL,
                "tagging_version":   "2.0",
            },
        }

        crossval_out = {
            "clip_id":           clip_id,
            "nl_implied":        crossval.nl_implied,
            "conflicts":         crossval.conflicts,
            "consistency_score": crossval.consistency_score,
            "quality_flag":      crossval.quality_flag(),
        }

        result = ClipResult(
            clip_id=clip_id,
            status="ok",
            nl_scenario=nl_scenario,
            final_caption=final_caption,
            odd_structured=odd_out,
            crossval=crossval_out,
        )
        _save_outputs(result)
        log.info(
            "[%s] Done — unknown=%.2f conflicts=%d consistency=%.2f [%s]",
            clip_id[:8],
            extraction.unknown_ratio,
            len(crossval.conflicts),
            crossval.consistency_score,
            crossval.quality_flag(),
        )
        return result

    except Exception as exc:
        log.exception("[%s] Pipeline error: %s", clip_id[:8], exc)
        return ClipResult(clip_id=clip_id, status="error", error=str(exc))