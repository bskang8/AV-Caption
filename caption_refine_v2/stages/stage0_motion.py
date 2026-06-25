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
Stage 0 — Motion Pre-processing

egomotion.offline.parquet → MotionSummary
  ① to_phase1_dict()  → pipeline ClipMeta 호환 (Phase 1 센서 확정값)
  ② narrative (str)   → Stage 1 / Stage 5 프롬프트 주입용 동적 서사

알고리즘:
  1. 이동 평균 스무딩 (window=5) → 유한 차분으로 vx,vy,vz,ax,ay 계산
  2. 2D 곡률 κ = (vx·ay - vy·ax) / (vx²+vy²)^1.5  (부호: + = 좌회전)
  3. 쿼터니언 → pitch(경사) / yaw(heading) 추출
  4. Heading-aligned 측방 변위 누적 → 차선변경 감지
  5. 이벤트 감지: 회전 / 차선변경 / 가감속
  6. 5초 세그먼트 + 이벤트 → narrative 텍스트 생성
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── 임계값 ────────────────────────────────────────────────────────────────────
_SMOOTH_WIN          = 5       # 이동 평균 윈도우 (샘플 수, ~10Hz 기준 0.5s)
_KAPPA_STRAIGHT      = 1/500   # 직선: |κ| < 0.002  (turn_events / LC 감지용)
_KAPPA_GENTLE        = 1/200   # gentle curve: 0.002 ≤ |κ| < 0.005 (turn_events용)
# sharp: |κ| ≥ 0.005

# Segment 라벨 전용 임계값 — turn_events보다 보수적:
# κ는 2차 미분이라 노이즈가 크므로 세그먼트 평균 κ 기준으로 더 넓은 "직선" 범위 사용
_KAPPA_SEG_STRAIGHT  = 1/300   # segment 직선: 세그먼트 평균 |κ| < 0.0033 (r > 300m)
_KAPPA_SEG_GENTLE    = 1/100   # segment gentle: 0.0033 ≤ 평균 |κ| < 0.01 (100m < r < 300m)
# segment sharp: 평균 |κ| ≥ 0.01 (r < 100m)

_TURN_MIN_S          = 2.0     # 회전 이벤트 최소 지속 시간 (s)
_LC_DISP_M           = 2.5     # 차선변경 측방 변위 임계값 (m)
_LC_MIN_S            = 2.0     # 차선변경 최소 지속 시간 (s)
_LC_MAX_S            = 8.0     # 차선변경 최대 지속 시간 (s)
_ACCEL_DELTA_KPH     = 10.0    # 가감속 이벤트: 5s 내 속도 변화 임계값
_ACCEL_WINDOW_S      = 5.0
_SEG_SEC             = 5.0     # 세그먼트 길이 (s)
_GRAD_FLAT_DEG       = 1.5     # 경사 flat 임계값 (degrees)
_MIN_ROWS            = 10      # 최소 유효 행 수
_STOP_SPEED_KPH      = 3.0     # 정지 판정 속도 임계값 (km/h)
_STOP_MIN_S          = 1.0     # 정지 이벤트 최소 지속 시간 (s)
_SPEED_RANGE_THR_KPH = 8.0     # 세그먼트 내 속도 변화가 이 이상이면 range 표시
# 가감속 강도 분류 임계값 (m/s²)
_HARD_ACCEL_MS2      = 2.5
_MODERATE_ACCEL_MS2  = 1.0
# 클립 수준 기동 분류 임계값 (degrees)
_MANEUVER_UTURN_DEG      = 150.0   # |net| > 150° → U턴
_MANEUVER_ROUNDABOUT_DEG = 270.0   # total > 270° → 로터리
_MANEUVER_ROUNDABOUT_NET = 60.0    # 로터리 판정 시 |net| < 60° (순 방향 변화 작음)
_MANEUVER_TURN_MIN_DEG   = 60.0    # 교차로 회전: |net| 60°~135°
_MANEUVER_TURN_MAX_DEG   = 135.0
_MANEUVER_WINDING_RATIO  = 2.5     # 굴곡 도로: total / |net| > 2.5
_MANEUVER_WINDING_TOT    = 90.0    # 굴곡 도로: total > 90°
# 키네마틱 도로 유형 힌트 임계값
_HINT_PARKING_KPH        = 10.0    # 주차/저속 기동: 평균 속도 < 10 km/h
_HINT_HIGHWAY_KPH        = 100.0   # 고속도로: 평균 속도 > 100 km/h
_HINT_SHARP_TURN_M       = 50.0    # 급회전 반경 임계값 (m)


# ── 이벤트 데이터클래스 ───────────────────────────────────────────────────────

@dataclass
class TurnEvent:
    t_start: float
    t_end: float
    direction: str    # "left" | "right"
    radius_m: float

@dataclass
class LaneChangeEvent:
    t_start: float
    t_end: float
    direction: str    # "left" | "right"
    disp_m: float

@dataclass
class AccelEvent:
    t_start: float
    t_end: float
    kind: str         # "acceleration" | "deceleration"
    delta_kph: float

@dataclass
class StopEvent:
    t_start: float
    t_end: float
    duration_s: float

@dataclass
class Segment:
    t_start: float
    t_end: float
    speed_start_kph: float
    speed_mean_kph: float
    speed_end_kph: float
    curvature_label: str   # "straight"|"gentle_left"|"gentle_right"|"sharp_left"|"sharp_right"
    gradient_label: str    # "flat"|"uphill"|"downhill"
    gradient_deg: float    # 실제 pitch 평균 (degrees)


# ── 최종 출력 ─────────────────────────────────────────────────────────────────

@dataclass
class MotionSummary:
    # Phase 1 호환 4 필드
    curvature_radius_m:  float | None = None
    imu_pitch_mean_deg:  float | None = None
    speed_mean_kph:      float | None = None
    timestamp_hour:      int   | None = None

    # 확장 통계
    speed_std_kph:       float | None = None
    total_distance_m:    float | None = None
    heading_net_deg:     float | None = None
    heading_total_deg:   float | None = None
    clip_maneuver:       str   | None = None
    road_hint:           str   | None = None
    curvature_dist:      dict  = field(default_factory=dict)

    # 이벤트 목록
    turn_events:         list  = field(default_factory=list)
    lane_change_events:  list  = field(default_factory=list)
    accel_events:        list  = field(default_factory=list)
    stop_events:         list  = field(default_factory=list)

    # 세그먼트
    segments:            list  = field(default_factory=list)

    # 서사 텍스트
    narrative:           str   = ""

    valid:               bool  = False

    def to_phase1_dict(self) -> dict:
        """pipeline.py ClipMeta 호환 딕셔너리."""
        return {
            "curvature_radius_m": self.curvature_radius_m,
            "imu_pitch_mean_deg": self.imu_pitch_mean_deg,
            "speed_mean_kph":     self.speed_mean_kph,
            "timestamp_hour":     self.timestamp_hour,
        }


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = _SMOOTH_WIN) -> np.ndarray:
    """이동 평균 스무딩. 양 끝은 edge padding."""
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    pad    = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _finite_diff(arr: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """유한 차분 미분. 양 끝은 인접값으로 채움."""
    diff       = np.diff(arr)
    dt_safe    = np.where(dt[:-1] > 1e-6, dt[:-1], 1e-6)
    deriv      = diff / dt_safe
    return np.concatenate([[deriv[0]], deriv])


def _quat_to_pitch_yaw(qx, qy, qz, qw) -> tuple[np.ndarray, np.ndarray]:
    """쿼터니언 배열 → pitch(rad), yaw(rad)."""
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    siny  = 2.0 * (qw * qz + qx * qy)
    cosy  = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw   = np.arctan2(siny, cosy)
    return pitch, yaw


def _curvature_label(kappa: float) -> str:
    """스칼라 곡률 → 레이블."""
    abs_k = abs(kappa)
    if abs_k < _KAPPA_STRAIGHT:
        return "straight"
    direction = "left" if kappa > 0 else "right"
    if abs_k < _KAPPA_GENTLE:
        return f"gentle_{direction}"
    return f"sharp_{direction}"


def _curvature_distribution(kappa_arr: np.ndarray) -> dict:
    labels = [_curvature_label(k) for k in kappa_arr]
    counts: dict[str, int] = {}
    for lbl in labels:
        # 방향 없는 카테고리로 집계
        base = lbl.replace("_left", "").replace("_right", "")
        counts[base] = counts.get(base, 0) + 1
    total = len(labels)
    return {k: round(v / total, 3) for k, v in counts.items()}


# ── 이벤트 감지 ───────────────────────────────────────────────────────────────

def _detect_turns(t: np.ndarray, kappa: np.ndarray) -> list[TurnEvent]:
    """연속적으로 |κ| ≥ _KAPPA_GENTLE 인 구간을 회전 이벤트로 추출."""
    in_turn  = np.abs(kappa) >= _KAPPA_GENTLE
    events: list[TurnEvent] = []
    i = 0
    while i < len(t):
        if not in_turn[i]:
            i += 1
            continue
        j = i
        while j < len(t) and in_turn[j]:
            j += 1
        duration = t[j - 1] - t[i]
        if duration >= _TURN_MIN_S:
            seg_kappa = kappa[i:j]
            mean_kappa = np.mean(seg_kappa)
            direction  = "left" if mean_kappa > 0 else "right"
            radius_m   = float(1.0 / max(abs(mean_kappa), 1e-6))
            events.append(TurnEvent(
                t_start=float(t[i]),
                t_end=float(t[j - 1]),
                direction=direction,
                radius_m=round(radius_m, 1),
            ))
        i = j
    return events


def _detect_lane_changes(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    kappa: np.ndarray,
) -> list[LaneChangeEvent]:
    """
    Heading 수직 방향 누적 변위가 _LC_DISP_M 이상인 구간을 차선변경으로 추출.
    회전 구간(_KAPPA_GENTLE 이상)에서는 감지하지 않음.

    참고: 실제 도로 커브가 있는 환경에서 global heading 기준을 사용하면
    커브 누적으로 인한 false positive가 다수 발생함. 따라서 순간 heading(instantaneous)
    기준을 유지한다. 이 방식에서 LC 감지는 주로 VLM 시각 검출로 보완됨.
    """
    dx  = np.diff(x)
    dy  = np.diff(y)
    hdg = yaw[:-1]
    # heading 수직(lateral) 성분: -sin(yaw)*dx + cos(yaw)*dy
    lat_step = -np.sin(hdg) * dx + np.cos(hdg) * dy

    events: list[LaneChangeEvent] = []
    n = len(lat_step)
    i = 0
    while i < n:
        if abs(kappa[i]) >= _KAPPA_GENTLE:
            i += 1
            continue

        found = False
        cumulative = 0.0
        for j in range(i, n):
            cumulative += lat_step[j]
            elapsed = t[j] - t[i]
            if elapsed > _LC_MAX_S:
                break
            if abs(cumulative) >= _LC_DISP_M and elapsed >= _LC_MIN_S:
                direction = "left" if cumulative > 0 else "right"
                events.append(LaneChangeEvent(
                    t_start=float(t[i]),
                    t_end=float(t[j]),
                    direction=direction,
                    disp_m=round(abs(cumulative), 2),
                ))
                i = j + 1
                found = True
                break

        if not found:
            i += 1
    return events


def _detect_accel_events(
    t: np.ndarray,
    speed_kph: np.ndarray,
    kappa: np.ndarray,
) -> list[AccelEvent]:
    """5초 윈도우에서 속도 변화가 _ACCEL_DELTA_KPH 이상인 구간 감지.
    급회전 구간은 속도 미분 노이즈가 크므로 제외."""
    events: list[AccelEvent] = []
    n = len(t)
    i = 0
    while i < n:
        if abs(kappa[i]) >= _KAPPA_GENTLE:
            i += 1
            continue
        j = i
        while j < n and (t[j] - t[i]) < _ACCEL_WINDOW_S:
            if abs(kappa[j]) >= _KAPPA_GENTLE:
                break
            j += 1
        if j > i:
            delta = speed_kph[j - 1] - speed_kph[i]
            if abs(delta) >= _ACCEL_DELTA_KPH:
                kind = "acceleration" if delta > 0 else "deceleration"
                events.append(AccelEvent(
                    t_start=float(t[i]),
                    t_end=float(t[j - 1]),
                    kind=kind,
                    delta_kph=round(abs(delta), 1),
                ))
                i = j
                continue
        i += 1
    # 연속 이벤트 병합 (같은 방향이고 인접)
    merged: list[AccelEvent] = []
    for ev in events:
        if merged and merged[-1].kind == ev.kind and ev.t_start - merged[-1].t_end < 2.0:
            merged[-1] = AccelEvent(
                t_start=merged[-1].t_start, t_end=ev.t_end,
                kind=ev.kind,
                delta_kph=round(merged[-1].delta_kph + ev.delta_kph, 1),
            )
        else:
            merged.append(ev)
    return merged


# ── 세그먼트 빌더 ─────────────────────────────────────────────────────────────

def _build_segments(
    t: np.ndarray,
    speed_kph: np.ndarray,
    kappa: np.ndarray,
    pitch_deg: np.ndarray,
) -> list[Segment]:
    from collections import Counter

    duration = t[-1] - t[0]
    n_segs   = max(1, int(round(duration / _SEG_SEC)))
    seg_boundaries = np.linspace(t[0], t[-1], n_segs + 1)

    segments: list[Segment] = []
    for si in range(n_segs):
        t0, t1 = seg_boundaries[si], seg_boundaries[si + 1]
        mask   = (t >= t0) & (t <= t1)
        if not mask.any():
            continue

        seg_speed  = speed_kph[mask]
        seg_kappa  = kappa[mask]
        seg_pitch  = pitch_deg[mask]

        # 세그먼트 곡률 레이블: 개별 샘플 최빈값(노이즈에 취약) 대신
        # 세그먼트 평균 κ + 보수적 임계값 사용.
        # κ는 2차 미분이라 노이즈가 증폭되므로 개별 샘플로 분류하면
        # 직선 도로도 "sharp"로 오분류될 수 있음.
        mean_kappa = float(np.mean(seg_kappa))
        abs_mk = abs(mean_kappa)
        if abs_mk < _KAPPA_SEG_STRAIGHT:
            dominant = "straight"
        elif abs_mk < _KAPPA_SEG_GENTLE:
            dominant = "gentle_left" if mean_kappa > 0 else "gentle_right"
        else:
            dominant = "sharp_left" if mean_kappa > 0 else "sharp_right"

        # 경사 레이블
        mean_pitch = float(np.mean(seg_pitch))
        if abs(mean_pitch) < _GRAD_FLAT_DEG:
            grad_label = "flat"
        elif mean_pitch > 0:
            grad_label = "uphill"
        else:
            grad_label = "downhill"

        segments.append(Segment(
            t_start=round(t0, 1),
            t_end=round(t1, 1),
            speed_start_kph=round(float(seg_speed[0]), 1),
            speed_mean_kph=round(float(np.mean(seg_speed)), 1),
            speed_end_kph=round(float(seg_speed[-1]), 1),
            curvature_label=dominant,
            gradient_label=grad_label,
            gradient_deg=round(mean_pitch, 1),
        ))
    return segments


# ── Heading Profile & 클립 기동 분류 ─────────────────────────────────────────

def _compute_heading_profile(yaw_rad: np.ndarray) -> tuple[float, float]:
    """yaw 배열 → (net_deg, total_deg). unwrap 후 계산."""
    yaw_uw = np.unwrap(yaw_rad)
    net_deg   = round(float(np.degrees(yaw_uw[-1] - yaw_uw[0])), 1)
    total_deg = round(float(np.degrees(np.sum(np.abs(np.diff(yaw_uw))))), 1)
    return net_deg, total_deg


def _classify_clip_maneuver(net_deg: float, total_deg: float) -> str | None:
    """net/total heading 변화로 클립 수준 기동 유형 분류."""
    abs_net = abs(net_deg)
    dir_str = "left" if net_deg > 0 else "right"

    # 로터리: 총 변화 크고 순 변화 작음 (돌아서 원위치에 가까움)
    if total_deg > _MANEUVER_ROUNDABOUT_DEG and abs_net < _MANEUVER_ROUNDABOUT_NET:
        return f"roundabout (total {total_deg:.0f}°)"

    # U턴
    if abs_net > _MANEUVER_UTURN_DEG:
        return f"U-turn {dir_str} ({net_deg:+.0f}°)"

    # 교차로 회전
    if _MANEUVER_TURN_MIN_DEG <= abs_net <= _MANEUVER_TURN_MAX_DEG:
        return f"intersection turn {dir_str} ({net_deg:+.0f}°)"

    # 굴곡 도로: 총 변화가 순 변화보다 훨씬 크고, 절대값도 충분
    if abs_net > 1e-3 and total_deg > abs_net * _MANEUVER_WINDING_RATIO and total_deg > _MANEUVER_WINDING_TOT:
        return f"winding road (total {total_deg:.0f}°)"

    return None


# ── 정지 감지 ─────────────────────────────────────────────────────────────────

def _detect_stops(t: np.ndarray, speed_kph: np.ndarray) -> list[StopEvent]:
    stopped = speed_kph < _STOP_SPEED_KPH
    events: list[StopEvent] = []
    i = 0
    while i < len(t):
        if not stopped[i]:
            i += 1
            continue
        j = i
        while j < len(t) and stopped[j]:
            j += 1
        t_end_idx = min(j - 1, len(t) - 1)
        duration = float(t[t_end_idx] - t[i])
        if duration >= _STOP_MIN_S:
            events.append(StopEvent(
                t_start=round(float(t[i]), 1),
                t_end=round(float(t[t_end_idx]), 1),
                duration_s=round(duration, 1),
            ))
        i = j
    return events


# ── 서사 생성 ─────────────────────────────────────────────────────────────────

def _format_segment_line(seg: Segment) -> str:
    # 속도 + 경사 + 곡률(직선 아닐 때만) 출력
    t_str = f"{int(seg.t_start)}–{int(seg.t_end)} s"

    # 세그먼트 내 속도 변화가 크면 범위로, 작으면 평균으로 표시
    delta = abs(seg.speed_end_kph - seg.speed_start_kph)
    if delta >= _SPEED_RANGE_THR_KPH:
        sp_str = f"{seg.speed_start_kph:.0f}→{seg.speed_end_kph:.0f} km/h"
    else:
        sp_str = f"{seg.speed_mean_kph:.0f} km/h"

    if seg.gradient_label == "flat":
        grad_str = ""
    else:
        grad_str = f", {seg.gradient_label} {abs(seg.gradient_deg):.1f}°"

    # 곡률: gentle/sharp curve일 때만 표시 (straight는 생략)
    if "gentle" in seg.curvature_label:
        direction = "left" if "left" in seg.curvature_label else "right"
        curve_str = f", gentle {direction} curve"
    elif "sharp" in seg.curvature_label:
        direction = "left" if "left" in seg.curvature_label else "right"
        curve_str = f", sharp {direction} curve"
    else:
        curve_str = ""

    return f"{t_str}: {sp_str}{grad_str}{curve_str}"


def _detect_junction_merge(
    turn_events: list[TurnEvent],
    speed_start_kph: float,
    speed_end_kph: float,
    clip_duration_s: float,
) -> str | None:
    """
    Detect perpendicular junction entry + road merge pattern.

    Pattern: two consecutive turns in opposite directions concentrated in the
    first 40% of the clip, with low initial speed and significant acceleration.
    This distinguishes "junction entry" from "winding road" (same net/total stats).
    """
    if len(turn_events) < 2:
        return None

    t1, t2 = turn_events[0], turn_events[1]

    # Must be opposite directions
    if t1.direction == t2.direction:
        return None

    # Both turns must occur in the first 40% of the clip
    early_cutoff = clip_duration_s * 0.45
    if t1.t_start > early_cutoff or t2.t_start > early_cutoff:
        return None

    # Second turn must follow first turn within 4 seconds (no long straight between)
    if t2.t_start - t1.t_end > 4.0:
        return None

    # Starting speed must be low (just entering from a side road)
    if speed_start_kph > 30.0:
        return None

    # Must accelerate significantly (merging onto a faster road)
    if speed_end_kph - speed_start_kph < 20.0:
        return None

    return (
        f"junction entry ({t1.direction}→{t2.direction} turn) — "
        f"perpendicular access road merging onto main road"
    )


def _infer_road_hint(
    speed_mean_kph: float,
    stop_events: list,
    turn_events: list,
    clip_maneuver: str | None = None,
) -> str | None:
    """
    키네마틱 데이터만으로 도로 유형을 추론한 "힌트" 문자열 반환.
    확신도가 높은 케이스만 반환하고, 모호한 케이스는 None 반환.

    이 값은 VLM 프롬프트에 참고용(hint)으로만 주입되며,
    sensor_confirmed ground truth가 아님.
    """
    has_stops  = len(stop_events) > 0
    has_sharp  = any(ev.radius_m < _HINT_SHARP_TURN_M for ev in turn_events)
    stop_count = len(stop_events)

    # junction entry: 교차로 수직 진입 → 고속 합류 (speed 통계가 중간값이라도 명확)
    if clip_maneuver and "junction entry" in clip_maneuver:
        return (
            "junction entry — ego crossed from a side/access road onto the main road; "
            "road structure and direction change dramatically in the first few frames"
        )

    # 주차 / 저속 기동: 평균 속도 < 10 km/h → 거의 확실
    if speed_mean_kph < _HINT_PARKING_KPH:
        return f"parking or low-speed maneuvering (avg {speed_mean_kph:.0f} km/h)"

    # 고속도로: 고속 + 정지 없음 → 높은 확신
    if speed_mean_kph > _HINT_HIGHWAY_KPH and not has_stops:
        return f"highway (avg {speed_mean_kph:.0f} km/h, no stops)"

    # 정지 구간: 행동 사실만 기술하고 원인(신호/표지판/장애물 등)은 VLM이 영상에서 판별하도록 함.
    # "stop-controlled section" 표현은 VLM이 STOP 표지판을 환각하는 원인이 되므로 사용 금지.
    if has_stops:
        if has_sharp:
            return (
                f"intersection area: {stop_count} stop event(s) and sharp turn detected — "
                f"look at the VIDEO to determine what caused each stop "
                f"(traffic light, STOP sign, vehicle/pedestrian blocking, person exiting vehicle ahead, etc.)"
            )
        return (
            f"intersection area: {stop_count} stop event(s) detected — "
            f"look at the VIDEO to determine what caused each stop "
            f"(traffic light, STOP sign, vehicle/pedestrian blocking, etc.)"
        )

    # 나머지(국도~고속도로 경계 50~100 km/h, 정지 없음): 불확실 → 힌트 없음
    return None


def _accel_severity(kind: str, rate_ms2: float) -> str:
    abs_r = abs(rate_ms2)
    if kind == "deceleration":
        if abs_r >= _HARD_ACCEL_MS2:    return "hard braking"
        if abs_r >= _MODERATE_ACCEL_MS2: return "braking"
        return "gentle deceleration"
    else:
        if abs_r >= _HARD_ACCEL_MS2:    return "hard acceleration"
        if abs_r >= _MODERATE_ACCEL_MS2: return "acceleration"
        return "gentle acceleration"


def _detect_sustained_drift(
    t: np.ndarray,
    yaw_uw: np.ndarray,
    clip_duration_s: float,
    min_drift_deg: float = 3.0,
    min_duration_s: float = 5.0,
) -> str | None:
    """
    turn_events 임계값(r<200m) 미만이지만 시각적으로 인지 가능한
    지속적인 heading 드리프트를 감지한다.
    클립 후반부(40% 이후)만 검사하여 junction entry 오탐 방지.
    """
    second_half_start = clip_duration_s * 0.40
    mask = t >= second_half_start
    if not np.any(mask):
        return None

    t_h   = t[mask]
    yaw_h = yaw_uw[mask]
    if len(t_h) < int(min_duration_s * 8):  # 최소 샘플 수 (10Hz 기준)
        return None

    total_drift_deg = float(np.degrees(yaw_h[-1] - yaw_h[0]))
    duration = float(t_h[-1] - t_h[0])

    # 너무 작으면 노이즈, 너무 크면 이미 turn_event로 처리됐어야 함
    if duration < min_duration_s or abs(total_drift_deg) < min_drift_deg or abs(total_drift_deg) > 20.0:
        return None

    # 일관성 검사: 4분위 모두 같은 방향으로 드리프트
    n = len(yaw_h)
    quarters = [
        float(np.degrees(yaw_h[n // 4]     - yaw_h[0])),
        float(np.degrees(yaw_h[n // 2]     - yaw_h[n // 4])),
        float(np.degrees(yaw_h[3 * n // 4] - yaw_h[n // 2])),
        float(np.degrees(yaw_h[-1]          - yaw_h[3 * n // 4])),
    ]
    sign = 1 if total_drift_deg > 0 else -1
    consistent = sum(1 for q in quarters if q * sign > 0)
    if consistent < 3:
        return None

    direction = "left" if total_drift_deg > 0 else "right"
    return (
        f"sustained gentle {direction} drift "
        f"{second_half_start:.0f}–{t_h[-1]:.0f}s "
        f"({total_drift_deg:+.1f}° total, visually perceivable {direction} curve)"
    )


def _build_narrative(
    segments: list[Segment],
    turn_events: list[TurnEvent],
    lane_change_events: list[LaneChangeEvent],
    accel_events: list[AccelEvent],
    stop_events: list[StopEvent],
    clip_duration_s: float,
    speed_start_kph: float,
    speed_end_kph: float,
    total_distance_m: float,
    clip_maneuver: str | None,
    road_hint: str | None,
    drift_note: str | None = None,
) -> str:
    maneuver_str = f" | {clip_maneuver}" if clip_maneuver else ""
    header = (
        f"[Vehicle kinematics — {clip_duration_s:.0f}s"
        f" | {speed_start_kph:.0f}→{speed_end_kph:.0f} km/h"
        f" | {total_distance_m:.0f}m"
        f"{maneuver_str}]"
    )
    lines = [header]
    for seg in segments:
        lines.append(_format_segment_line(seg))

    # 모든 이벤트를 시간순으로 통합
    all_events: list[tuple[float, str]] = []
    for ev in turn_events:
        all_events.append((
            ev.t_start,
            f"{ev.direction} turn {ev.t_start:.0f}–{ev.t_end:.0f}s (r≈{ev.radius_m:.0f}m)",
        ))
    for ev in lane_change_events:
        all_events.append((
            ev.t_start,
            f"lane change {ev.direction} {ev.t_start:.0f}–{ev.t_end:.0f}s ({ev.disp_m:.1f}m)",
        ))
    for ev in accel_events:
        duration = max(ev.t_end - ev.t_start, 0.1)
        rate_ms2 = (ev.delta_kph / 3.6) / duration
        sign     = -1 if ev.kind == "deceleration" else 1
        label    = _accel_severity(ev.kind, rate_ms2)
        all_events.append((
            ev.t_start,
            f"{label} {ev.delta_kph:.0f} km/h {ev.t_start:.0f}–{ev.t_end:.0f}s ({sign * rate_ms2:.1f} m/s²)",
        ))
    for ev in stop_events:
        all_events.append((
            ev.t_start,
            f"stopped {ev.t_start:.0f}–{ev.t_end:.0f}s ({ev.duration_s:.0f}s)",
        ))

    all_events.sort(key=lambda x: x[0])
    if all_events:
        lines.append("Key: " + " · ".join(text for _, text in all_events))

    # 정지 이벤트가 없고 전 구간 연속 주행한 경우 명시적으로 기록
    # → VLM이 시각적 패턴으로 "정지" 환각을 일으키는 것을 방지
    if not stop_events and speed_start_kph > _STOP_SPEED_KPH and speed_end_kph > _STOP_SPEED_KPH:
        lines.append("Motion: continuous — no stop detected throughout clip")

    if drift_note:
        lines.append(f"Drift: {drift_note}")

    if road_hint:
        lines.append(f"Kinematic road hint: {road_hint}")

    return "\n".join(lines)


# ── 메인 진입점 ───────────────────────────────────────────────────────────────

def compute_motion_summary(
    parquet_path: str | Path,
    timestamp_hour: int | None = None,
) -> MotionSummary:
    """
    egomotion.offline.parquet → MotionSummary.

    Args:
        parquet_path:   {clip_id}.egomotion.offline.parquet 경로
        timestamp_hour: data_collection.parquet 의 hour_of_day (외부에서 주입)
    """
    import pandas as pd

    path = Path(parquet_path)
    if not path.exists():
        log.debug("egomotion.offline not found: %s", path)
        return MotionSummary()

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("Failed to read egomotion.offline %s: %s", path.name, exc)
        return MotionSummary()

    required = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not required.issubset(df.columns):
        log.warning("egomotion.offline missing columns: %s", required - set(df.columns))
        return MotionSummary()

    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < _MIN_ROWS:
        log.warning("egomotion.offline too short: %d rows", len(df))
        return MotionSummary()

    # ── 기본 배열 추출 ─────────────────────────────────────────────────────────
    t_us  = df["timestamp"].to_numpy(dtype=float)
    t     = (t_us - t_us[0]) / 1e6   # seconds from clip start
    x_raw = df["x"].to_numpy(dtype=float)
    y_raw = df["y"].to_numpy(dtype=float)
    z_raw = df["z"].to_numpy(dtype=float)
    qx_a  = df["qx"].to_numpy(dtype=float)
    qy_a  = df["qy"].to_numpy(dtype=float)
    qz_a  = df["qz"].to_numpy(dtype=float)
    qw_a  = df["qw"].to_numpy(dtype=float)

    clip_duration_s = float(t[-1])

    # ── 스무딩 ────────────────────────────────────────────────────────────────
    x = _smooth(x_raw)
    y = _smooth(y_raw)
    z = _smooth(z_raw)

    # ── 시간 간격 ─────────────────────────────────────────────────────────────
    dt = np.diff(t)

    # ── 속도: 호 길이 스칼라 방식 (방향 전환에 의한 상쇄 없음) ─────────────────
    # raw position으로 step 거리 계산 → 스무딩 → 속도
    dt_safe   = np.where(dt > 1e-6, dt, 1e-6)
    step_dist = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2 + np.diff(z_raw)**2)
    speed_raw_ms = step_dist / dt_safe
    # 양 끝 패딩 후 스무딩
    speed_ms  = _smooth(np.concatenate([[speed_raw_ms[0]], speed_raw_ms]))
    speed_kph = speed_ms * 3.6

    # ── 곡률용 방향성 속도 (2D, 스무딩된 위치에서) ───────────────────────────
    vx = _finite_diff(x, np.append(dt, dt[-1]))
    vy = _finite_diff(y, np.append(dt, dt[-1]))

    # ── 가속도 (2차 미분, 곡률 계산용) ───────────────────────────────────────
    ax = _finite_diff(vx, np.append(dt, dt[-1]))
    ay = _finite_diff(vy, np.append(dt, dt[-1]))

    # ── 2D 곡률 ───────────────────────────────────────────────────────────────
    denom = (vx**2 + vy**2) ** 1.5
    denom = np.where(denom < 1e-9, 1e-9, denom)
    kappa = (vx * ay - vy * ax) / denom
    # 정지/저속 구간 곡률 무효화 (< 3 km/h)
    kappa = np.where(speed_kph < 3.0, 0.0, kappa)
    # 이상치 클리핑 (|κ| max = 1/10m)
    kappa = np.clip(kappa, -0.1, 0.1)

    # ── 쿼터니언 → pitch, yaw ─────────────────────────────────────────────────
    pitch_rad, yaw_rad = _quat_to_pitch_yaw(qx_a, qy_a, qz_a, qw_a)
    pitch_deg = np.degrees(pitch_rad)
    heading_net_deg, heading_total_deg = _compute_heading_profile(yaw_rad)
    clip_maneuver = _classify_clip_maneuver(heading_net_deg, heading_total_deg)

    # ── 전체 통계 ─────────────────────────────────────────────────────────────
    speed_mean_kph   = float(np.mean(speed_kph))
    speed_std_kph    = float(np.std(speed_kph))
    pitch_mean_deg   = float(np.mean(pitch_deg))
    total_distance_m = round(float(step_dist.sum()), 1)
    abs_kappa_mean = float(np.mean(np.abs(kappa[speed_kph >= 3.0]))) if (speed_kph >= 3.0).any() else 1e-6
    radius_m       = round(1.0 / max(abs_kappa_mean, 1e-6), 1)
    curvature_dist = _curvature_distribution(kappa)

    # ── 이벤트 감지 ───────────────────────────────────────────────────────────
    turn_events        = _detect_turns(t, kappa)
    lane_change_events = _detect_lane_changes(t, x, y, yaw_rad, kappa)
    # 가감속 감지용: 호 길이 속도를 추가 스무딩 (고주파 노이즈 제거)
    speed_kph_smooth   = _smooth(speed_kph, window=_SMOOTH_WIN * 2)
    accel_events       = _detect_accel_events(t, speed_kph_smooth, kappa)
    stop_events        = _detect_stops(t, speed_kph_smooth)

    # ── 세그먼트 ──────────────────────────────────────────────────────────────
    segments = _build_segments(t, speed_kph, kappa, pitch_deg)

    # ── 클립 시작/종료 속도 (초기화 행 제외: index 1부터) ────────────────────
    speed_start_kph = float(np.mean(speed_kph[1:min(6, len(speed_kph))]))
    speed_end_kph   = float(np.mean(speed_kph[-5:]))

    # ── Junction entry 감지 → clip_maneuver 오버라이드 ──────────────────────
    # "winding road"와 동일한 net/total 통계를 갖지만, 좌→우(혹은 우→좌) 연속 회전이
    # 클립 초반에 집중되고 속도가 크게 증가하면 junction entry로 재분류.
    junction = _detect_junction_merge(
        turn_events, speed_start_kph, speed_end_kph, clip_duration_s
    )
    if junction:
        clip_maneuver = junction

    # ── 도로 유형 힌트 ───────────────────────────────────────────────────────
    road_hint = _infer_road_hint(speed_mean_kph, stop_events, turn_events, clip_maneuver)

    # ── Sustained drift 감지 (r>500m 완만한 커브, "straight" 분류되지만 시각 인지 가능) ──
    # U턴/교차로 기동 클립은 대각도 회전이 이미 narrative에 있으므로 억제
    _no_drift_maneuvers = ("U-turn", "roundabout", "intersection turn")
    _skip_drift = clip_maneuver and any(m in clip_maneuver for m in _no_drift_maneuvers)
    drift_note = None if _skip_drift else _detect_sustained_drift(
        t, np.unwrap(yaw_rad), clip_duration_s, min_drift_deg=4.5
    )

    # ── 서사 ──────────────────────────────────────────────────────────────────
    narrative = _build_narrative(
        segments, turn_events, lane_change_events, accel_events, stop_events,
        clip_duration_s, speed_start_kph, speed_end_kph, total_distance_m,
        clip_maneuver, road_hint, drift_note=drift_note,
    )

    log.info(
        "Stage 0 [%s]: %.0fs, %.1f km/h, r=%.0fm, turns=%d, lc=%d, stops=%d%s",
        path.stem[:8], clip_duration_s, speed_mean_kph, radius_m,
        len(turn_events), len(lane_change_events), len(stop_events),
        f", {clip_maneuver}" if clip_maneuver else "",
    )

    return MotionSummary(
        curvature_radius_m  = radius_m,
        imu_pitch_mean_deg  = round(pitch_mean_deg, 4),
        speed_mean_kph      = round(speed_mean_kph, 2),
        timestamp_hour      = timestamp_hour,
        speed_std_kph       = round(speed_std_kph, 2),
        total_distance_m    = total_distance_m,
        heading_net_deg     = heading_net_deg,
        heading_total_deg   = heading_total_deg,
        clip_maneuver       = clip_maneuver,
        road_hint           = road_hint,
        curvature_dist      = curvature_dist,
        turn_events         = turn_events,
        lane_change_events  = lane_change_events,
        accel_events        = accel_events,
        stop_events         = stop_events,
        segments            = segments,
        narrative           = narrative,
        valid               = True,
    )


def load_motion_summary(
    clip_id: str,
    egomotion_dir: Path,
    timestamp_hour: int | None = None,
) -> MotionSummary:
    """clip_id로 egomotion.offline parquet를 찾아 MotionSummary 반환."""
    parquet_path = egomotion_dir / f"{clip_id}.egomotion.offline.parquet"
    return compute_motion_summary(parquet_path, timestamp_hour=timestamp_hour)