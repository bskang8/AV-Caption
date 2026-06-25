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
Stage 4 — 교차 검증 (Phase 3)

두 단계:
  Step 1: VLM이 NL 시나리오에서 implied ODD 키워드 추출 (텍스트 전용, 영상 없음)
  Step 2: 규칙 기반 충돌 감지 + 해소 + consistency_score 계산

충돌 해소 원칙 (odd_tagging_practical.md):
  NL 우선: precipitation, fog, road_surface   (맥락 추론에 강함)
  ODD 우선: road_type, traffic_density, lighting (시각 분류에 강함)
  flag:    나머지 → review_needed

odd_final: 충돌 해소가 반영된 최종 ODD dict.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from caption_refine_v2.config import (
    MAX_TOKENS_STAGE4,
    CONSISTENCY_GOOD,
    CONSISTENCY_WARN,
)
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.prompts import stage4_nl_to_odd

log = logging.getLogger(__name__)

# 충돌 감지 대상 속성 → (ODD group, ODD field, NL implied key)
# road_type 제외: HD맵 기반 도로 명칭(ODD)과 시각적 외관(NL)이 다른 것은 설계상 정상.
# ODD가 항상 우선하므로 최종 캡션에 영향 없음 → 충돌 감지 불필요.
_CONFLICT_CHECKS: list[tuple[str, str, str, str]] = [
    ("precipitation",   "environment",      "precipitation",       "implied_precipitation"),
    ("fog",             "environment",      "fog",                 "implied_fog"),
    ("road_surface",    "environment",      "road_surface",        "implied_road_surface"),
    ("lighting",        "environment",      "lighting_condition",  "implied_lighting"),
    ("traffic_density", "dynamic_elements", "traffic_density",     "implied_traffic_density"),
    ("special_event",   "dynamic_elements", "special_event",       "implied_special_event"),
]

# 파이프라인 우선순위
_NL_PREFERRED  = frozenset({"precipitation", "fog", "road_surface"})
_ODD_PREFERRED = frozenset({"road_type", "traffic_density", "lighting"})


@dataclass
class CrossValResult:
    nl_implied:        dict       = field(default_factory=dict)
    conflicts:         list[dict] = field(default_factory=list)
    consistency_score: float      = 1.0
    odd_final:         dict       = field(default_factory=dict)  # 충돌 해소 반영 ODD

    def quality_flag(self) -> str:
        if self.consistency_score >= 0.9:
            return "good"
        if self.consistency_score >= 0.7:
            return "warn"
        return "poor"


def _detect_and_resolve(odd_grouped: dict, nl_implied: dict) -> tuple[list[dict], dict, float]:
    """
    충돌 감지 → 해소 → odd_final 생성 → consistency_score 계산.
    반환: (conflicts, odd_final, consistency_score)
    """
    odd_final  = copy.deepcopy(odd_grouped)
    conflicts: list[dict] = []
    comparable = 0

    for attr, group, odd_field, nl_key in _CONFLICT_CHECKS:
        odd_val = odd_grouped.get(group, {}).get(odd_field, "unknown")
        nl_val  = nl_implied.get(nl_key)

        # 비교 불가 케이스 제외
        if nl_val is None or odd_val == "unknown":
            continue

        # 특수 케이스: ODD가 강수 없음(none)인데 NL이 snow/rain을 추론한 경우 — 충돌로 처리하지 않음
        # NL 2B 모델이 젖은 노면이나 도로변 잔설을 보고 강수로 과잉 추론하는 패턴을 억제.
        # ODD(Stage 2)는 영상에서 직접 강수를 판단하므로 none이 맞을 가능성이 높음.
        if attr == "precipitation" and odd_val == "none" and nl_val in ("snow", "rain"):
            log.debug("Stage 4: skipping precipitation-vs-none conflict (NL likely inferred from wet road/roadside snow, not active precipitation)")
            continue

        comparable += 1

        if odd_val == nl_val:
            continue  # 일치 — 충돌 없음

        # 불일치 → 해소 규칙 적용
        if attr in _NL_PREFERRED:
            winner, final_value = "nl",   nl_val
            reason = "NL이 장면 맥락에서 추론하는 속성"
        elif attr in _ODD_PREFERRED:
            winner, final_value = "odd",  odd_val
            reason = "ODD가 시각적 분류로 직접 판단하는 속성"
        else:
            winner, final_value = "flag", "review_needed"
            reason = "규칙 미정의, 수동 검토 필요"

        conflicts.append({
            "attribute": attr,
            "odd_value": odd_val,
            "nl_implied": nl_val,
            "resolution": {
                "winner":      winner,
                "final_value": final_value,
                "reason":      reason,
            },
        })

        # odd_final에 반영
        if winner != "flag" and group in odd_final and odd_field in odd_final[group]:
            odd_final[group][odd_field] = final_value

    score = 1.0 - (len(conflicts) / comparable) if comparable > 0 else 1.0
    return conflicts, odd_final, score


async def run(
    client: CosmosClient,
    nl_scenario: str,
    odd_grouped: dict,
) -> CrossValResult:
    """
    Step 1: NL 시나리오에서 implied ODD 키워드 추출 (VLM, 텍스트 전용)
    Step 2: 규칙 기반 충돌 감지 + 해소 + consistency_score
    """
    if not nl_scenario:
        log.warning("Stage 4: empty NL scenario — skipping cross-validation.")
        return CrossValResult(odd_final=copy.deepcopy(odd_grouped))
    if not odd_grouped:
        log.warning("Stage 4: empty ODD — skipping cross-validation.")
        return CrossValResult()

    # ── Step 1: NL → implied ODD (VLM, 텍스트 전용) ──────────────────────────
    prompt = stage4_nl_to_odd(nl_scenario)
    try:
        nl_implied = await client.text_chat_json(prompt, max_tokens=MAX_TOKENS_STAGE4)
        if not isinstance(nl_implied, dict):
            nl_implied = {}
    except Exception as exc:
        log.error("Stage 4 Step 1 failed: %s — using empty implied ODD", exc)
        nl_implied = {}

    # ── Step 2: 규칙 기반 충돌 감지 + 해소 ──────────────────────────────────
    conflicts, odd_final, score = _detect_and_resolve(odd_grouped, nl_implied)

    result = CrossValResult(
        nl_implied=nl_implied,
        conflicts=conflicts,
        consistency_score=round(score, 3),
        odd_final=odd_final,
    )

    flag = result.quality_flag()
    log.info(
        "Stage 4 done — conflicts=%d  consistency=%.2f [%s]",
        len(conflicts), score, flag,
    )
    if flag == "poor":
        log.warning("Stage 4: consistency < %.1f — full manual review recommended", CONSISTENCY_WARN)
    elif flag == "warn":
        log.warning("Stage 4: consistency < %.1f — check conflicted attributes", CONSISTENCY_GOOD)

    return result