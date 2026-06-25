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
Stage 2 — Structured ODD Extraction

영상에서 기존 ODD 필드 + 확장 필드를 JSON으로 추출.
confidence가 낮은 필드 목록을 함께 반환해 Stage 3 재검증에 사용.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from caption_refine.config import CONFIDENCE_THRESHOLD, MAX_TOKENS_STAGE2, NUM_FRAMES
from caption_refine.cosmos_client import CosmosClient
from caption_refine.prompts import stage2_extract

log = logging.getLogger(__name__)

# Stage 2 응답에서 값을 담고 있는 서브키 (value / present)
_VALUE_KEYS = ("value", "present", "types")


@dataclass
class ExtractionResult:
    fields: dict = field(default_factory=dict)          # 전체 필드 (confidence 포함)
    low_confidence: dict = field(default_factory=dict)  # confidence < threshold 필드만

    def to_dict(self) -> dict:
        return self.fields

    def to_odd_dict(self) -> dict:
        """기존 odd_tags 스키마와 호환되는 단순 dict 반환."""
        mapping = {
            "time_of_day":       ("time_of_day",    "value"),
            "weather":           ("weather",         "value"),
            "road_type":         ("road_type",       "value"),
            "traffic_density":   ("traffic_density", "value"),
            "hazard_level":      ("hazard_level",    "value"),
            "agent_type":        ("surrounding_vehicles", "types"),
            "ego_action":        ("ego_actions",     "value"),
        }
        result = {}
        for out_key, (src_key, sub_key) in mapping.items():
            src = self.fields.get(src_key, {})
            result[out_key] = src.get(sub_key, "unknown") if isinstance(src, dict) else "unknown"
        return result


def _collect_low_confidence(fields: dict, threshold: float) -> dict:
    low = {}
    for key, val in fields.items():
        if not isinstance(val, dict):
            continue
        conf = val.get("confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < threshold:
            low[key] = val
    return low


async def run(
    client: CosmosClient,
    video_path: str | Path,
) -> ExtractionResult:
    prompt = stage2_extract()
    try:
        # ODD 추출 JSON이 길어서 max_seq_len 초과를 막기 위해 프레임 수를 최소화.
        # 12프레임 입력 ~5500 tokens, Qwen3 thinking ~2000+ tokens → 총 초과.
        # 4프레임 입력 ~2200 tokens → 출력 공간 ~6000 tokens (thinking 포함 충분).
        n_frames = max(4, NUM_FRAMES // 3)
        data = await client.chat_json(
            video_path, prompt, max_tokens=MAX_TOKENS_STAGE2, n_frames=n_frames
        )
    except Exception as exc:
        log.error("Stage 2 failed: %s", exc)
        return ExtractionResult()

    if not isinstance(data, dict):
        log.warning("Stage 2: unexpected response type %s", type(data))
        return ExtractionResult()

    low = _collect_low_confidence(data, CONFIDENCE_THRESHOLD)
    return ExtractionResult(fields=data, low_confidence=low)