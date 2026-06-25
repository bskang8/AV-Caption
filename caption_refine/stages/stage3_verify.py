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
Stage 3 — Self-Verification

Stage 2에서 confidence < threshold인 필드만 영상 재확인.
모델이 CONFIRM / CORRECT 판정을 내리면 Stage 2 결과에 반영해 최종 ODD 생성.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

from caption_refine.config import MAX_TOKENS_STAGE3
from caption_refine.cosmos_client import CosmosClient
from caption_refine.prompts import stage3_verify
from caption_refine.stages.stage2_extract import ExtractionResult

log = logging.getLogger(__name__)


async def run(
    client: CosmosClient,
    video_path: str | Path,
    extraction: ExtractionResult,
) -> dict:
    """
    Stage 2 fields를 복사해 low-confidence 항목을 재검증.
    최종 검증된 fields dict 반환.
    """
    verified = copy.deepcopy(extraction.fields)

    if not extraction.low_confidence:
        log.debug("Stage 3: no low-confidence fields, skipping.")
        return verified

    prompt = stage3_verify(extraction.low_confidence)
    try:
        data = await client.chat_json(video_path, prompt, max_tokens=MAX_TOKENS_STAGE3)
    except Exception as exc:
        log.error("Stage 3 failed: %s — keeping Stage 2 values", exc)
        return verified

    if not isinstance(data, dict):
        log.warning("Stage 3: unexpected response type %s", type(data))
        return verified

    for field_key, verdict_info in data.items():
        if field_key not in verified:
            continue
        if not isinstance(verdict_info, dict):
            continue

        verdict = verdict_info.get("verdict", "").upper()
        if verdict == "CORRECT":
            corrected = verdict_info.get("corrected_value")
            if corrected is not None:
                # 값 교체 + confidence를 재검증 완료 표시로 업데이트
                field = verified[field_key]
                if isinstance(field, dict):
                    value_key = "present" if "present" in field else "value"
                    if value_key not in field and "types" in field:
                        value_key = "types"
                    field[value_key] = corrected
                    field["confidence"] = 0.85  # 재검증 후 상향 조정
                    field["verified"] = True
                log.info("Stage 3 CORRECT: %s → %s", field_key, corrected)
        elif verdict == "CONFIRM":
            if isinstance(verified.get(field_key), dict):
                verified[field_key]["verified"] = True
            log.debug("Stage 3 CONFIRM: %s", field_key)

    return verified