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
Stage 3 — unknown 필드 재확인

Stage 2에서 "unknown"으로 남은 필드만 video를 다시 보고 재판단.
confidence 기반 재검증(v1)에서 unknown 기반 재확인(v2)으로 전환.

반환: 그룹화된 ODD dict (unknown이 해소 가능한 경우 값이 업데이트됨).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

from caption_refine_v2.config import MAX_TOKENS_STAGE3, NUM_FRAMES_STAGE3
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.prompts import stage3_resolve_unknown
from caption_refine_v2.stages.stage2_odd import ExtractionResult

log = logging.getLogger(__name__)


async def run(
    client: CosmosClient,
    video_path: str | Path,
    extraction: ExtractionResult,
    sensor_confirmed: dict | None = None,
) -> dict:
    """
    unknown 필드를 재확인해 해소 가능한 값으로 업데이트.
    반환: grouped ODD dict (road_structure / environment / dynamic_elements / scene_complexity).
    """
    grouped = copy.deepcopy(extraction.to_grouped_dict())

    if not extraction.unknown_fields:
        log.debug("Stage 3: no unknown fields, skipping.")
        return grouped

    log.info("Stage 3: resolving %d unknown fields", len(extraction.unknown_fields))
    prompt = stage3_resolve_unknown(extraction.unknown_fields, sensor_confirmed=sensor_confirmed)

    try:
        data = await client.chat_json(
            video_path,
            prompt,
            max_tokens=MAX_TOKENS_STAGE3,
            force_frames=True,
            n_frames=NUM_FRAMES_STAGE3,
        )
    except Exception as exc:
        log.error("Stage 3 failed: %s — keeping unknown values", exc)
        return grouped

    if not isinstance(data, dict):
        log.warning("Stage 3: unexpected response type %s", type(data))
        return grouped

    resolved = still_unknown = 0
    for dotted_key, new_value in data.items():
        if "." not in dotted_key:
            continue
        group, field_name = dotted_key.split(".", 1)
        if group not in grouped or field_name not in grouped[group]:
            continue
        if not isinstance(new_value, str):
            continue

        if new_value != "unknown":
            # lanes_opposite=0 means "truly one-way road" — a very strong claim.
            # If Stage 2 was already uncertain (unknown), Stage 3 resolving to 0 is
            # almost always a misclassification of "I can't see" as "doesn't exist".
            # Keep as unknown; Stage 5 will omit the field rather than make a false claim.
            if field_name == "lanes_opposite" and new_value == "0":
                log.debug("Stage 3: lanes_opposite→0 suppressed (Stage 2 was unknown; keeping unknown)")
                still_unknown += 1
                continue
            grouped[group][field_name] = new_value
            resolved += 1
            log.debug("Stage 3 resolved: %s → %s", dotted_key, new_value)
        else:
            still_unknown += 1

    log.info("Stage 3 done — resolved=%d still_unknown=%d", resolved, still_unknown)
    return grouped