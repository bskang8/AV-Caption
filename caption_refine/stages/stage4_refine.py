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
Stage 4 — Caption Refinement

Stage 1·3 결과를 바탕으로 정제된 caption 생성.
"""
from __future__ import annotations

import logging
from pathlib import Path

from caption_refine.config import NUM_FRAMES, MAX_TOKENS_STAGE4
from caption_refine.cosmos_client import CosmosClient
from caption_refine.prompts import stage4_refine
from caption_refine.stages.stage1_ground import GroundingResult

log = logging.getLogger(__name__)


async def run(
    client: CosmosClient,
    video_path: str | Path,
    original_caption: str,
    grounding: GroundingResult,
    verified_odd: dict,
) -> str:
    prompt = stage4_refine(
        original_caption=original_caption,
        verified_odd=verified_odd,
        hallucinated=grounding.hallucinated,
        missed=grounding.missed,
    )
    try:
        # 긴 텍스트 프롬프트(ODD JSON + caption + hal/miss) + 이미지 토큰 초과 방지.
        # Stage 4는 텍스트에 충분한 정보가 있으므로 프레임 수를 줄여도 품질 유지.
        n_frames = max(4, NUM_FRAMES // 2)
        caption = await client.chat_text(video_path, prompt, max_tokens=MAX_TOKENS_STAGE4,
                                         n_frames=n_frames)
        return caption.strip()
    except Exception as exc:
        log.error("Stage 4 failed: %s — returning original caption", exc)
        return original_caption