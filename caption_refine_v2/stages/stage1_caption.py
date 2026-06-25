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
Stage 1 — NL 시나리오 생성 (Phase 2-B)

기존 caption 미참조. video만 보고 4섹션 구조화 자연어 시나리오 생성.

4섹션 구조:
  1. Road & Environment
  2. Surrounding Situation
  3. Key Challenge
  4. Events

Stage 4 교차검증에서 이 텍스트를 LLM이 분석해 implied ODD를 추출하므로,
구조화된 섹션이 자유 서술보다 추출 정확도를 높인다.

2회 병렬 실행 후 내용어 수가 많은 쪽 선택.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from caption_refine_v2.config import MAX_TOKENS_STAGE1, NUM_FRAMES_STAGE1
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.prompts import stage1_nl_scenario

log = logging.getLogger(__name__)

_STAGE1_PASSES      = 2
_STAGE1_TEMPERATURE = 0.2

_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'and', 'or', 'but', 'not',
    'it', 'its', 'this', 'that', 'these', 'those', 'also', 'which', 'while',
    'ego', 'vehicle',
})


def _content_word_count(text: str) -> int:
    words = set(re.findall(r'[a-z]+', text.lower()))
    return len(words - _STOPWORDS)


async def _single_pass(client: CosmosClient, video_path: str | Path, prompt: str) -> str:
    try:
        return await client.chat_text(
            video_path,
            prompt,
            max_tokens=MAX_TOKENS_STAGE1,
            force_frames=True,
            n_frames=NUM_FRAMES_STAGE1,
            temperature=_STAGE1_TEMPERATURE,
        )
    except Exception as exc:
        log.warning("Stage 1 pass failed: %s", exc)
        return ""


async def run(client: CosmosClient, video_path: str | Path, motion_narrative: str = "") -> str:
    """
    4섹션 NL 시나리오를 생성해 반환.
    모든 패스 실패 시 빈 문자열 반환.
    """
    prompt  = stage1_nl_scenario(motion_narrative=motion_narrative)
    results = await asyncio.gather(
        *[_single_pass(client, video_path, prompt) for _ in range(_STAGE1_PASSES)]
    )

    valid = [r.strip() for r in results if r.strip()]
    if not valid:
        log.error("Stage 1: all %d passes failed", _STAGE1_PASSES)
        return ""

    best = max(valid, key=_content_word_count)
    log.info("Stage 1 done — NL scenario (%d content words, %d/%d passes ok)",
             _content_word_count(best), len(valid), _STAGE1_PASSES)
    return best