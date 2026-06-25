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
vLLM 서버(OpenAI 호환 API)를 통한 cosmos-reason2 비동기 클라이언트.

v2 추가사항:
  text_chat() — 영상 없이 텍스트만으로 LLM 호출 (Stage 4 cross-validation용).
"""
import asyncio
import base64
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from openai import AsyncOpenAI, APIError, APITimeoutError

from caption_refine_v2.config import (
    FRAME_QUALITY,
    MAX_FRAME_H,
    MAX_FRAME_W,
    MAX_RETRIES,
    MAX_TOKENS_STAGE1,
    NUM_FRAMES,
    REQUEST_TIMEOUT,
    RETRY_DELAY_BASE,
    VIDEO_FPS,
    VIDEO_INPUT_MODE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)

SYSTEM_PROMPT = (
    "You are an expert autonomous driving video analyst. "
    "You reason carefully about what you observe in the video before answering. "
    "Always ground your answers in visual evidence from the video."
)

log = logging.getLogger(__name__)


# ── 프레임 유틸 ───────────────────────────────────────────────────────────────

def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > MAX_FRAME_W or h > MAX_FRAME_H:
        scale = min(MAX_FRAME_W / w, MAX_FRAME_H / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def sample_frames(video_path: str | Path, n: int = NUM_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"No frames found in {video_path}")

    indices = np.linspace(0, max(total - 1, 0), min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(_resize_frame(frame))
    cap.release()

    if not frames:
        raise ValueError(f"Failed to decode any frame from {video_path}")
    return frames


def _frame_to_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode()


# ── content 블록 빌더 ─────────────────────────────────────────────────────────

def _build_content(
    video_path: str | Path,
    prompt: str,
    n_frames: int | None = None,
    force_frames: bool = False,
) -> list[dict]:
    if VIDEO_INPUT_MODE == "video" and not force_frames:
        file_url = f"file://{Path(video_path).resolve()}"
        return [
            {"type": "video_url", "video_url": {"url": file_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        frames = sample_frames(video_path, n=n_frames or NUM_FRAMES)
        content: list[dict] = []
        for frame in frames:
            b64 = _frame_to_b64(frame)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        return content


# ── JSON 파싱 헬퍼 ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | list:
    from json_repair import repair_json

    text = text.strip()
    md = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if md:
        text = md.group(1).strip()

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        candidate = text[start:end + 1] if end > start else text[start:]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, (dict, list)):
                return repaired
        except Exception:
            pass

    raise ValueError(f"No valid JSON found in response: {text[:200]}")


# ── 메인 클라이언트 ───────────────────────────────────────────────────────────

class CosmosClient:
    def __init__(self, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=REQUEST_TIMEOUT,
        )

    async def _chat(
        self,
        video_path: str | Path,
        prompt: str,
        max_tokens: int,
        enable_thinking: bool = False,
        n_frames: int | None = None,
        force_frames: bool = False,
        temperature: float = 0.0,
    ) -> str:
        content = await asyncio.get_event_loop().run_in_executor(
            None, _build_content, video_path, prompt, n_frames, force_frames
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        extra_body: dict = {}
        if VIDEO_INPUT_MODE == "video" and not force_frames:
            extra_body["mm_processor_kwargs"] = {
                "fps": VIDEO_FPS,
                "do_sample_frames": True,
            }
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except (APIError, APITimeoutError) as exc:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_DELAY_BASE ** attempt
                log.warning("API error (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, MAX_RETRIES, exc, delay)
                await asyncio.sleep(delay)
        return ""

    async def chat_json(
        self,
        video_path: str | Path,
        prompt: str,
        max_tokens: int = MAX_TOKENS_STAGE1,
        enable_thinking: bool = False,
        n_frames: int | None = None,
        force_frames: bool = False,
        temperature: float = 0.0,
    ) -> dict | list:
        for attempt in range(1, MAX_RETRIES + 1):
            raw = await self._chat(video_path, prompt, max_tokens,
                                   enable_thinking=enable_thinking, n_frames=n_frames,
                                   force_frames=force_frames, temperature=temperature)
            try:
                return _extract_json(raw)
            except ValueError as exc:
                if attempt == MAX_RETRIES:
                    log.error("JSON parse failed after %d attempts: %s\nraw=%s",
                              MAX_RETRIES, exc, raw[:300])
                    raise
                log.warning("JSON parse failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                await asyncio.sleep(RETRY_DELAY_BASE)

    async def chat_text(
        self,
        video_path: str | Path,
        prompt: str,
        max_tokens: int,
        enable_thinking: bool = False,
        n_frames: int | None = None,
        force_frames: bool = False,
        temperature: float = 0.0,
    ) -> str:
        return await self._chat(video_path, prompt, max_tokens,
                                enable_thinking=enable_thinking, n_frames=n_frames,
                                force_frames=force_frames, temperature=temperature)

    async def text_chat(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """영상 없이 텍스트만으로 LLM 호출. Stage 4 cross-validation 전용."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except (APIError, APITimeoutError) as exc:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_DELAY_BASE ** attempt
                log.warning("text_chat error (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, MAX_RETRIES, exc, delay)
                await asyncio.sleep(delay)
        return ""

    async def text_chat_json(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict | list:
        """텍스트 전용 JSON 응답. Stage 4 전용."""
        for attempt in range(1, MAX_RETRIES + 1):
            raw = await self.text_chat(prompt, max_tokens, temperature=temperature)
            try:
                return _extract_json(raw)
            except ValueError as exc:
                if attempt == MAX_RETRIES:
                    log.error("text_chat JSON parse failed: %s\nraw=%s", exc, raw[:300])
                    raise
                log.warning("text_chat JSON parse failed (attempt %d/%d)", attempt, MAX_RETRIES)
                await asyncio.sleep(RETRY_DELAY_BASE)

    async def aclose(self) -> None:
        await self._client.close()