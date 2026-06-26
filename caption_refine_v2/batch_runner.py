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
배치 처리 + 진행 추적 (v2).

사용법:
  uv run python -m caption_refine_v2.batch_runner --source gap
  uv run python -m caption_refine_v2.batch_runner --source all --limit 100
  uv run python -m caption_refine_v2.batch_runner --ids-file my_clips.json --concurrent 4

분산 실행:
  CR_VLLM_URL=http://localhost:8000/v1 \\
  uv run python -m caption_refine_v2.batch_runner \\
      --source all --shard-index 0 --total-shards 10 --concurrent 4

클립 메타데이터 (Phase 1 센서 확정값):
  CR_META_DIR 환경변수로 디렉토리 지정.
  {clip_id}.meta.json 파일이 있으면 자동으로 Phase 1 적용.
  없으면 VLM만으로 처리.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from caption_refine_v2.config import (
    DEFAULT_CONCURRENT,
    INDEX_DIR,
    OUTPUT_ROOT,
    PROGRESS_FILE,
    SANFLOW_GAP_PATH,
    VIDEO_SUFFIX,
    VIDEOS_DIR,
    VLLM_BASE_URL,
)
from caption_refine_v2.cosmos_client import CosmosClient
from caption_refine_v2.pipeline import ClipMeta, ClipResult, process_clip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 진행 상태 관리 ─────────────────────────────────────────────────────────────

def _get_progress_file(shard_index: int, total_shards: int) -> Path:
    if total_shards > 1:
        return OUTPUT_ROOT / f"progress_shard{shard_index:02d}of{total_shards}.json"
    return PROGRESS_FILE


def _load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"done": [], "error": [], "skipped": []}


def _save_progress(state: dict, progress_file: Path) -> None:
    progress_file.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── clip_id 소스 로더 ─────────────────────────────────────────────────────────

def _load_clip_ids(source: str, ids_file: str | None, limit: int | None) -> list[str]:
    if ids_file:
        ids = json.loads(Path(ids_file).read_text())
    elif source == "gap":
        gaps = json.loads(SANFLOW_GAP_PATH.read_text())
        ids = [g["clip_id"] for g in gaps]
    elif source == "longtail":
        ids = json.loads((INDEX_DIR / "longtail_clips.json").read_text())
    elif source == "all":
        if not VIDEOS_DIR.exists():
            raise FileNotFoundError(f"VIDEOS_DIR 를 찾을 수 없습니다: {VIDEOS_DIR}")
        ids = sorted(
            p.name[: -len(VIDEO_SUFFIX)]
            for p in VIDEOS_DIR.iterdir()
            if p.name.endswith(VIDEO_SUFFIX)
        )
        log.info("VIDEOS_DIR 스캔: %d개 클립 발견 (%s)", len(ids), VIDEOS_DIR)
    else:
        raise ValueError(f"Unknown source: {source}")

    if limit:
        ids = ids[:limit]
    return ids


def _shard_clip_ids(ids: list[str], shard_index: int, total_shards: int) -> list[str]:
    return ids[shard_index::total_shards]


# ── 배치 실행 ─────────────────────────────────────────────────────────────────

async def _run_batch(
    clip_ids: list[str],
    max_concurrent: int,
    state: dict,
    progress_file: Path,
    vllm_url: str,
) -> None:
    already_done = set(state["done"]) | set(state["error"])
    pending = [cid for cid in clip_ids if cid not in already_done]

    log.info("Total: %d  |  Already done: %d  |  Pending: %d",
             len(clip_ids), len(already_done), len(pending))
    log.info("vLLM endpoint: %s", vllm_url)

    if not pending:
        log.info("Nothing to process.")
        return

    sem    = asyncio.Semaphore(max_concurrent)
    client = CosmosClient(base_url=vllm_url)

    ok_count = err_count = 0
    t0 = time.monotonic()

    # consistency_score 분포 추적
    scores: list[float] = []

    async def _process(clip_id: str) -> ClipResult:
        async with sem:
            return await process_clip(clip_id, client)

    tasks = [asyncio.create_task(_process(cid)) for cid in pending]

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result: ClipResult = await coro

        if result.status == "ok":
            state["done"].append(result.clip_id)
            ok_count += 1
            if result.crossval:
                scores.append(result.crossval.get("consistency_score", 1.0))
        else:
            state["error"].append(result.clip_id)
            err_count += 1
            log.warning("FAILED [%s]: %s / %s",
                        result.clip_id[:8], result.status, result.error)

        if i % 10 == 0:
            _save_progress(state, progress_file)
            elapsed = time.monotonic() - t0
            rate    = i / elapsed
            eta     = (len(pending) - i) / rate if rate > 0 else 0
            avg_score = sum(scores) / len(scores) if scores else float("nan")
            log.info(
                "Progress: %d/%d  ok=%d err=%d  rate=%.1f/min  ETA=%.0fmin  avg_consistency=%.2f",
                i, len(pending), ok_count, err_count, rate * 60, eta / 60, avg_score,
            )

    _save_progress(state, progress_file)
    await client.aclose()

    elapsed   = time.monotonic() - t0
    avg_score = sum(scores) / len(scores) if scores else float("nan")
    poor_count = sum(1 for s in scores if s < 0.7)

    log.info("─" * 60)
    log.info("Complete: %d ok / %d error / %.1f min", ok_count, err_count, elapsed / 60)
    log.info("Consistency — avg=%.2f  poor(<0.7)=%d (%.1f%%)",
             avg_score, poor_count, 100 * poor_count / len(scores) if scores else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption refinement v2 batch runner")
    parser.add_argument("--source", choices=["gap", "longtail", "all"], default="gap")
    parser.add_argument("--ids-file",   type=str, default=None)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument("--reset",      action="store_true")
    parser.add_argument("--shard-index",  type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--vllm-url",   type=str, default=None)

    args = parser.parse_args()

    if args.shard_index >= args.total_shards:
        parser.error(
            f"--shard-index ({args.shard_index}) must be < --total-shards ({args.total_shards})"
        )

    vllm_url = args.vllm_url or VLLM_BASE_URL

    clip_ids = _load_clip_ids(args.source, args.ids_file, args.limit)
    log.info("Loaded %d clip IDs from source=%s", len(clip_ids), args.ids_file or args.source)

    if args.total_shards > 1:
        clip_ids = _shard_clip_ids(clip_ids, args.shard_index, args.total_shards)
        log.info("Shard %d/%d → %d clips assigned",
                 args.shard_index, args.total_shards, len(clip_ids))

    progress_file = _get_progress_file(args.shard_index, args.total_shards)
    state = {} if args.reset else _load_progress(progress_file)
    if args.reset:
        state = {"done": [], "error": [], "skipped": []}
        log.info("Progress state reset. (%s)", progress_file.name)

    asyncio.run(_run_batch(clip_ids, args.concurrent, state, progress_file, vllm_url))


if __name__ == "__main__":
    main()