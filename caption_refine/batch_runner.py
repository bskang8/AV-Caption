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
배치 처리 + 진행 추적.

단일 서버 사용법:
  uv run python -m caption_refine.batch_runner --source gap
  uv run python -m caption_refine.batch_runner --source longtail
  uv run python -m caption_refine.batch_runner --ids-file my_clips.json
  uv run python -m caption_refine.batch_runner --source all --limit 1000
  uv run python -m caption_refine.batch_runner --source gap --concurrent 4

분산 실행 (GPU 10장, 10개 프로세스):
  # GPU 0 담당 (shard 0/10)
  CR_VLLM_URL=http://localhost:8000/v1 \\
  uv run python -m caption_refine.batch_runner \\
      --source all --shard-index 0 --total-shards 10 --concurrent 4

  # GPU 1 담당 (shard 1/10) — 별도 터미널
  CR_VLLM_URL=http://localhost:8001/v1 \\
  uv run python -m caption_refine.batch_runner \\
      --source all --shard-index 1 --total-shards 10 --concurrent 4

  # 또는 run_distributed.sh 스크립트로 한번에 실행
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from caption_refine.config import (
    DEFAULT_CONCURRENT,
    INDEX_DIR,
    OUTPUT_ROOT,
    PROGRESS_FILE,
    SANFLOW_GAP_PATH,
    VLLM_BASE_URL,
)
from caption_refine.cosmos_client import CosmosClient
from caption_refine.pipeline import ClipResult, process_clip

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
        longtail_path = INDEX_DIR / "longtail_clips.json"
        ids = json.loads(longtail_path.read_text())
    elif source == "all":
        clip_ids_path = INDEX_DIR / "clip_ids.json"
        ids = json.loads(clip_ids_path.read_text())
    else:
        raise ValueError(f"Unknown source: {source}")

    if limit:
        ids = ids[:limit]
    return ids


def _shard_clip_ids(ids: list[str], shard_index: int, total_shards: int) -> list[str]:
    """전체 목록에서 이 워커가 담당할 슬라이스를 반환 (스트라이드 방식)."""
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

    sem = asyncio.Semaphore(max_concurrent)
    client = CosmosClient(base_url=vllm_url)

    ok_count = err_count = 0
    t0 = time.monotonic()

    async def _process(clip_id: str) -> ClipResult:
        async with sem:
            return await process_clip(clip_id, client)

    tasks = [asyncio.create_task(_process(cid)) for cid in pending]

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result: ClipResult = await coro

        if result.status == "ok":
            state["done"].append(result.clip_id)
            ok_count += 1
        else:
            state["error"].append(result.clip_id)
            err_count += 1
            log.warning("FAILED [%s]: %s / %s", result.clip_id[:8], result.status, result.error)

        if i % 10 == 0:
            _save_progress(state, progress_file)
            elapsed = time.monotonic() - t0
            rate = i / elapsed
            eta = (len(pending) - i) / rate if rate > 0 else 0
            log.info("Progress: %d/%d  ok=%d err=%d  rate=%.1f/min  ETA=%.0fmin",
                     i, len(pending), ok_count, err_count, rate * 60, eta / 60)

    _save_progress(state, progress_file)
    await client.aclose()

    elapsed = time.monotonic() - t0
    log.info("─" * 60)
    log.info("Complete: %d ok / %d error / %.1f min", ok_count, err_count, elapsed / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption refinement batch runner")
    parser.add_argument(
        "--source",
        choices=["gap", "longtail", "all"],
        default="gap",
        help="clip_id 소스 (default: gap)",
    )
    parser.add_argument("--ids-file", type=str, default=None,
                        help="clip_id 목록 JSON 파일 경로 (--source 무시)")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 최대 클립 수")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT,
                        help=f"동시 처리 클립 수 (default: {DEFAULT_CONCURRENT})")
    parser.add_argument("--reset", action="store_true",
                        help="진행 상태 초기화 후 처음부터 재시작")

    # 분산 실행 옵션
    parser.add_argument("--shard-index", type=int, default=0,
                        help="이 워커가 담당할 샤드 번호 (0-based, default: 0)")
    parser.add_argument("--total-shards", type=int, default=1,
                        help="전체 샤드(워커) 수 (default: 1 = 분산 없음)")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="vLLM 서버 URL (기본값: CR_VLLM_URL 환경변수 또는 http://localhost:8000/v1)")

    args = parser.parse_args()

    if args.shard_index >= args.total_shards:
        parser.error(f"--shard-index ({args.shard_index}) must be < --total-shards ({args.total_shards})")

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