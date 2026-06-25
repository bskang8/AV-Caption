#!/usr/bin/env python3
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
nvidia/PhysicalAI-Autonomous-Vehicles 에서 egomotion.offline 전체 다운로드.

구조:
  HuggingFace: labels/egomotion.offline/egomotion.offline.chunk_{N:04d}.zip
  zip 내부:    {clip_id}.egomotion.offline.parquet
  출력:        {output_dir}/{clip_id}.egomotion.offline.parquet

사용법:
  # 전체 다운로드 (306,152 클립, 3,146 청크)
  uv run python download_egomotion_offline.py

  # 출력 경로 지정
  uv run python download_egomotion_offline.py --output-dir /Data1/home/bskang/cds-data/egomotion_offline

  # zip 캐시 삭제하여 디스크 절약
  uv run python download_egomotion_offline.py --delete-zips

  # 특정 청크 범위만 (테스트용)
  uv run python download_egomotion_offline.py --chunk-start 0 --chunk-end 10

  # 분산 실행: 0~9번 노드 중 0번 (총 chunks를 10등분해서 처리)
  uv run python download_egomotion_offline.py --shard-index 0 --total-shards 10
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ID      = "nvidia/PhysicalAI-Autonomous-Vehicles"
FEATURE_NAME = "egomotion.offline"
TOTAL_CHUNKS = 3146
ZIP_PATTERN  = "labels/egomotion.offline/egomotion.offline.chunk_{chunk_id:04d}.zip"
FILE_SUFFIX  = ".egomotion.offline.parquet"


def parse_args():
    parser = argparse.ArgumentParser(
        description="PhysicalAI-AV egomotion.offline 전체 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="/Data1/home/bskang/cds-data/egomotion_offline",
        help="parquet 파일 저장 경로 (기본: /Data1/home/bskang/cds-data/egomotion_offline)",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Data1/home/bskang/cds-data/egomotion_offline_zips",
        help="청크 zip 캐시 경로 (기본: /Data1/home/bskang/cds-data/egomotion_offline_zips)",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="추출 완료 후 zip 삭제 (디스크 절약)",
    )
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="시작 청크 번호 (기본: 0)",
    )
    parser.add_argument(
        "--chunk-end",
        type=int,
        default=TOTAL_CHUNKS - 1,
        help=f"끝 청크 번호 포함 (기본: {TOTAL_CHUNKS - 1})",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="분산 실행 시 이 노드의 인덱스 (0-based)",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=1,
        help="분산 실행 시 전체 노드 수",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 토큰 (미지정 시 ~/.cache/huggingface/token 사용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 다운로드 없이 계획만 출력",
    )
    return parser.parse_args()


def get_chunk_ids(args) -> list[int]:
    """처리할 청크 ID 목록 반환."""
    all_chunks = list(range(args.chunk_start, args.chunk_end + 1))
    if args.total_shards > 1:
        all_chunks = all_chunks[args.shard_index::args.total_shards]
    return all_chunks


def download_and_extract_chunk(
    chunk_id: int,
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    delete_zip: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """
    하나의 청크 zip을 다운로드하고 parquet 파일을 추출.

    Returns:
        (extracted_count, skipped_count)
    """
    from huggingface_hub import hf_hub_download

    zip_repo_path = ZIP_PATTERN.format(chunk_id=chunk_id)
    zip_local = cache_dir / f"egomotion.offline.chunk_{chunk_id:04d}.zip"

    # 이미 zip이 캐시에 있으면 재다운로드 생략
    if not zip_local.exists():
        if dry_run:
            logger.info(f"[DRY RUN] 청크 {chunk_id:04d}: {zip_repo_path} 다운로드 예정")
            return 1, 0

        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_repo_path,
                repo_type="dataset",
                local_dir=str(cache_dir),
                token=token,
            )
            # hf_hub_download가 반환하는 경로로 덮어쓰기
            zip_local = Path(downloaded)
        except Exception as e:
            logger.error(f"청크 {chunk_id:04d} 다운로드 실패: {e}")
            return 0, 0
    else:
        logger.debug(f"청크 {chunk_id:04d} 캐시 사용: {zip_local.name}")

    # zip 열어서 parquet 추출
    extracted = skipped = 0
    try:
        with zipfile.ZipFile(zip_local, "r") as zf:
            parquet_files = [n for n in zf.namelist() if n.endswith(FILE_SUFFIX)]

            for pq_name in parquet_files:
                out_path = output_dir / pq_name
                if out_path.exists():
                    skipped += 1
                    continue

                tmp_path = out_path.with_suffix(".tmp")
                try:
                    with zf.open(pq_name) as src:
                        tmp_path.write_bytes(src.read())
                    tmp_path.rename(out_path)
                    extracted += 1
                except Exception as e:
                    logger.warning(f"  {pq_name} 추출 실패: {e}")
                    if tmp_path.exists():
                        tmp_path.unlink()

    except zipfile.BadZipFile:
        logger.error(f"청크 {chunk_id:04d} 손상된 zip: {zip_local}")
        zip_local.unlink(missing_ok=True)
        return 0, 0

    if delete_zip and zip_local.exists():
        zip_local.unlink()

    return extracted, skipped


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    cache_dir  = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = get_chunk_ids(args)

    print("=" * 70)
    print("PhysicalAI-AV egomotion.offline 다운로드")
    print("=" * 70)
    print(f"처리 청크 수  : {len(chunk_ids):,}  (전체 {TOTAL_CHUNKS}개 중)")
    if args.total_shards > 1:
        print(f"분산 실행     : shard {args.shard_index}/{args.total_shards}")
    print(f"출력 디렉토리 : {output_dir}")
    print(f"zip 캐시 경로 : {cache_dir}")
    print(f"zip 삭제      : {'예' if args.delete_zips else '아니오'}")
    if args.dry_run:
        print("모드          : DRY RUN")
    print("=" * 70)

    # 기존 완료 파일 수 확인
    existing = sum(1 for _ in output_dir.glob(f"*{FILE_SUFFIX}"))
    if existing:
        logger.info(f"이미 완료된 파일: {existing:,}개 (건너뜀)")

    token = args.token

    total_extracted = total_skipped = 0
    failed_chunks = []

    pbar = tqdm(chunk_ids, desc="청크 처리", unit="chunk")
    for chunk_id in pbar:
        pbar.set_postfix(chunk=f"{chunk_id:04d}", extracted=total_extracted)

        try:
            ext, skip = download_and_extract_chunk(
                chunk_id=chunk_id,
                output_dir=output_dir,
                cache_dir=cache_dir,
                token=token,
                delete_zip=args.delete_zips,
                dry_run=args.dry_run,
            )
            total_extracted += ext
            total_skipped   += skip
        except Exception as e:
            logger.error(f"청크 {chunk_id:04d} 오류: {e}")
            failed_chunks.append(chunk_id)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("완료 요약")
    print("=" * 70)
    print(f"새로 추출됨       : {total_extracted:,}개")
    print(f"기존 파일 재사용  : {total_skipped:,}개")
    print(f"총 완료           : {total_extracted + total_skipped:,}개")
    if failed_chunks:
        print(f"실패한 청크       : {len(failed_chunks)}개  {failed_chunks[:10]}{'...' if len(failed_chunks) > 10 else ''}")

    actual = sum(1 for _ in output_dir.glob(f"*{FILE_SUFFIX}"))
    print(f"\n출력 디렉토리 내 parquet 파일: {actual:,}개")
    print(f"출력 경로: {output_dir}")

    return 1 if failed_chunks else 0


if __name__ == "__main__":
    sys.exit(main())