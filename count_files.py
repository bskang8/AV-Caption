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

"""파일 개수를 확인하는 스크립트 (효율적 버전)"""

import os
from pathlib import Path
from collections import Counter
import time


def count_files_fast(directory_path: str, recursive: bool = False, show_progress: bool = True) -> dict:
    """
    지정된 디렉토리의 파일 개수를 빠르게 세는 함수
    
    Args:
        directory_path: 확인할 디렉토리 경로
        recursive: 하위 디렉토리 포함 여부
        show_progress: 진행 상황 표시 여부
    
    Returns:
        파일 개수 정보를 담은 딕셔너리
    """
    path = Path(directory_path)
    
    if not path.exists():
        return {"error": "경로가 존재하지 않습니다."}
    
    if not path.is_dir():
        return {"error": "디렉토리가 아닙니다."}
    
    file_count = 0
    dir_count = 0
    extensions = Counter()
    
    start_time = time.time()
    
    try:
        if recursive:
            # 하위 디렉토리 포함 - os.walk 사용 (더 빠름)
            for root, dirs, files in os.walk(directory_path):
                dir_count += len(dirs)
                file_count += len(files)
                
                # 확장자 통계
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if not ext:
                        ext = "(확장자 없음)"
                    extensions[ext] += 1
                
                # 진행 상황 표시
                if show_progress and file_count % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r진행 중... 파일: {file_count:,}, 디렉토리: {dir_count:,}, 경과시간: {elapsed:.1f}초", end="", flush=True)
        else:
            # 현재 디렉토리만 - scandir 사용 (가장 빠름)
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                        ext = os.path.splitext(entry.name)[1].lower()
                        if not ext:
                            ext = "(확장자 없음)"
                        extensions[ext] += 1
                    elif entry.is_dir(follow_symlinks=False):
                        dir_count += 1
    
    except PermissionError as e:
        return {"error": f"권한 오류: {e}"}
    except Exception as e:
        return {"error": f"오류 발생: {e}"}
    
    if show_progress and recursive:
        print()  # 줄바꿈
    
    elapsed_time = time.time() - start_time
    
    return {
        "directory": str(path),
        "file_count": file_count,
        "dir_count": dir_count,
        "total_items": file_count + dir_count,
        "recursive": recursive,
        "extensions": dict(extensions.most_common()),
        "elapsed_time": elapsed_time
    }


def main():
    # 확인할 경로
    target_path = "/Data1/home/bskang/cds-data/captions"
    
    print(f"📁 디렉토리 분석: {target_path}\n")
    
    # 현재 디렉토리만 확인
    print("=" * 60)
    print("현재 디렉토리만 확인 중...")
    print("=" * 60)
    result = count_files_fast(target_path, recursive=False, show_progress=False)
    
    if "error" in result:
        print(f"❌ 오류: {result['error']}")
        return
    
    print(f"📄 파일 개수: {result['file_count']:,}")
    print(f"📂 디렉토리 개수: {result['dir_count']:,}")
    print(f"📊 전체 항목: {result['total_items']:,}")
    print(f"⏱️  소요 시간: {result['elapsed_time']:.2f}초")
    
    # 확장자별 통계 (현재 디렉토리)
    if result['extensions']:
        print("\n📋 파일 확장자별 통계 (현재 디렉토리):")
        for ext, count in list(result['extensions'].items())[:10]:
            print(f"  {ext:20s}: {count:,}")
        if len(result['extensions']) > 10:
            print(f"  ... 외 {len(result['extensions']) - 10}개")
    
    # 하위 디렉토리 포함 확인
    print("\n" + "=" * 60)
    print("하위 디렉토리 포함 확인 중...")
    print("=" * 60)
    result_recursive = count_files_fast(target_path, recursive=True, show_progress=True)
    
    print(f"📄 파일 개수: {result_recursive['file_count']:,}")
    print(f"📂 디렉토리 개수: {result_recursive['dir_count']:,}")
    print(f"📊 전체 항목: {result_recursive['total_items']:,}")
    print(f"⏱️  소요 시간: {result_recursive['elapsed_time']:.2f}초")
    
    # 확장자별 통계 (하위 포함)
    if result_recursive['extensions']:
        print("\n📋 파일 확장자별 통계 (하위 디렉토리 포함) - 상위 10개:")
        for ext, count in list(result_recursive['extensions'].items())[:10]:
            print(f"  {ext:20s}: {count:,}")
        if len(result_recursive['extensions']) > 10:
            print(f"  ... 외 {len(result_recursive['extensions']) - 10}개")


if __name__ == "__main__":
    main()