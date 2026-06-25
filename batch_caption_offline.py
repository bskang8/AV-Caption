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
Batch video captioning using cosmos-reason2 offline mode (vLLM.LLM).

Loads the model ONCE at startup, then processes all assigned videos.
Use CUDA_VISIBLE_DEVICES to pin this process to a specific GPU.

Usage (external server — shard file + path remapping):
    CUDA_VISIBLE_DEVICES=0 python batch_caption_offline.py \
        --video-list shards/shard_02.txt \
        --input-dir /root/kadap/MyDisk/cds-data/front_views \
        --output-dir /local/captions --skip-existing

  shard 파일의 경로에서 파일명만 추출해 --input-dir 아래 경로로 재매핑합니다.
  외부 서버의 영상 경로가 현재 서버와 다를 때 사용합니다.

Usage (현재 서버 — shard 파일, 경로 동일):
    CUDA_VISIBLE_DEVICES=0 python batch_caption_offline.py \
        --video-list shards/shard_00.txt --output-dir /local/captions --skip-existing

Usage (legacy shard-id mode):
    CUDA_VISIBLE_DEVICES=0 python batch_caption_offline.py \
        --shard-id 2 --num-shards 10 --output-dir /local/captions
"""

import argparse
import sys
from pathlib import Path

import qwen_vl_utils
import transformers
import vllm
import yaml

from cosmos_reason2_utils.text import SYSTEM_PROMPT, create_conversation
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_INPUT_DIR = Path("/Data1/home/bskang/cds-data/front_camera_videos")


def build_llm_input(video_path: Path, user_prompt: str, system_prompt: str,
                    processor, total_pixels: int, fps: int) -> dict:
    vision_kwargs = {"total_pixels": total_pixels, "fps": float(fps)}
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        videos=[str(video_path)],
        vision_kwargs=vision_kwargs,
    )
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_ids=False,
    )
    _, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def load_prompt(prompt_file: str) -> tuple[str, str]:
    with open(prompt_file) as f:
        config = yaml.safe_load(f)
    user_prompt = config.get("user_prompt", "").strip()
    system_prompt = config.get("system_prompt", SYSTEM_PROMPT)
    return user_prompt, system_prompt


def load_video_list(video_list: Path, input_dir: Path | None = None) -> list[Path]:
    lines = video_list.read_text(encoding="utf-8").splitlines()
    paths = [Path(line.strip()) for line in lines if line.strip()]
    if input_dir is not None:
        # shard 파일의 경로에서 파일명만 추출해 외부 서버 경로로 재매핑
        paths = [input_dir / p.name for p in paths]
    return paths


def process_videos(
    video_files: list[Path],
    output_dir: Path,
    model: str,
    prompt_file: str,
    fps: int,
    max_model_len: int,
    max_tokens: int,
    skip_existing: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = (
        [v for v in video_files if not (output_dir / (v.stem + ".txt")).exists()]
        if skip_existing else video_files
    )
    skipped = len(video_files) - len(pending)
    if skipped:
        print(f"  Skipping {skipped} already done, {len(pending)} remaining")
    if not pending:
        print("All videos already processed.")
        return

    user_prompt, system_prompt = load_prompt(prompt_file)

    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        seed=3407,
    )
    total_pixels = int((max_model_len - max_tokens) * PIXELS_PER_TOKEN * 0.9)

    print(f"Loading model: {model}")
    llm = vllm.LLM(
        model=model,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"video": 1},
    )
    processor: transformers.Qwen3VLProcessor = transformers.AutoProcessor.from_pretrained(model)
    print("Model loaded. Starting inference.")

    success = 0
    failed = 0

    for idx, video_path in enumerate(pending, 1):
        output_path = output_dir / (video_path.stem + ".txt")
        print(f"[{idx}/{len(pending)}] {video_path.name}")
        try:
            llm_input = build_llm_input(video_path, user_prompt, system_prompt,
                                        processor, total_pixels, fps)
            outputs = llm.generate([llm_input], sampling_params=sampling_params)
            content = outputs[0].outputs[0].text.strip()
            output_path.write_text(content, encoding="utf-8")
            print(f"  -> Saved: {output_path}")
            success += 1
        except Exception as e:
            print(f"  -> Failed: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}")


def main():
    parser = argparse.ArgumentParser(description="Offline batch captioning with cosmos-reason2")
    parser.add_argument(
        "--video-list",
        type=Path,
        default=None,
        help="Text file with one video path per line (generated by generate_shards.py)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "--video-list 없이 쓸 경우: 영상 디렉터리 직접 지정. "
            "--video-list 와 함께 쓸 경우: shard 파일 경로를 이 디렉터리로 재매핑 "
            "(외부 서버 경로가 다를 때 사용)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default="prompts/caption_detail.yaml")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose output .txt already exists (safe restart after failure)",
    )
    args = parser.parse_args()

    if args.video_list:
        # --input-dir가 함께 오면 경로 재매핑, 없으면 shard 파일 경로 그대로 사용
        remap_dir = args.input_dir  # None이면 재매핑 안 함
        video_files = load_video_list(args.video_list, input_dir=remap_dir)
        label = f"(remapped to {remap_dir})" if remap_dir else ""
        print(f"Loaded {len(video_files)} videos from {args.video_list} {label}")
    else:
        base_dir = args.input_dir or DEFAULT_INPUT_DIR
        all_videos = sorted(base_dir.glob("*.mp4"))
        video_files = [v for i, v in enumerate(all_videos) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(video_files)}/{len(all_videos)} videos")

    process_videos(
        video_files=video_files,
        output_dir=args.output_dir,
        model=args.model,
        prompt_file=args.prompt_file,
        fps=args.fps,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()