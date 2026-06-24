"""
Generate shard files for distributed captioning.

Scans the video directory, excludes already-captioned videos,
then splits the remaining work evenly across N shard files.

Run this ONCE on the current server before starting any workers.

Usage:
    python generate_shards.py --num-shards 10 --output-dir shards/
    python generate_shards.py --num-shards 10 --output-dir shards/ --dry-run
"""

import argparse
from pathlib import Path


DEFAULT_INPUT_DIR = Path("/Data1/home/bskang/cds-data/front_camera_videos")
DEFAULT_CAPTIONS_DIR = Path("/Data1/home/bskang/cds-data/captions")


def generate_shards(
    input_dir: Path,
    captions_dir: Path,
    num_shards: int,
    output_dir: Path,
    dry_run: bool,
):
    print(f"Scanning videos : {input_dir}")
    all_videos = sorted(input_dir.glob("*.mp4"))
    print(f"  Total videos   : {len(all_videos)}")

    done_stems: set[str] = set()
    if captions_dir.exists():
        done_stems = {p.stem for p in captions_dir.glob("*.txt")}
    print(f"  Already done   : {len(done_stems)}")

    pending = [v for v in all_videos if v.stem not in done_stems]
    print(f"  Pending        : {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    shards: list[list[Path]] = [[] for _ in range(num_shards)]
    for i, video in enumerate(pending):
        shards[i % num_shards].append(video)

    if dry_run:
        print("\n[dry-run] Shard sizes:")
        for idx, shard in enumerate(shards):
            print(f"  shard_{idx:02d}.txt : {len(shard)} videos")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(shards):
        shard_file = output_dir / f"shard_{idx:02d}.txt"
        shard_file.write_text("\n".join(str(v) for v in shard) + "\n", encoding="utf-8")
        print(f"  Written: {shard_file}  ({len(shard)} videos)")

    print(f"\nDone. {num_shards} shard files written to {output_dir}")
    print("\nNext steps:")
    print(f"  Current server  : use shard_00.txt (port 8000) and shard_01.txt (port 8001)")
    print(f"  External server : rsync shards/shard_02.txt ~ shard_09.txt then run batch_caption_offline.py")


def main():
    parser = argparse.ArgumentParser(description="Generate shard files for distributed captioning")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--captions-dir", type=Path, default=DEFAULT_CAPTIONS_DIR,
                        help="Directory with already-completed .txt captions")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("shards"),
                        help="Where to write shard_XX.txt files (default: ./shards)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show shard sizes without writing files")
    args = parser.parse_args()

    generate_shards(
        input_dir=args.input_dir,
        captions_dir=args.captions_dir,
        num_shards=args.num_shards,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
