#!/usr/bin/env python3
"""Prepare two overlapping 704x1280 crops for full-frame Waymo DR inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


SOURCE_SIZE = (1920, 1280)  # width, height
CROP_SIZE = (1920, 1056)
DR_SIZE = (1280, 704)
CROPS = {
    "top": (0, 0, 1920, 1056),
    "bottom": (0, 224, 1920, 1280),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Folder containing ordered *_0.jpg frames")
    parser.add_argument("--dst", required=True, help="Output dataset root for Diffusion Renderer")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First source frame ID (inclusive)")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--label", default="snow", help="Condition prefix for the two crop folders")
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.frames < 1:
        parser.error("--frames must be positive")

    src = Path(args.src)
    dst = Path(args.dst)
    end_frame = args.start_frame + args.frames
    frame_ids = range(args.start_frame, end_frame)
    frames = [src / f"{index:03d}_0.jpg" for index in frame_ids]
    missing = [path for path in frames if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"expected consecutive source frames {args.start_frame:03d}..{end_frame - 1:03d}; "
            f"missing {missing[0]}"
        )

    for position in CROPS:
        (dst / f"{args.label}_{position}").mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    for index, path in zip(frame_ids, frames):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if image.size != SOURCE_SIZE:
            raise ValueError(f"{path} is {image.size}, expected {SOURCE_SIZE}")

        for position, box in CROPS.items():
            tile = image.crop(box).resize(DR_SIZE, Image.Resampling.LANCZOS)
            tile.save(dst / f"{args.label}_{position}" / f"{index:03d}.png")
        manifest_frames.append({"index": index, "source": str(path.resolve())})

    manifest = {
        "source_size_wh": SOURCE_SIZE,
        "crop_size_wh": CROP_SIZE,
        "dr_size_wh": DR_SIZE,
        "condition": args.label,
        "start_frame": args.start_frame,
        "frame_count": args.frames,
        "crops_xyxy": {f"{args.label}_{position}": box for position, box in CROPS.items()},
        "blend": "cosine feather across y=224:1056 after restoring each crop to 1920x1056",
        "frames": manifest_frames,
    }
    (dst / "layout.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(frames)} frames x {len(CROPS)} crops under {dst}")


if __name__ == "__main__":
    main()
