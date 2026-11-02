#!/usr/bin/env python3
"""Restore and blend top/bottom Diffusion Renderer G-buffer crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


FULL_SIZE = (1920, 1280)  # width, height
CROP_SIZE = (1920, 1056)
BOTTOM_Y = 224
OVERLAP_START = BOTTOM_Y
OVERLAP_END = CROP_SIZE[1]
PASSES = ("basecolor", "normal", "depth", "roughness", "metallic")
NORMAL_VALID_EPS = 0.1


def resize_pass(path: Path, pass_name: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    resample = Image.Resampling.BILINEAR if pass_name == "depth" else Image.Resampling.BICUBIC
    return np.asarray(image.resize(CROP_SIZE, resample), dtype=np.float32) / 255.0


def blend(top: np.ndarray, bottom: np.ndarray, pass_name: str) -> np.ndarray:
    canvas_top = np.empty((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.float32)
    canvas_bottom = np.empty_like(canvas_top)
    canvas_top[: CROP_SIZE[1]] = top
    canvas_top[CROP_SIZE[1] :] = top[-1]
    canvas_bottom[:BOTTOM_Y] = bottom[0]
    canvas_bottom[BOTTOM_Y:] = bottom

    weights_top = np.ones(FULL_SIZE[1], dtype=np.float32)
    weights_top[OVERLAP_END:] = 0.0
    t = np.linspace(0.0, 1.0, OVERLAP_END - OVERLAP_START, dtype=np.float32)
    weights_top[OVERLAP_START:OVERLAP_END] = 0.5 * (1.0 + np.cos(math.pi * t))
    weights_top = weights_top[:, None, None]

    if pass_name == "normal":
        top_vec = canvas_top * 2.0 - 1.0
        bottom_vec = canvas_bottom * 2.0 - 1.0
        top_norm = np.linalg.norm(top_vec, axis=-1, keepdims=True)
        bottom_norm = np.linalg.norm(bottom_vec, axis=-1, keepdims=True)
        top_valid = top_norm > NORMAL_VALID_EPS
        bottom_valid = bottom_norm > NORMAL_VALID_EPS

        # DR encodes undefined background normals as neutral gray.  Resizing and
        # 8-bit quantization perturb that zero vector slightly, so normalizing it
        # would amplify the perturbation into saturated, random sky normals.
        top_unit = np.divide(top_vec, top_norm, out=np.zeros_like(top_vec), where=top_valid)
        bottom_unit = np.divide(
            bottom_vec, bottom_norm, out=np.zeros_like(bottom_vec), where=bottom_valid
        )
        top_weight = weights_top * top_valid
        bottom_weight = (1.0 - weights_top) * bottom_valid
        value = top_weight * top_unit + bottom_weight * bottom_unit
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        value = np.divide(value, norm, out=np.zeros_like(value), where=norm > 1e-6)
        value = value * 0.5 + 0.5
    else:
        value = weights_top * canvas_top + (1.0 - weights_top) * canvas_bottom
    return (value * 255.0).round().clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", required=True, help="DR output root containing gbuffer_frames")
    parser.add_argument("--dst", required=True, help="Destination folder for blended full-frame maps")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame ID to stitch (inclusive)")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--condition", default="snow", help="Condition prefix used by the tile folders")
    parser.add_argument("--passes", nargs="+", default=list(PASSES),
                        choices=PASSES, help="Passes present in the tile folders")
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.frames < 1:
        parser.error("--frames must be positive")

    tile_root = Path(args.tile_root) / "gbuffer_frames"
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    end_frame = args.start_frame + args.frames
    for index in range(args.start_frame, end_frame):
        for pass_name in args.passes:
            filename = f"full.{index:05d}.{pass_name}.png"
            top_path = tile_root / f"{args.condition}_top" / filename
            bottom_path = tile_root / f"{args.condition}_bottom" / filename
            if not top_path.is_file() or not bottom_path.is_file():
                raise FileNotFoundError(f"missing tile pair for {filename}: {top_path}, {bottom_path}")
            result = blend(
                resize_pass(top_path, pass_name),
                resize_pass(bottom_path, pass_name),
                pass_name,
            )
            Image.fromarray(result).save(dst / filename)

    metadata = {
        "frames": args.frames,
        "start_frame": args.start_frame,
        "frame_ids": [args.start_frame, end_frame - 1],
        "condition": args.condition,
        "passes": args.passes,
        "output_size_wh": FULL_SIZE,
        "top_source_box_xyxy": [0, 0, 1920, 1056],
        "bottom_source_box_xyxy": [0, 224, 1920, 1280],
        "overlap_y": [OVERLAP_START, OVERLAP_END],
        "blend": "cosine feather; normals decoded, blended, and renormalized",
    }
    (dst.parent / "stitch_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"stitched {args.frames * len(args.passes)} maps into {dst}")


if __name__ == "__main__":
    main()
