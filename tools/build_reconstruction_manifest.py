#!/usr/bin/env python3
"""Audit Base v4 input files without importing the training stack or using a GPU.

Paths in the JSON are relative to ``repo_root`` (including ``..`` for an external
data root). Image dimensions are read from every file header; this is not a full
pixel decode, LiDAR content audit, or SMPL pickle deserialization. G-buffer targets
belong to the next reconstruction phase and their absence does not fail this audit.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


DEFAULT_SCENES = (23, 94, 114, 172, 788)
DEFAULT_DATA_ROOT = Path("data/waymo/processed/training")
WEATHERS = ("raw", "rain", "snow")


def frame_split(start=0, end=99, stride=8):
    """Match DrivingDataset: end inclusive, first test at start + stride."""
    if start < 0 or end < start or stride < 0:
        raise ValueError("Expected 0 <= start <= end and stride >= 0")
    frames = list(range(start, end + 1))
    test = frames[stride::stride] if stride else []
    heldout = set(test)
    return {"train": [f for f in frames if f not in heldout], "test": test}


class SceneAudit:
    def __init__(self, repo_root, scene_root):
        self.repo_root = repo_root
        self.scene_root = scene_root
        self.errors = []

    def relative(self, path):
        return Path(os.path.relpath(path, self.repo_root)).as_posix()

    def error(self, path, reason):
        self.errors.append({"path": self.relative(path), "reason": reason})

    def file(self, path, required=True):
        present = path.is_file()
        result = {"path": self.relative(path), "required": required, "exists": present}
        if present:
            result["bytes"] = path.stat().st_size
            if required and not result["bytes"]:
                self.error(path, "empty_file")
        elif required:
            self.error(path, "missing_file")
        return result

    def image(self, path, expected_size=None, required=True):
        result = self.file(path, required)
        if result["exists"]:
            try:
                with Image.open(path) as image:
                    result.update(width=image.width, height=image.height, mode=image.mode)
                    if expected_size and image.size != expected_size:
                        self.error(path, f"size_mismatch: expected {expected_size}, got {image.size}")
            except (OSError, ValueError) as exc:
                self.error(path, f"unreadable_image_header: {exc}")
        return result

    def numeric(self, path, shape, rigid=False):
        result = self.file(path)
        if result["exists"]:
            try:
                value = np.loadtxt(path)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"expected finite {shape}, got {value.shape}")
                if rigid and (
                    not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5)
                    or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-3)
                ):
                    raise ValueError("invalid homogeneous rigid transform")
                result["values"] = value.tolist()
            except (OSError, ValueError) as exc:
                self.error(path, f"invalid_numeric_file: {exc}")
        return result

    def json_file(self, path):
        result = self.file(path)
        value = None
        if result["exists"]:
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                result["entry_count"] = len(value)
            except (OSError, ValueError) as exc:
                self.error(path, f"invalid_json: {exc}")
                value = None
        return result, value


def _audit_objects(audit, frame_ids):
    root = audit.scene_root / "instances"
    info_rec, info = audit.json_file(root / "instances_info.json")
    frame_rec, frames = audit.json_file(root / "frame_instances.json")
    if info is not None and frames is not None:
        requested = set(frame_ids)
        annotation_frames = {}
        try:
            if {int(k) for k in info} != set(range(len(info))):
                raise ValueError("instance IDs must be contiguous from zero")
            for key, instance in info.items():
                if instance["class_name"] not in ("Vehicle", "Pedestrian", "Cyclist"):
                    raise ValueError(f"unknown class for instance {key}")
                ann = instance["frame_annotations"]
                if not (len(ann["frame_idx"]) == len(ann["obj_to_world"]) == len(ann["box_size"])):
                    raise ValueError(f"annotation length mismatch for instance {key}")
                annotation_frames[key] = set(ann["frame_idx"])
                for frame, pose, size in zip(ann["frame_idx"], ann["obj_to_world"], ann["box_size"]):
                    if frame not in requested:
                        continue
                    pose, size = np.asarray(pose), np.asarray(size)
                    if pose.size != 16 or size.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(size).all() or not (size > 0).all():
                        raise ValueError(f"invalid pose/size for instance {key}, frame {frame}")
            for frame in frame_ids:
                if str(frame) not in frames:
                    raise ValueError(f"missing frame {frame}")
                for instance_id in frames[str(frame)]:
                    key = str(instance_id)
                    if key not in annotation_frames or frame not in annotation_frames[key]:
                        raise ValueError(f"missing instance annotation {key}, frame {frame}")
            frame_rec["requested_frames_validated"] = len(frame_ids)
        except (KeyError, ValueError, TypeError) as exc:
            audit.error(root / "instances_info.json", f"invalid_object_annotations: {exc}")
    return {"instances": info_rec, "frame_instances": frame_rec}


def audit_scene(repo_root, data_root, scene, start=0, end=99, stride=8):
    """Build an exact per-frame inventory for camera 0; scene IDs are not aliased."""
    if scene < 0:
        raise ValueError("scene must be nonnegative")
    scene_root = data_root / f"{scene:03d}"
    audit = SceneAudit(repo_root, scene_root)
    splits = frame_split(start, end, stride)
    frame_ids = list(range(start, end + 1))
    selected_dynamic = "fine_dynamic_masks" if (scene_root / "fine_dynamic_masks").exists() else "dynamic_masks"
    calibration = {
        "intrinsics_camera_0": audit.numeric(scene_root / "intrinsics/0.txt", (9,)),
        # Waymo load_objects(load_smpl=True) constructs transforms for all cameras.
        "extrinsics_for_smpl_cameras": {
            str(cam): audit.numeric(scene_root / f"extrinsics/{cam}.txt", (4, 4), rigid=True)
            for cam in range(5)
        },
        "camera_to_world_formula": "inv(ego_pose[start]) @ ego_pose[frame] @ extrinsics[camera] @ OPENCV2DATASET",
        "opencv_to_dataset": [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    }
    records = []
    dimensions = {weather: set() for weather in WEATHERS}
    for frame in frame_ids:
        stem = f"{frame:03d}_0"
        images = {}
        for weather in WEATHERS:
            images[weather] = audit.image(scene_root / f"images/{weather}/{stem}.jpg", (1920, 1280))
            if "width" in images[weather]:
                dimensions[weather].add((images[weather]["width"], images[weather]["height"]))
        lidar = audit.file(scene_root / f"lidar/{frame:03d}.bin")
        if lidar["exists"]:
            # WaymoLiDARSource reads float32 Nx14, never infer points from filename.
            if lidar["bytes"] % (14 * 4):
                audit.error(scene_root / f"lidar/{frame:03d}.bin", "lidar_size_not_divisible_by_float32_x14")
            else:
                lidar["point_count"] = lidar["bytes"] // (14 * 4)
        records.append({
            "frame_id": frame,
            "frame_name": f"{frame:03d}",
            "camera_id": 0,
            "unique_image_index": frame - start,
            "split": "test" if frame in splits["test"] else "train",
            "images": images,
            "masks": {
                "sky": audit.image(scene_root / f"sky_masks/{stem}.png", (1920, 1280)),
                **{kind: audit.image(scene_root / f"{selected_dynamic}/{kind}/{stem}.png", (1920, 1280))
                   for kind in ("all", "human", "vehicle")},
            },
            "ego_pose": audit.numeric(scene_root / f"ego_pose/{frame:03d}.txt", (4, 4), rigid=True),
            "lidar": lidar,
        })
    objects = _audit_objects(audit, frame_ids)
    objects["smpl_parameters"] = audit.file(scene_root / "humanpose/smpl.pkl")
    objects["smpl_template"] = audit.file(repo_root / "smpl_models/SMPL_NEUTRAL.pkl")
    ego_mask = audit.image(repo_root / "data/ego_masks/waymo/0.png", required=False)
    return {
        "scene_id": scene,
        "scene_name": f"{scene:03d}",
        "scene_root": audit.relative(scene_root),
        "status": "ready_for_phase_a" if not audit.errors else "required_inputs_invalid",
        "seed": 0,
        "camera_ids": [0],
        "weathers": list(WEATHERS),
        "start_timestep": start,
        "end_timestep_inclusive": end,
        "test_image_stride": stride,
        "split": splits,
        "counts": {"frames": len(frame_ids), "train": len(splits["train"]), "test": len(splits["test"])},
        "observed_dimensions_wh": {w: [list(size) for size in sorted(sizes)] for w, sizes in dimensions.items()},
        "calibration": calibration,
        "dynamic_mask_directory": selected_dynamic,
        "ego_mask": ego_mask,
        "ego_mask_absence_behavior": "CameraData.egocar_mask = None; Waymo mask is optional",
        "objects": objects,
        "planned_gbuffer": {
            "required_in_phase_a": False,
            "status": "requires_phase_b_generation_and_validation",
            "channels": ["albedo", "metallic", "roughness", "normal"],
            "weathers": list(WEATHERS),
            "target_size_wh": [1920, 1280],
            "crop_top": 0,
            "path": None,
            "path_note": "Assign only after the recovered generation/loader contract is validated.",
        },
        "frames": records,
        "required_errors": audit.errors,
    }


def build_manifest(repo_root, data_root, scenes=DEFAULT_SCENES, start=0, end=99, stride=8):
    repo_root, data_root = Path(repo_root).resolve(), Path(data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    data_root = data_root.resolve()
    scene_records = [audit_scene(repo_root, data_root, scene, start, end, stride) for scene in scenes]
    return {
        "schema_version": 1,
        "purpose": "base_v4_reconstruction_phase_a_input_audit",
        "repo_root": str(repo_root),
        "path_base": "repo_root",
        "data_root": Path(os.path.relpath(data_root, repo_root)).as_posix(),
        "status": "ready_for_phase_a" if all(s["status"] == "ready_for_phase_a" for s in scene_records) else "required_inputs_invalid",
        "validation_scope": {
            "images": "all requested image and mask headers; expected dimensions; no full pixel decode",
            "calibration": "finite intrinsic9 and rigid extrinsic/ego 4x4 matrices",
            "lidar": "nonempty file and size compatible with float32 Nx14; no point content decode",
            "objects": "JSON shape, requested frame coverage, referenced IDs, pose/size shape and finiteness",
            "smpl": "nonempty pickle/template files only; no unpickling",
            "gbuffer": "planned separately; not asserted present or valid",
            "training_readiness": "data audit only; does not validate GPU, imports, or restored PBR model",
        },
        "scenes": scene_records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--out", type=Path, help="Write JSON here; otherwise print JSON without writing files")
    args = parser.parse_args(argv)
    if any(s < 0 for s in args.scenes) or len(set(args.scenes)) != len(args.scenes):
        parser.error("scene IDs must be nonnegative and unique")
    manifest = build_manifest(args.repo_root, args.data_root, args.scenes)
    serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
        for scene in manifest["scenes"]:
            print(f"scene {scene['scene_name']}: {scene['status']}; "
                  f"train={scene['counts']['train']} test={scene['counts']['test']} "
                  f"errors={len(scene['required_errors'])}")
        print(f"Manifest: {args.out.resolve()}")
    else:
        sys.stdout.write(serialized)
    return 0 if manifest["status"] == "ready_for_phase_a" else 1


if __name__ == "__main__":
    raise SystemExit(main())
