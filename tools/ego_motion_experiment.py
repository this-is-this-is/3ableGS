#!/usr/bin/env python3
"""Frozen-input ego-motion diagnosis, material-head training and evaluation.

All model features originate in the frozen renderer. References are kept under
separate keys and explicitly removed before calling a learned head. Evaluation
history is confined to each held-out contiguous block; train history never
crosses a held-out frame. First held-out block is validation, remaining blocks
are test. Existing inspected scenes remain exploratory, not confirmatory tests.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = frozenset({"rgb", "albedo", "roughness", "metallic", "normal", "depth",
                        "opacity", "static_mask", "camera_to_world", "intrinsics",
                        "viewdirs", "frame_id"})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_to(value, device):
    import torch
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    return value


def model_inputs(packet):
    return {key: packet[key] for key in MODEL_KEYS}


def verify_cache_source(identity):
    checkpoint = Path(identity["checkpoint"])
    if (digest(checkpoint) != identity["checkpoint_sha256"] or
            digest(checkpoint.parent / "config.yaml") != identity["config_sha256"]):
        raise ValueError("Source checkpoint/config changed since render-cache creation")


def read_motion(path, base_identity):
    data = json.loads(Path(path).read_text())
    if data.get("cache_identity") != base_identity:
        raise ValueError("Motion diagnostics must come from this frozen base cache")
    return {row["frame_id"]: row["motion_bin"] for row in data["frames"]}


def history_frames(frame, allowed, gaps=(1, 2, 4)):
    """Missing intermediate frames are a split boundary, even for a long gap."""
    allowed = set(allowed)
    if frame not in allowed:
        return []
    return [frame - gap for gap in gaps
            if all(index in allowed for index in range(frame - gap, frame + 1))]


class FrameCache:
    def __init__(self, path, capacity=12):
        self.path = Path(path)
        self.meta = json.loads((self.path / "manifest.json").read_text())
        if self.meta.get("status") != "completed":
            raise ValueError(f"Incomplete render cache: {path}")
        self.frames = set(self.meta["frames"])
        self.capacity = capacity
        self.memory = OrderedDict()

    def get(self, frame, device="cpu"):
        import torch
        if frame not in self.frames:
            raise ValueError(f"Frame {frame} absent from cache")
        if frame not in self.memory:
            path = self.path / "frames" / f"{frame:06d}.pt"
            if digest(path) != self.meta["hashes"].get(str(frame)):
                raise ValueError(f"Render cache checksum mismatch: {path}")
            self.memory[frame] = torch.load(path,
                                           map_location="cpu", weights_only=False)
            if len(self.memory) > self.capacity:
                self.memory.popitem(last=False)
        self.memory.move_to_end(frame)
        return tree_to(self.memory[frame], device)


def export_cache(args):
    from tools.ego_motion_runtime import SceneRuntime
    runtime = SceneRuntime(args.checkpoint, args.downscale, args.antialiased)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
                "config_sha256": digest(args.checkpoint.parent / "config.yaml"),
                "downscale": args.downscale, "antialiased": args.antialiased, "split": args.split}
    manifest = output / "manifest.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {}
    if state and state["identity"] != identity:
        raise ValueError("Cache inputs changed; use a new output directory")
    frames = list(runtime.frame_ids if args.split == "all" else runtime.val_frames + runtime.test_frames)
    state.update(identity=identity, status="running", frames=frames,
                 train_frames=list(runtime.train_frames), val_frames=list(runtime.val_frames),
                 test_frames=list(runtime.test_frames), hashes=state.get("hashes", {}),
                 scene=int(runtime.cfg.data.scene_idx))
    atomic_json(manifest, state)
    for frame in frames:
        path = output / "frames" / f"{frame:06d}.pt"
        if path.exists() and state["hashes"].get(str(frame)) == digest(path):
            continue
        packet = runtime.render(frame)
        atomic_torch(path, tree_to(packet, "cpu"))
        state["hashes"][str(frame)] = digest(path)
        atomic_json(manifest, state)
        print(f"cached {frame + 1}/{len(runtime.frame_ids)}", flush=True)
    state["status"] = "completed"
    atomic_json(manifest, state)


def motion_statistics(current, previous):
    """Pose/depth displacement before variant gates; no RGB/GT dependence."""
    import torch
    h, w = current["depth"].shape[:2]
    depth = current["depth"][..., 0]
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device),
                            torch.arange(w, device=depth.device), indexing="ij")
    pixels = torch.stack((xx.float() + .5, yy.float() + .5, torch.ones_like(xx).float()), -1)
    xyz = (pixels @ torch.linalg.inv(current["intrinsics"]).T) * depth[..., None]
    transform = torch.linalg.inv(previous["camera_to_world"]) @ current["camera_to_world"]
    xyz_previous = xyz @ transform[:3, :3].T + transform[:3, 3]
    projected = xyz_previous @ previous["intrinsics"].T
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    displacement = (uv - pixels[..., :2]).norm(dim=-1)
    valid = current["static_mask"].bool().reshape(h, w) & torch.isfinite(displacement) & (depth > 0)
    positive = valid & (xyz_previous[..., 2] > 0)
    bounds = positive & (uv[..., 0] >= .5) & (uv[..., 0] <= w - .5) & (uv[..., 1] >= .5) & (uv[..., 1] <= h - .5)
    values = displacement[positive]
    rotation = transform[:3, :3]
    angle = torch.acos(((rotation.trace() - 1) / 2).clamp(-1, 1)) * (180 / math.pi)
    return {"displacement_p50": float(values.median()) if values.numel() else None,
            "displacement_p90": float(torch.quantile(values, .9)) if values.numel() else None,
            "rotation_deg": float(angle), "translation_m": float(transform[:3, 3].norm()),
            "frustum_overlap": float(bounds.sum() / valid.sum().clamp_min(1))}


def region_masks(packet, previous=None):
    import torch
    import torch.nn.functional as F
    from models.ego_temporal import project_history
    depth = packet["depth"].permute(2, 0, 1)[None]
    high = F.max_pool2d(depth, 3, 1, 1)
    low = -F.max_pool2d(-depth, 3, 1, 1)
    edge = ((high - low) / depth.clamp_min(.01))[0, 0] > .05
    static = packet["evaluation_static"].bool().reshape(edge.shape)
    cosine = (F.normalize(packet["normal"], dim=-1) *
              -F.normalize(packet["viewdirs"], dim=-1)).sum(-1).abs()
    supported = torch.zeros_like(static)
    if previous is not None:
        supported = project_history(model_inputs(packet), model_inputs(previous))["valid"].squeeze(-1)
    return {"static_interior": static & ~edge, "depth_boundary": static & edge,
            "reflective_grazing": static & ((packet["roughness"][..., 0] < .25) | (cosine < .2)),
            "unsupported_proxy": static & ~supported,
            "geometry_supported": static & supported}


def diagnose(args):
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cache = FrameCache(args.cache)
    rows = []
    for frame in sorted(cache.frames):
        packet = cache.get(frame)
        previous = cache.get(frame - 1) if frame - 1 in cache.frames else None
        row = {"frame_id": frame, "displacement_p50": None}
        if previous is not None:
            row.update(motion_statistics(packet, previous))
        rows.append(row)
        split_frames = next(set(cache.meta[key]) for key in ("train_frames", "val_frames", "test_frames")
                            if frame in cache.meta[key])
        region_previous = previous if frame - 1 in split_frames else None
        atomic_torch(args.output_dir / "regions" / f"{frame:06d}.pt", region_masks(packet, region_previous))
    # Thresholds are fit on adjacent TRAIN pairs only; no validation/test tuning.
    train = set(cache.meta["train_frames"])
    fit = [row["displacement_p50"] for row in rows if row["frame_id"] in train and
           row["frame_id"] - 1 in train and row["displacement_p50"] is not None]
    if not fit:
        raise ValueError("No consecutive training pair for motion calibration")
    thresholds = torch.quantile(torch.tensor(fit), torch.tensor([1/3, 2/3])).tolist()
    for row in rows:
        value = row["displacement_p50"]
        row["motion_bin"] = ("unknown" if value is None else "low" if value <= thresholds[0]
                             else "high" if value > thresholds[1] else "medium")
    atomic_json(args.output_dir / "motion.json", {
        "frames": rows, "thresholds": thresholds, "units": "render-resolution pixels",
        "calibration": "frozen base camera/depth, adjacent training pairs only",
        "cache_identity": cache.meta["identity"],
        "limitations": ["unsupported_proxy is failed geometric reprojection, not measured disocclusion",
                        "Pose/depth diagnostics cannot distinguish pose error from depth error alone"]})


def make_head(name, **overrides):
    from models.ego_temporal import EgoTemporalHead
    settings = {"kind": "transformer" if name == "transformer" else "cnn",
                "guides": "geometry" if name == "geometry" else "full",
                "history": name != "current", "use_offset": name != "no_offset", "width": 32}
    settings.update(overrides)
    return EgoTemporalHead(**settings)


def sample_frame(rng, frames, motion, balanced=True):
    if not balanced or not motion:
        return rng.choice(list(frames))
    bins = {}
    for frame in frames:
        label = motion.get(frame, "low")
        bins.setdefault("low" if label == "unknown" else label, []).append(frame)
    return rng.choice(bins[rng.choice(sorted(bins))])


def gradient_loss(predicted, reference, mask):
    import torch
    terms = []
    for dim in (0, 1):
        difference = torch.diff(predicted - reference, dim=dim).abs().mean(-1)
        selected = (mask[1:] & mask[:-1]) if dim == 0 else (mask[:, 1:] & mask[:, :-1])
        terms.append(torch.where(selected, difference, torch.zeros_like(difference)).sum() /
                     selected.sum().clamp_min(1))
    return sum(terms) / 2


def train_head(args):
    import numpy as np
    import torch
    from models.ego_temporal import material_consistency
    from models.pbr import srgb_to_linear
    from tools.ego_motion_runtime import SceneRuntime
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    cache = FrameCache(args.cache)
    metadata = cache.meta["identity"]
    verify_cache_source(metadata)
    runtime = SceneRuntime(Path(metadata["checkpoint"]), metadata["downscale"], metadata["antialiased"])
    model = make_head(args.variant).to(runtime.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    allowed = sorted(set(cache.meta["train_frames"]) & cache.frames)
    if not allowed:
        raise ValueError("Training frames absent from cache")
    motion = read_motion(args.motion_manifest, metadata)
    identity = {"cache_identity": metadata, "variant": args.variant, "steps": args.steps,
                "seed": args.seed, "lr": args.lr, "temporal_weight": args.temporal_weight,
                "motion_sha256": digest(args.motion_manifest)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "resume.pt"
    step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location=runtime.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Head resume options changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        step = state["step"]
    elif any(args.output_dir.iterdir()):
        raise ValueError("Nonempty head output without resume.pt")
    else:
        atomic_torch(state_path, {"identity": identity, "step": 0, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "sampler_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
    atomic_json(args.output_dir / "manifest.json", {"identity": identity, "status": "training"})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses_path = args.output_dir / "losses.jsonl"
    if losses_path.exists():
        retained = [line for line in losses_path.read_text().splitlines() if json.loads(line)["step"] <= step]
        losses_path.write_text("".join(line + "\n" for line in retained))
    with (args.output_dir / "losses.jsonl").open("a") as log:
        for step in range(step + 1, args.steps + 1):
            frame = sample_frame(rng, allowed, motion)
            current = cache.get(frame, runtime.device)
            prior_ids = history_frames(frame, allowed)
            histories = [model_inputs(cache.get(index, runtime.device)) for index in prior_ids]
            optimizer.zero_grad(set_to_none=True)
            result = model(model_inputs(current), histories)
            predicted = runtime.shade(current, result["albedo"])
            mask = current["static_mask"].bool().reshape(predicted.shape[:2])
            photometric = ((predicted - current["target_rgb"]).abs().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            edge = gradient_loss(predicted, current["target_rgb"], mask)
            anchor = (srgb_to_linear(result["albedo"]) - srgb_to_linear(current["albedo"])).abs().mean()
            temporal = predicted.sum() * 0
            if prior_ids and args.temporal_weight:
                previous = cache.get(prior_ids[0], runtime.device)
                older = [model_inputs(cache.get(index, runtime.device))
                         for index in history_frames(prior_ids[0], allowed)]
                with torch.no_grad():
                    previous_result = model(model_inputs(previous), older)
                temporal = material_consistency({**model_inputs(current), "albedo": result["albedo"]},
                                                {**model_inputs(previous), "albedo": previous_result["albedo"]})["loss"]
            loss = photometric + .2 * edge + .01 * anchor + args.temporal_weight * temporal
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite head loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 20 == 0 or step == args.steps:
                record = {"step": step, "frame_id": frame, "loss": float(loss.detach()),
                          "rgb": float(photometric.detach()), "temporal": float(temporal.detach())}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)
            if step % 100 == 0 or step == args.steps:
                atomic_torch(state_path, {"identity": identity, "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "sampler_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
    atomic_torch(args.output_dir / "head_final.pt", {"variant": args.variant, "model": model.state_dict(),
                                                    "identity": identity, "step": step})
    atomic_json(args.output_dir / "manifest.json", {"identity": identity, "status": "completed",
                                                   "trainable_parameters": trainable, "steps": step})


def metric_summary(records):
    from utils.temporal_metrics import summarize_metric_records
    result = summarize_metric_records(records)
    result["frames"] = len(records)
    keys = set().union(*(row.keys() for row in records)) if records else set()
    for key in sorted(keys):
        if key.endswith("_squared_error"):
            prefix = key[:-len("_squared_error")]
            error = sum(row.get(key, 0) for row in records)
            count = sum(row.get(prefix + "_values", 0) for row in records)
            result[prefix + "_psnr"] = -10 * math.log10(max(error / count, 1e-12)) if count else None
    for key in ("lpips", "forward_ms", "head_ms", "history_coverage", "history_weight"):
        values = [row[key] for row in records if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def evaluate(args):
    import numpy as np
    import torch
    from utils.temporal_metrics import TemporalEvaluator
    from tools.ego_motion_runtime import SceneRuntime
    cache = FrameCache(args.cache)
    base = FrameCache(args.base_cache or args.cache)
    if any(base.meta[key] != cache.meta[key] for key in ("scene", "train_frames", "val_frames", "test_frames")):
        raise ValueError("Baseline and evaluated cache must have the same scene and splits")
    identity = cache.meta["identity"]
    verify_cache_source(identity)
    runtime = SceneRuntime(Path(identity["checkpoint"]), identity["downscale"], identity["antialiased"])
    model = None
    if args.head:
        saved = torch.load(args.head, map_location=runtime.device, weights_only=False)
        if saved["identity"]["cache_identity"] != identity:
            raise ValueError("Head was trained for a different renderer cache")
        model = make_head(saved["variant"], guide_ablation=args.guide_ablation).to(runtime.device)
        model.load_state_dict(saved["model"])
        model.eval()
    elif args.guide_ablation != "none":
        raise ValueError("Guide ablation requires --head")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "cache_identity": identity,
                "head": str(args.head) if args.head else None, "guide_ablation": args.guide_ablation}
    atomic_json(args.output_dir / "manifest.json", manifest)
    motion = read_motion(args.motion_manifest, base.meta["identity"])
    all_records = []
    summaries = {}
    with torch.no_grad():
        for split in ("val", "test"):
            frames = sorted(cache.meta[split + "_frames"])
            evaluator = TemporalEvaluator(args.flow_cache, metadata={"scene": cache.meta["scene"], "camera": 0})
            records = []
            for frame in frames:
                current = cache.get(frame, runtime.device)
                base_packet = base.get(frame)
                if (base_packet["rgb"].shape != current["rgb"].shape or
                        not torch.equal(base_packet["target_rgb"], current["target_rgb"].cpu())):
                    raise ValueError("Baseline reference/resolution differs from evaluated cache")
                histories = [model_inputs(cache.get(index, runtime.device)) for index in history_frames(frame, frames)]
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = model(model_inputs(current), histories) if model is not None else None
                predicted = runtime.shade(current, result["albedo"]) if result is not None else current["rgb"]
                torch.cuda.synchronize()
                head_ms = (time.perf_counter() - start) * 1000
                reference = current["target_rgb"].cpu().numpy().clip(0, 1)
                rgb = predicted.clamp(0, 1).cpu().numpy()
                baseline = base_packet["rgb"].numpy().clip(0, 1)
                static = current["evaluation_static"].cpu().numpy().astype(bool).reshape(rgb.shape[:2])
                row = {"frame_id": frame, "split": split, "motion_bin": motion.get(frame, "unknown"),
                       "head_ms": head_ms, "forward_ms": current.get("forward_ms"),
                       **evaluator.evaluate(reference, {"prediction": rgb, "baseline": baseline}, static, frame)}
                runtime.trainer.lpips.reset()
                row["lpips"] = float(runtime.trainer.lpips(predicted.clamp(0, 1).permute(2, 0, 1)[None],
                                                          current["target_rgb"].clamp(0, 1).permute(2, 0, 1)[None]))
                regions = torch.load(args.motion_manifest.parent / "regions" / f"{frame:06d}.pt", weights_only=True)
                regions.update(full=torch.ones_like(current["static_mask"].cpu(), dtype=torch.bool).reshape(rgb.shape[:2]),
                               static=torch.from_numpy(static))
                for name, region in regions.items():
                    mask = region.numpy().astype(bool).reshape(rgb.shape[:2])
                    for method, array in (("prediction", rgb), ("baseline", baseline)):
                        errors = np.square(array.astype(np.float64) - reference)[mask]
                        row[f"{method}_{name}_squared_error"] = float(errors.sum())
                        row[f"{method}_{name}_values"] = int(errors.size)
                valid = result["valid"].cpu().numpy().reshape(rgb.shape[:2]) if result is not None else np.zeros(rgb.shape[:2], bool)
                row["history_coverage"] = float(valid[static].mean()) if static.any() else None
                row["history_weight"] = float(result["history_weight"].mean()) if result is not None else 0.
                prediction_path = args.output_dir / "predictions" / f"{frame:06d}.npz"
                prediction_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(prediction_path, rgb=rgb, valid=valid)
                records.append(row)
                all_records.append(row)
            summaries[split] = metric_summary(records)
            summaries[split]["by_motion"] = {name: metric_summary([r for r in records if r["motion_bin"] == name])
                                              for name in ("low", "medium", "high")}
    (args.output_dir / "frames.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in all_records))
    if not getattr(args, "no_video", False):
        import imageio.v2 as imageio
        from tools.render_temporal_material import comparison_frame
        with imageio.get_writer(str(args.output_dir / "comparison.mp4"), fps=10, codec="libx264",
                                quality=8, macro_block_size=1, pixelformat="yuv420p") as writer:
            for row in all_records:
                frame = row["frame_id"]
                with np.load(args.output_dir / "predictions" / f"{frame:06d}.npz") as file:
                    writer.append_data(comparison_frame(base.get(frame)["rgb"].numpy().clip(0, 1),
                        file["rgb"], file["valid"][..., None].astype(np.float32),
                        f"{row['split']} frame {frame:04d}", domain="material",
                        titles=("Frozen base", "Prediction", "History support [0, 1]")))
    summaries["limitations"] = ["Single scene exploratory results; no scene-level confidence interval",
        "Reference flow is estimated; unsupported_proxy is not GT disocclusion",
        "Cached forward timing excludes I/O; head timing includes re-shading",
        "First saved held-out block is validation; remaining blocks are test; never used by optimizer"]
    atomic_json(args.output_dir / "summary.json", summaries)
    atomic_json(args.output_dir / "manifest.json", {**manifest, "status": "completed"})


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cache = subs.add_parser("cache")
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--downscale", type=float, default=2.)
    cache.add_argument("--antialiased", action="store_true")
    cache.add_argument("--split", choices=("all", "heldout"), default="all")
    for name in ("diagnose", "train-head", "evaluate"):
        sub = subs.add_parser(name)
        sub.add_argument("--cache", type=Path, required=True)
        if name != "diagnose":
            sub.add_argument("--motion-manifest", type=Path, required=True)
        if name == "train-head":
            sub.add_argument("--variant", choices=("current", "geometry", "full", "no_offset", "transformer"), required=True)
            sub.add_argument("--steps", type=int, default=2000)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--lr", type=float, default=1e-4)
            sub.add_argument("--temporal-weight", type=float, default=.05)
        elif name == "evaluate":
            sub.add_argument("--base-cache", type=Path)
            sub.add_argument("--head", type=Path)
            sub.add_argument("--flow-cache", type=Path)
            sub.add_argument("--no-video", action="store_true")
            sub.add_argument("--guide-ablation", default="none", choices=("none", "constant_normal", "shuffle_normal", "constant_roughness", "shuffle_roughness"))
    for sub in subs.choices.values():
        sub.add_argument("--output-dir", type=Path, required=True)
    return parser


def main():
    args = make_parser().parse_args()
    os.chdir(ROOT)
    for key in ("steps", "lr", "downscale"):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if hasattr(args, "temporal_weight") and (not math.isfinite(args.temporal_weight) or args.temporal_weight < 0):
        raise ValueError("temporal-weight must be finite and nonnegative")
    {"cache": export_cache, "diagnose": diagnose, "train-head": train_head, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
