#!/usr/bin/env python3
"""Validate fresh joint layered training on CUDA; not a quality experiment.

Initialize every scene component from the dataset, retain the original loss,
optimizer groups and schedules, and run the ordinary training callbacks on
training images. Native saved image resolution is the default. No checkpoint
is loaded or written. The JSON records finite gradients, observed parameter
updates and the layered contributions consumed by densification statistics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Artificial schedule position for callback validation, e.g. 599 with --steps 3")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Additional divisor of saved resolution; 1 is native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Original training config overrides, e.g. data.scene_idx=788")
    return parser


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gradient_inventory(optimizer):
    """Record exact finite-gradient checks and one update witness per tensor.

    The witness is the element with the largest gradient. It avoids retaining
    a second full scene while still directly observing the optimizer update.
    """
    import torch

    groups, witnesses = {}, {}
    for group in optimizer.param_groups:
        name = group["name"]
        record = {"lr": float(group["lr"]), "elements": 0,
                  "trainable_elements": 0, "tensors_with_grad": 0,
                  "gradient_max_abs": 0.0}
        points = []
        for parameter in group["params"]:
            record["elements"] += parameter.numel()
            record["trainable_elements"] += parameter.numel() if parameter.requires_grad else 0
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                raise ValueError("This dense native-resolution probe requires sparse_grad=false")
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite gradient in {name}")
            record["tensors_with_grad"] += 1
            if gradient.numel():
                absolute = gradient.abs().reshape(-1)
                index = int(absolute.argmax())
                maximum = float(absolute[index])
                record["gradient_max_abs"] = max(record["gradient_max_abs"], maximum)
                if maximum > 0:
                    points.append((parameter, index, parameter.detach().reshape(-1)[index].clone()))
        groups[name] = record
        witnesses[name] = points
    return groups, witnesses


def update_inventory(groups, witnesses):
    import torch

    with torch.no_grad():
        for name, points in witnesses.items():
            differences = [float((parameter.reshape(-1)[index] - previous).abs())
                           for parameter, index, previous in points]
            groups[name]["update_witness_count"] = len(differences)
            groups[name]["updated_witness_count"] = sum(value > 0 for value in differences)
            groups[name]["update_witness_max_abs"] = max(differences, default=0.0)


def required_categories(groups):
    """Names are the original trainer's optimizer groups, not a new optimizer."""
    categories = {
        "geometry_positions": lambda name: name.endswith("#xyz"),
        "opacity": lambda name: name.endswith("#opacity"),
        "material": lambda name: "#mlp.raw" in name and "normal" not in name,
        "environment": lambda name: name.startswith("EnvLight#"),
        "camera_response": lambda name: name.startswith("CameraResponse#"),
    }
    result = {category: [name for name in groups if predicate(name)]
              for category, predicate in categories.items()}
    missing = [category for category, names in result.items() if not names]
    if missing:
        raise AssertionError(f"Required original optimizer categories missing: {missing}")
    return result


def verify_optimizer_membership(trainer):
    expected = {name: list(parameters) for model in trainer.models.values()
                for name, parameters in model.get_param_groups().items()}
    actual = {group["name"]: group["params"] for group in trainer.optimizer.param_groups}
    if set(expected) != set(actual) or any(
            {id(parameter) for parameter in expected[name]} != {id(parameter) for parameter in actual[name]}
            for name in expected):
        raise AssertionError("Original model optimizer groups were omitted or replaced")
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            state = trainer.optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state and state[name].shape != parameter.shape:
                    raise AssertionError(f"Optimizer {name} shape disagrees after refinement: {group['name']}")
    return actual


def run(args, report):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from datasets.driving_dataset import DrivingDataset
    from models.temporal_training import TemporalMaterialTraining
    from tools.ego_motion_runtime import packet_to
    from utils.config import load_training_config
    from utils.misc import import_str

    if not torch.cuda.is_available():
        raise RuntimeError("This integration validation requires CUDA")
    os.chdir(ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = load_training_config(args.config_file, args.opts)
    cfg.seed = args.seed
    cfg.log_dir = str(args.output.parent)
    if cfg.trainer.render.get("packed", False) or cfg.trainer.render.get("sparse_grad", False):
        raise ValueError("The original densification callbacks require packed=false and sparse_grad=false")
    if cfg.trainer.get("layered_gbuffer") is None:
        cfg.trainer.layered_gbuffer = {
            "grouping": "instance", "color_space": "srgb", "max_layers": 0,
            "ablation": "none", "diagnostics": False,
        }
    report.update({"purpose": "implementation_validation_only_not_quality_evaluation",
                   "initialization": "fresh_dataset_no_checkpoint",
                   "layered_gbuffer": OmegaConf.to_container(cfg.trainer.layered_gbuffer, resolve=True),
                   "resolution_downscale": args.downscale, "seed": args.seed,
                   "start_step": args.start_step,
                   "artificial_schedule_position": args.start_step != 0,
                   "optimizer_policy": "original_initialize_optimizer_all_groups_and_schedules",
                   "loss_policy": "original_compute_losses_and_temporal_training_config",
                   "update_measurement": "largest-gradient element per parameter tensor",
                   "gpu": torch.cuda.get_device_name(), "steps": []})
    atomic_json(args.output, report)
    initialization_start = time.perf_counter()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **copy.deepcopy(cfg.trainer), weathers=["raw"], dataset=dataset,
        model_config=copy.deepcopy(cfg.model), vgg_config=cfg.get("vgg"), device=torch.device("cuda"),
    )
    trainer.init_gaussians_from_dataset(dataset=dataset)
    if getattr(trainer, "_layered_gbuffer", None) is None:
        raise AssertionError("Persisted trainer.layered_gbuffer was not enabled before optimizer initialization")
    trainer.initialize_optimizer()
    actual = verify_optimizer_membership(trainer)
    report["original_optimizer_groups"] = sorted(actual)
    report["scheduled_optimizer_groups"] = sorted(trainer.lr_schedulers)
    report["required_categories"] = required_categories(actual)
    temporal = TemporalMaterialTraining(dataset, cfg.trainer.losses.get("temporal_material"))
    torch.cuda.synchronize()
    report["full_initialization_seconds"] = time.perf_counter() - initialization_start
    report["initial_gaussian_counts"] = {name: int(count) for name, count in trainer.get_gaussian_count().items()}
    train_indices = list(map(int, dataset.train_indices))
    if not train_indices:
        raise ValueError("No training images")
    positions = [min(len(train_indices) - 1, (index + 1) * len(train_indices) // (args.steps + 1))
                 for index in range(args.steps)]
    for step, position in enumerate(positions, start=args.start_step):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        image_dict, camera_dict = dataset.train_image_set.get_image(position, args.downscale, ["raw"])
        image_infos = packet_to(image_dict["raw"], "cuda")
        camera_infos = packet_to(camera_dict["raw"], "cuda")
        previous = temporal.prepare_previous(trainer, image_infos, args.downscale, "raw", step)
        outputs = trainer(image_infos, camera_infos, "raw")
        trainer.update_visibility_filter()
        losses = trainer.compute_losses(outputs, image_infos, camera_infos, weather="raw")
        losses.update(temporal.compute(trainer, outputs, image_infos, previous, "raw", step))
        if not losses:
            raise AssertionError("Original training loss is empty")
        for name, value in losses.items():
            if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
                raise FloatingPointError(f"Invalid original training loss {name}")
        trainer.backward(losses)
        groups, witnesses = gradient_inventory(trainer.optimizer)
        trainer.optimizer_step()
        update_inventory(groups, witnesses)
        counts_before_refinement = {name: int(count) for name, count in trainer.get_gaussian_count().items()}
        refinement_due = [name for name in trainer.gaussian_classes
                          if step > trainer.models[name].ctrl_cfg.warmup_steps
                          and step % trainer.models[name].ctrl_cfg.refine_interval == 0]
        trainer.postprocess_per_train_step(step=step)
        verify_optimizer_membership(trainer)
        projections = trainer.info.get("layered_gradient_stats")
        if not projections or int(projections["auxiliary_backward_passes"]) <= 0:
            raise AssertionError("Layered raster passes did not reach densification statistics")
        if not math.isfinite(float(projections["auxiliary_grad_norm"])) or float(projections["auxiliary_grad_norm"]) <= 0:
            raise AssertionError("No finite nonzero layered contribution to densification")
        torch.cuda.synchronize()
        record = {"step": step, "image_index": train_indices[position],
                  "image_shape": list(image_infos["pixels"].shape),
                  "seconds": time.perf_counter() - step_start,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "loss": float(sum(losses.values()).detach()),
                  "losses": {name: float(value.detach()) for name, value in losses.items()},
                  "optimizer_groups": groups, "layered_gradient_stats": projections,
                  "refinement_due_nodes": refinement_due,
                  "gaussian_counts_before_postprocess": counts_before_refinement,
                  "gaussian_counts": {name: int(count) for name, count in trainer.get_gaussian_count().items()}}
        report["steps"].append(record)
        atomic_json(args.output, report)
        print(json.dumps({"stage": "joint_training_validation_step", "step": step,
                          "image_index": record["image_index"], "loss": record["loss"],
                          "seconds": record["seconds"], "peak_allocated_gib": record["peak_allocated_gib"]}), flush=True)
        del witnesses, outputs, losses, image_infos, camera_infos, image_dict, camera_dict, previous
    for category, names in report["required_categories"].items():
        records = [step["optimizer_groups"][name] for step in report["steps"] for name in names]
        if not any(group["gradient_max_abs"] > 0 for group in records):
            raise AssertionError(f"No nonzero finite gradient for {category}")
        active = [group for group in records if group["lr"] > 0 and group["gradient_max_abs"] > 0]
        if active and not any(group["updated_witness_count"] > 0 for group in active):
            raise AssertionError(f"No observed scheduled optimizer update for {category}")
    report["status"] = "passed"
    report["completed_steps"] = len(report["steps"])
    report["notes"] = [
        "This probe does not measure validation/test quality or train a final model.",
        "Original delayed learning rates remain delayed; their gradients are checked even when lr=0.",
        "Original postprocess is called; refinement_due_nodes identifies actual refinement callbacks after warmup.",
        "A nonzero start_step tests an artificial schedule position on a fresh untrained model, not prior training.",
        "Step timings include gradient/update validation overhead and are not steady-state throughput.",
    ]
    return report


def main():
    args = make_parser().parse_args()
    args.config_file = args.config_file.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.steps < 1 or args.start_step < 0 or args.seed < 0 or not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("steps must be positive, start-step/seed nonnegative and downscale finite and >=1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {args.output}")
    report = {"schema": 1, "status": "running", "config_file": str(args.config_file)}
    start = time.perf_counter()
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(args.output, report)
        raise
    report["elapsed_seconds"] = time.perf_counter() - start
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "completed_steps": report["completed_steps"]}), flush=True)


if __name__ == "__main__":
    main()
