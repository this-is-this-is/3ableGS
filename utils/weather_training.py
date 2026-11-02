"""Checkpoint-inherited weather stages and complete same-stage continuation.

Stage creation inherits the source model, but resets the optimizer and step.
Continuation uses the saved resolved config, optimizer and all sampling RNGs.
The raw trainer's older model-only checkpoint contract is left unchanged.
"""
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from utils.config import load_config_with_base, load_training_config


STAGES = {
    "material": {"iterations": 30000, "material_lr": 1e-4, "environment_lr": 1e-3, "save_every": 5000},
    "environment": {"iterations": 3000, "material_lr": 1e-4, "environment_lr": 1e-3, "save_every": 1000},
    "joint": {"iterations": 5000, "material_lr": 5e-5, "environment_lr": 5e-4, "save_every": 1000},
}
STATE_KEY = "weather_optimizer_state"


def training_signature(config):
    """Fields affecting fitting; output paths and evaluation/logging are independent."""
    result = OmegaConf.to_container(config, resolve=True)
    for name in ("log_dir", "logging", "render", "weather_run"):
        result.pop(name, None)
    return result


def load_weather_config(config_file, checkpoint, weather=None, stage=None, overrides=(), *, resume=False):
    source_path = Path(checkpoint).resolve()
    saved = load_training_config(source_path.parent / "config.yaml")
    source_weather = OmegaConf.select(saved, "trainer.weather_training.weather")
    source_is_weather = bool(OmegaConf.select(saved, "trainer.weather_training.enabled", default=False))
    if resume:
        if not source_is_weather:
            raise ValueError("Weather resume requires a weather checkpoint; use --init_from_base for raw")
        cfg = copy.deepcopy(saved)
    else:
        overlay = load_config_with_base(config_file)
        if not overlay.pop("base_config_from_checkpoint", False):
            raise ValueError("Weather overlay must declare base_config_from_checkpoint: true")
        cfg = OmegaConf.merge(saved, overlay)
        if source_is_weather:
            # Never reset a trained decoder's architecture/caps or target options
            # to template defaults when entering the next stage.
            cfg.trainer.weather_training = copy.deepcopy(saved.trainer.weather_training)
        weather = weather or source_weather or cfg.trainer.weather_training.weather
        stage = stage or "material"
        if source_is_weather and weather != source_weather:
            raise ValueError("Rain and Snow must start independently from the raw checkpoint")
        if weather not in ("rain", "snow") or stage not in STAGES:
            raise ValueError("Expected rain/snow and material/environment/joint")
        expected_previous = {"environment": "material", "joint": "environment"}
        if stage in expected_previous:
            if not source_is_weather or saved.trainer.weather_training.stage != expected_previous[stage]:
                raise ValueError(f"{stage} must initialize from the completed {expected_previous[stage]} stage")
            if source_path.name != "checkpoint_final.pth":
                raise ValueError("Stage transitions require checkpoint_final.pth")
        elif source_is_weather:
            raise ValueError("A new material stage must initialize from raw, not another weather stage")
        options = STAGES[stage]
        cfg.trainer.weather_training.weather = weather
        cfg.trainer.weather_training.stage = stage
        cfg.trainer.weather_training.material_lr = options["material_lr"]
        cfg.trainer.weather_training.environment_lr = options["environment_lr"]
        cfg.trainer.optim.num_iters = options["iterations"]
        if stage in ("environment", "joint"):
            cfg.trainer.res_schedule.downscale_times = 2
            cfg.trainer.res_schedule.double_steps = 250
        cfg.logging.saveckpt_freq = options["save_every"]
        cfg.data.pixel_source.load_weather = [weather]
        cfg.model.EnvLight.params.weather_residuals = [weather]
        cfg.weather_run = {
            "source_checkpoint": str(source_path),
            "raw_checkpoint": str(OmegaConf.select(saved, "weather_run.raw_checkpoint", default=str(source_path))),
        }
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    actual_weather = cfg.trainer.weather_training.weather
    actual_stage = cfg.trainer.weather_training.stage
    if weather is not None and weather != actual_weather:
        raise ValueError("Requested weather differs from the saved/resolved config")
    if stage is not None and stage != actual_stage:
        raise ValueError("Requested stage differs from the saved/resolved config")
    if resume and training_signature(cfg) != training_signature(saved):
        raise ValueError("Same-stage resume cannot change fitting settings; use the saved config")
    if int(cfg.trainer.optim.num_iters) < 1:
        raise ValueError("num_iters must be positive")
    if cfg.trainer.optim.get("use_grad_scaler", False):
        raise ValueError("Historical weather stages use use_grad_scaler=false")
    if cfg.trainer.optim.get("cache_buffer_freq", -1) > 0:
        raise ValueError("Weather continuation requires the historical disabled error-map cache")
    if list(cfg.data.pixel_source.load_weather) != [actual_weather]:
        raise ValueError("Data must contain raw plus exactly the selected paired weather")
    if list(cfg.model.EnvLight.params.weather_residuals) != [actual_weather]:
        raise ValueError("EnvLight residual must match the selected weather")
    return cfg


def validate_weather_inputs(config):
    """Fail before GPU setup if any paired RGB/teacher pass is missing."""
    data = config.data
    pixel = data.pixel_source
    if list(pixel.cameras) != [0] or not pixel.load_gbuffer:
        raise ValueError("Restored paired weather training requires camera 0 G-buffers")
    scene = Path(data.data_root) / f"{int(data.scene_idx):03d}"
    teacher = Path(pixel.gbuffer_path)
    frames = range(int(data.start_timestep), int(data.end_timestep) + 1)
    if not frames:
        raise ValueError("Empty frame interval")
    missing, count = [], 0
    for weather in ("raw", config.trainer.weather_training.weather):
        for frame in frames:
            paths = [scene / "images" / weather / f"{frame:03d}_0.jpg"]
            paths += [teacher / weather / f"{int(pixel.get('gbuffer_seq_id', 0)):04d}.{frame:04d}.{kind}.png"
                      for kind in ("normal", "basecolor", "roughness", "metallic")]
            for path in paths:
                count += 1
                if not path.is_file() or not path.stat().st_size:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} paired inputs: {missing[:12]}")
    return {"scene": int(data.scene_idx), "frames": len(frames), "files": count,
            "weather": config.trainer.weather_training.weather}


@contextlib.contextmanager
def exclusive_run_lock(path):
    """An OS lock prevents duplicate launches, including after a stale PID file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already using {path.parent}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_torch_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temp.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved training state")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_training_state(trainer, config, path):
    state = trainer.state_dict(only_model=True)
    state[STATE_KEY] = {
        "format_version": 1,
        "completed_step": int(trainer.step),
        "config": training_signature(config),
        "optimizer": trainer.optimizer.state_dict(),
        "group_names": [g["name"] for g in trainer.optimizer.param_groups],
        "rng": capture_rng(),
    }
    atomic_torch_save(state, path)


def model_state(checkpoint, device=None):
    state = {key: value for key, value in checkpoint.items() if key != STATE_KEY}
    if device is None:
        return state

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            result = copy.copy(value)
            for key, item in value.items():
                result[key] = move(item)
            return result
        if isinstance(value, (list, tuple)):
            return type(value)(move(item) for item in value)
        return value
    # Rigid/SMPL load_state_dict assigns several non-buffer tensors directly.
    # These must be on CUDA too; optimizer step counters and RNG remain on CPU.
    return move(state)


def restore_training_state(trainer, checkpoint, config):
    state = checkpoint.get(STATE_KEY)
    if state is None or state.get("format_version") != 1:
        raise ValueError("Resume needs checkpoint_training_latest.pth with optimizer/RNG state")
    if state["config"] != training_signature(config):
        raise ValueError("Checkpoint training configuration does not match the run")
    if int(trainer.step) != state["completed_step"]:
        raise ValueError("Checkpoint model step and training step disagree")
    if state["group_names"] != [g["name"] for g in trainer.optimizer.param_groups]:
        raise ValueError("Optimizer groups differ from the saved stage")
    trainer.optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return int(state["completed_step"]) + 1
