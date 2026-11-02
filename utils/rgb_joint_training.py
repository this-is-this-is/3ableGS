"""Scratch RGB-joint run validation and optimizer/density/RNG continuation."""
from pathlib import Path

from omegaconf import OmegaConf
import torch

from utils.weather_training import atomic_torch_save, capture_rng, restore_rng, model_state

STATE_KEY = "rgb_joint_training_state"
TRANSIENT_FIELDS = ("xys_grad_norm", "vis_counts", "max_2Dsize", "nn_ind")


def training_signature(cfg):
    value = OmegaConf.to_container(cfg, resolve=True)
    for key in ("log_dir", "logging", "render", "rgb_joint_run"):
        value.pop(key, None)
    return value


def validate_config(cfg):
    from models.rgb_joint_runtime import validate_rgb_joint_config
    if str(cfg.trainer.type) not in ("models.trainers.MultiTrainer_pbr", "models.trainers.scene_graph_pbr.MultiTrainer_pbr"):
        raise ValueError("RGB joint requires the raw Base-v4 PBR trainer")
    if not OmegaConf.select(cfg, "trainer.rgb_joint.enabled", default=False):
        raise ValueError("Enable trainer.rgb_joint for this scratch experiment")
    joint = validate_rgb_joint_config(OmegaConf.to_container(cfg.trainer.rgb_joint, resolve=True))
    from models.dynamic_gbuffer import validate_dynamic_gbuffer_config
    dynamic = validate_dynamic_gbuffer_config(cfg.trainer.get("dynamic_gbuffer"))
    if dynamic["enabled"]:
        for key in ("teacher_calibration", "teacher_confidence", "rdg_fusion"):
            if OmegaConf.select(cfg, f"trainer.{key}.enabled", default=False):
                raise ValueError(f"D1 dynamic_gbuffer cannot also enable {key}")
        if OmegaConf.select(cfg, "trainer.losses.diffreg_material.enabled", default=False):
            raise ValueError("D1 dynamic_gbuffer cannot also enable diffreg_material")
    if cfg.trainer.optim.get("use_grad_scaler", False):
        raise ValueError("Retain Base-v4 use_grad_scaler=false")
    if cfg.trainer.optim.get("cache_buffer_freq", -1) > 0:
        raise ValueError("RGB joint continuation requires the Base-v4 disabled error-map cache")
    if cfg.get("vgg") is not None or OmegaConf.select(cfg, "trainer.losses.temporal_material.enabled", default=False):
        raise ValueError("This experiment retains raw Base-v4 losses without VGG/temporal training hooks")
    if list(cfg.data.pixel_source.load_weather) or not cfg.data.pixel_source.load_gbuffer:
        raise ValueError("RGB joint currently trains raw images with G-buffer supervision")
    for key in ("gbuffer_attention", "layered_gbuffer", "weather_training"):
        if OmegaConf.select(cfg, f"trainer.{key}.enabled", default=False):
            raise ValueError(f"RGB joint experiment cannot also enable {key}")
    for name, model in cfg.model.items():
        if "feature" in model and not OmegaConf.select(model, "rgb_residual.enabled", default=False):
            raise ValueError(f"Missing correction features on {name}")
        if "feature" in model and OmegaConf.select(model, "rgb_residual.feature_dim") != joint["feature_dim"]:
            raise ValueError(f"Correction feature dimension mismatch on {name}")


def validate_inputs(cfg):
    """Validate raw images and train/test teacher files before CUDA setup."""
    pixel = cfg.data.pixel_source
    if list(pixel.cameras) != [0]:
        raise ValueError("This launcher expects FRONT camera 0")
    scene = Path(cfg.data.data_root) / f"{int(cfg.data.scene_idx):03d}"
    teachers = Path(pixel.gbuffer_path) / "raw"
    if int(cfg.data.end_timestep) <= int(cfg.data.start_timestep):
        raise ValueError("RGB joint requires at least two frames")
    missing = []
    count = 0
    for frame in range(int(cfg.data.start_timestep), int(cfg.data.end_timestep) + 1):
        paths = [scene / "images/raw" / f"{frame:03d}_0.jpg"]
        paths += [teachers / f"{int(pixel.get('gbuffer_seq_id', 0)):04d}.{frame:04d}.{kind}.png"
                  for kind in ("normal", "basecolor", "roughness", "metallic")]
        for path in paths:
            count += 1
            if not path.is_file() or not path.stat().st_size:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} raw/teacher files: {missing[:8]}")
    return {"files": count, "scene": int(cfg.data.scene_idx),
            "frames": [int(cfg.data.start_timestep), int(cfg.data.end_timestep)]}


def save_training_state(trainer, dataset, cfg, path):
    state = trainer.state_dict(only_model=True)
    density = {}
    for name in trainer.gaussian_classes:
        model = trainer.models[name]
        density[name] = {key: (getattr(model, key).detach().cpu().clone()
                              if isinstance(getattr(model, key), torch.Tensor) else getattr(model, key))
                         for key in TRANSIENT_FIELDS if hasattr(model, key)}
    state[STATE_KEY] = {
        "version": 1, "completed_step": int(trainer.step), "config": training_signature(cfg),
        "optimizer": trainer.optimizer.state_dict(),
        "group_names": [group["name"] for group in trainer.optimizer.param_groups],
        "grad_scaler": trainer.grad_scaler.state_dict(), "density": density,
        "rng": capture_rng(), "train_split_indices": list(dataset.train_image_set.split_indices),
    }
    atomic_torch_save(state, path)


def read_training_state(path, cfg):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    resume = state.get(STATE_KEY)
    if not isinstance(resume, dict) or resume.get("version") != 1:
        raise ValueError("Resume requires this experiment's complete optimizer/density/RNG checkpoint")
    step = state.get("step")
    if type(step) is not int or step < 0 or resume.get("completed_step") != step:
        raise ValueError("Invalid completed step in RGB joint checkpoint")
    if resume.get("config") != training_signature(cfg):
        raise ValueError("Resume fitting settings differ from saved checkpoint; use a new RUN_TAG")
    count = state.get("rgb_joint_camera_updates")
    if not isinstance(count, torch.Tensor) or count.numel() != 1 or int(count) != step + 1:
        raise ValueError("Completed camera counter must equal step + 1 for raw camera-only training")
    return state


def load_model_state(trainer, state):
    # Node loaders assign non-buffer tensors, so move the full model tree.
    tensors = model_state({key: value for key, value in state.items() if key != STATE_KEY}, trainer.device)
    trainer.load_state_dict(tensors, load_only_model=True, strict=True)


def restore_training_state(trainer, dataset, state, cfg):
    resume = state[STATE_KEY]
    if resume["config"] != training_signature(cfg):
        raise ValueError("Saved fitting settings do not match current configuration")
    if list(dataset.train_image_set.split_indices) != resume["train_split_indices"]:
        raise ValueError("Training split changed since checkpoint")
    if [g["name"] for g in trainer.optimizer.param_groups] != resume["group_names"]:
        raise ValueError("Optimizer group order changed since checkpoint")
    if set(trainer.gaussian_classes) != set(resume["density"]):
        raise ValueError("Gaussian population names changed since checkpoint")
    trainer.optimizer.load_state_dict(resume["optimizer"])
    trainer.grad_scaler.load_state_dict(resume["grad_scaler"])
    for name, values in resume["density"].items():
        model = trainer.models[name]
        for key, value in values.items():
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Invalid density state {name}.{key}")
                if key != "nn_ind" and value.shape != (model.num_points,):
                    raise ValueError(f"Density row mismatch: {name}.{key}")
                value = value.to(trainer.device)
            setattr(model, key, value)
    # LR schedules are pure functions of the original config and global step.
    # Restore random streams last, after construction/load/optimizer setup.
    restore_rng(resume["rng"])
    return int(resume["completed_step"]) + 1
