#!/usr/bin/env python3
"""Run Rain then Snow, independently initialized from one completed raw base.

Without --execute, prints the six stage commands and checks input paths only.
Completed/evaluated stages are skipped; interrupted stages resume Adam/RNG.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.weather_training import exclusive_run_lock, load_weather_config, validate_weather_inputs


def stage_command(args, weather, stage, number, source):
    run_dir = Path(args.output_dir).resolve() / f"{number:02d}_{weather}_{stage}"
    final = run_dir / "checkpoint_final.pth"
    latest = run_dir / "checkpoint_training_latest.pth"
    marker = run_dir / "stage_status.json"
    if final.exists() or latest.exists():
        saved = OmegaConf.load(run_dir / "config.yaml")
        saved_raw = Path(saved.weather_run.raw_checkpoint).resolve()
        if (saved_raw != Path(args.base_ckpt).resolve()
                or saved.trainer.weather_training.weather != weather
                or saved.trainer.weather_training.stage != stage
                or Path(saved.weather_run.source_checkpoint).resolve() != Path(source).resolve()):
            raise ValueError(f"Existing run belongs to a different base/weather/stage: {run_dir}")
    prefix = [sys.executable, "-u", str(ROOT / "tools/train_weather.py"),
              "--config_file", str(Path(args.config_file).resolve()), "--run_dir", str(run_dir),
              "--weather", weather, "--weather_stage", stage]
    if final.exists():
        status = json.loads(marker.read_text()) if marker.exists() else {}
        if status.get("evaluation_done") and status.get("status") == "completed":
            return run_dir, None
        return run_dir, prefix + ["--resume_from", str(final), "--evaluate"]
    if latest.exists():
        return run_dir, prefix + ["--resume_from", str(latest)]
    return run_dir, prefix + ["--init_from_base", str(source)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ckpt", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config_file", default=ROOT / "configs/base_v4/weather.yaml", type=Path)
    parser.add_argument("--weathers", nargs="+", choices=("rain", "snow"), default=["rain", "snow"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.weathers)) != len(args.weathers):
        parser.error("Duplicate weather names are not allowed")
    args.base_ckpt = args.base_ckpt.resolve()
    if not args.base_ckpt.is_file():
        raise FileNotFoundError(args.base_ckpt)
    for weather in args.weathers:
        cfg = load_weather_config(args.config_file, args.base_ckpt, weather, "material")
        print("Inputs:", validate_weather_inputs(cfg), flush=True)
    def run_queue():
        for weather in args.weathers:
            source = args.base_ckpt
            offset = 0 if weather == "rain" else 3
            for index, stage in enumerate(("material", "environment", "joint"), 1):
                run_dir, command = stage_command(args, weather, stage, offset + index, source)
                if command is None:
                    print(f"Already completed: {run_dir}", flush=True)
                else:
                    print(shlex.join(command), flush=True)
                    if args.execute:
                        subprocess.run(command, cwd=ROOT, check=True)
                source = run_dir / "checkpoint_final.pth"
        print("Rain/Snow queue completed." if args.execute else "Preview only; add --execute to train.", flush=True)
    if args.execute:
        with exclusive_run_lock(args.output_dir.resolve() / ".weather_queue.lock"):
            run_queue()
    else:
        run_queue()


if __name__ == "__main__":
    main()
