#!/usr/bin/env python3
"""Preview or explicitly execute matched scratch Base v5 / D1 experiments.

The default is read-only: no child process or output directory is created.
--prepare-only delegates input/resume checks to Base v5 without training.
Only --execute starts training. Each arm uses the native exact-resume launcher,
the original heldout metrics, and identical additional dynamic diagnostics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("B0", "D1B", "D1P")
CONFIG_DIR = ROOT / "configs/dynamic_gbuffer"
DEFAULT_OUTPUT = ROOT / "output/pbr/dynamic_gbuffer/scene788_from_scratch_v1"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=ROOT / ".conda/bin/python")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--iterations", type=int, default=30001,
                        help="TOTAL camera updates, not extra updates; native schedules stay at 30k")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dynamic-temporal", action="store_true",
                        help="Also render chronological train+test frames for a labeled reconstruction diagnostic")
    parser.add_argument("--save-images", action="store_true",
                        help="Opt in to paired PNG/float NPZ exports; disabled to limit disk usage")
    return parser


def fitting_signature(config):
    """Match utils.rgb_joint_training without importing PyTorch for previews."""
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(config, resolve=True)
    for key in ("log_dir", "logging", "render", "rgb_joint_run"):
        value.pop(key, None)
    return value


def file_identity(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return deepcopy(default)
    return json.loads(path.read_text())


def resolved_config(variant, seed):
    from utils.config import load_training_config
    return load_training_config(CONFIG_DIR / f"{variant}.yaml", [f"seed={seed}"])


def verify_saved_fitting(arm):
    from omegaconf import OmegaConf
    saved = Path(arm["run_dir"]) / "config.yaml"
    if saved.is_file() and fitting_signature(OmegaConf.load(saved)) != arm["fitting_signature"]:
        raise ValueError(f"Saved fitting settings differ for {arm['id']}; choose a new --output-dir")
    status = read_json(Path(arm["run_dir"]) / "training_status.json", {})
    completed = status.get("completed_updates", -1)
    if type(completed) is not int:
        raise ValueError(f"Invalid completed update count for {arm['id']}")
    if completed > arm["total_camera_updates"]:
        raise ValueError(f"Saved {arm['id']} has {completed} updates, above requested "
                         f"{arm['total_camera_updates']}; use a new --output-dir for a smaller budget")


def build_plan(args):
    if type(args.iterations) is not int or args.iterations < 1:
        raise ValueError("--iterations must be a positive TOTAL update count")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct nonnegative integers")
    if not args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must be distinct")
    if args.skip_eval and (args.dynamic_temporal or args.save_images):
        raise ValueError("--dynamic-temporal/--save-images require evaluation")
    output = args.output_dir.expanduser().resolve()
    python = args.python.expanduser().absolute()
    arms = []
    for seed in args.seeds:
        for variant in args.variants:
            config = resolved_config(variant, seed)
            run_dir = output / f"seed_{seed}" / variant
            train = [str(python), "-u", str(ROOT / "tools/train_base_v5.py"),
                     "--config-file", str(CONFIG_DIR / f"{variant}.yaml"),
                     "--run-dir", str(run_dir), "--iterations", str(args.iterations),
                     "--skip-eval", f"seed={seed}"]
            evaluate = [str(python), "-u", str(ROOT / "tools/eval_base_v5.py"),
                        "--checkpoint", str(run_dir / "checkpoint_final.pth"), "--dynamic-metrics"]
            if args.dynamic_temporal:
                evaluate.append("--dynamic-temporal")
            if args.save_images:
                evaluate.append("--save-images")
            signature = fitting_signature(config)
            arm = {"id": f"{variant}_s{seed}", "variant": variant, "seed": seed,
                   "run_dir": str(run_dir), "total_camera_updates": args.iterations,
                   "fitting_signature": signature,
                   "fitting_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                   "train_command": train, "eval_command": None if args.skip_eval else evaluate,
                   "evaluation_options": {"dynamic_metrics": True, "dynamic_temporal": bool(args.dynamic_temporal),
                                          "save_images": bool(args.save_images)}}
            verify_saved_fitting(arm)
            arms.append(arm)
    return {"schema_version": 1, "output_dir": str(output), "arms": arms,
            "initialization": "scratch_dataset_seed_points; completed runs resume their own exact state",
            "protocol": {"scene": 788, "camera": 0, "source_frames": [0, 99],
                         "train_count": 88, "test_ids": list(range(8, 100, 8)),
                         "updates_per_arm": args.iterations, "native_schedule_final_step": 30000,
                         "teacher_data": "unchanged Base v5 single_static teachers",
                         "checkpoint_selection": "final requested update; no test-based selection",
                         "temporal_diagnostic": "train-inclusive reconstruction, not heldout video"}}


def preview(plan):
    return {"mode": "preview", "output_dir": plan["output_dir"], "protocol": plan["protocol"],
            "commands": {arm["id"]: {"train": shlex.join(arm["train_command"]),
                                      "evaluate": shlex.join(arm["eval_command"]) if arm["eval_command"] else None}
                         for arm in plan["arms"]}}


def run_child(command, log_path=None):
    """Own the child's process group so interrupting the suite stops that child."""
    import contextlib
    output = contextlib.nullcontext(None) if log_path is None else Path(log_path).open("a")
    with output as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait()
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def completed_training(arm, previous):
    directory = Path(arm["run_dir"])
    checkpoint = directory / "checkpoint_final.pth"
    status = read_json(directory / "training_status.json", {})
    return bool(previous.get("train_state") == "completed"
                and previous.get("fitting_sha256") == arm["fitting_sha256"]
                and checkpoint.is_file() and checkpoint.stat().st_size
                and previous.get("training_checkpoint") == file_identity(checkpoint)
                and previous.get("completed_updates") == status.get("completed_updates")
                and int(status.get("completed_updates", -1)) == arm["total_camera_updates"])


def training_artifacts(arm):
    directory = Path(arm["run_dir"])
    verify_saved_fitting(arm)
    if not (directory / "config.yaml").is_file():
        raise RuntimeError(f"Training omitted config.yaml: {arm['id']}")
    checkpoint = directory / "checkpoint_final.pth"
    status = read_json(directory / "training_status.json", {})
    completed = status.get("completed_updates", -1)
    if (type(completed) is not int or completed != arm["total_camera_updates"]
            or not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        raise RuntimeError(f"Training did not produce the requested final state: {arm['id']}")
    result = {"completed_updates": completed, "training_checkpoint": file_identity(checkpoint)}
    process_stats = {key: status[key] for key in (
        "process_start_update", "loop_elapsed_seconds_this_process",
        "peak_allocated_vram_bytes_this_process",
    ) if key in status}
    if process_stats:
        result["last_training_process"] = process_stats
    return result


def evaluation_request(arm):
    return {"checkpoint": file_identity(Path(arm["run_dir"]) / "checkpoint_final.pth"),
            "fitting_sha256": arm["fitting_sha256"], "options": arm["evaluation_options"]}


def evaluation_artifacts(arm):
    directory = Path(arm["run_dir"])
    result = read_json(directory / "rgb_joint_eval/evaluation.json")
    if not isinstance(result, dict):
        raise RuntimeError(f"Evaluation omitted evaluation.json: {arm['id']}")
    if (Path(result["checkpoint"]).resolve() != directory / "checkpoint_final.pth"
            or int(result["training_budget"]["completed_camera_updates"]) != arm["total_camera_updates"]):
        raise RuntimeError(f"Evaluation checkpoint/budget mismatch: {arm['id']}")
    if not result.get("dynamic_object_metrics"):
        raise RuntimeError(f"Evaluation omitted requested dynamic object metrics: {arm['id']}")
    if arm["evaluation_options"]["dynamic_temporal"] and not result.get("dynamic_temporal_diagnostic"):
        raise RuntimeError(f"Evaluation omitted requested dynamic temporal diagnostic: {arm['id']}")
    if bool(result.get("images_saved")) != arm["evaluation_options"]["save_images"]:
        raise RuntimeError(f"Evaluation image-export option mismatch: {arm['id']}")
    return {key: result[key] for key in (
        "image_metrics", "pure_pbr_image_metrics", "dynamic_object_metrics",
        "dynamic_temporal_diagnostic", "elapsed_seconds", "smoke",
    ) if key in result}


def write_report(summary, path):
    rows = ["# Base v5 D1 실험", "", f"상태: {summary['status']}", "",
            "| Arm | 상태 | 완료 업데이트 | Hybrid PSNR | SSIM | LPIPS |",
            "|---|---|---:|---:|---:|---:|"]
    for name, row in sorted(summary["arms"].items()):
        metrics = row.get("evaluation", {}).get("image_metrics", {})
        numbers = [f"{metrics[key]:.5f}" if key in metrics else "—" for key in ("psnr", "ssim", "lpips")]
        rows.append(f"| {name} | {row.get('status', 'pending')} | {row.get('completed_updates', '—')} | "
                    + " | ".join(numbers) + " |")
    rows += ["", "B0는 원래 Base v5, D1B는 객체 균형 loss, D1P는 객체 균형+purity loss입니다.",
             "각 arm은 처음부터 별도로 학습하며 같은 arm의 중단 상태만 이어갑니다. "
             "업데이트 예산을 줄인 결과는 최종 30,001회 실험과 구분해야 합니다.", "",
             "Pure PBR는 같은 공동 학습 모델에서 display residual만 끈 결과입니다. "
             "동적 영역 지표와 reference 연결 성분 crop LPIPS는 각 arm의 rgb_joint_eval/evaluation.json에 있습니다. "
             "연결 성분은 실제 객체/track ID가 아닙니다.", "",
             "선택적 temporal 지표는 학습 프레임을 포함한 재구성 진단입니다. "
             "기존 12장 test split 자체가 연속 heldout 영상은 아닙니다.", "",
             "summary.json의 train_attempts 시간은 모델 준비·resume·checkpoint IO를 포함한 child-process 벽시계 시간입니다. "
             "중단된 실행과 여러 번 재개한 실행의 시간을 단일 uninterrupted 학습시간처럼 비교하지 마세요."]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_suite(args, plan, *, child_runner=run_child):
    if not args.execute:
        result = preview(plan)
        result["mode"] = "prepare-only" if args.prepare_only else "preview"
        print(json.dumps(result, indent=2), flush=True)
        if args.prepare_only:
            for arm in plan["arms"]:
                command = list(arm["train_command"])
                command.insert(-1, "--prepare-only")
                child_runner(command, None)
        return result

    from utils.weather_training import atomic_json_save, exclusive_run_lock
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(output / ".suite.lock"):
        summary = read_json(output / "summary.json", {"schema_version": 1, "arms": {}})
        summary.update(status="running", protocol=plan["protocol"],
                       requested_arms=[arm["id"] for arm in plan["arms"]])

        def save():
            atomic_json_save(summary, output / "summary.json")
            write_report(summary, output / "report.md")

        atomic_json_save(plan, output / "plan.json")
        save()
        try:
            for arm in plan["arms"]:
                row = summary["arms"].setdefault(arm["id"], {"status": "pending", "train_attempts": [], "eval_attempts": []})
                verify_saved_fitting(arm)
                if row.get("fitting_sha256", arm["fitting_sha256"]) != arm["fitting_sha256"]:
                    raise ValueError(f"Stored suite fitting settings differ for {arm['id']}; choose a new --output-dir")
                row.update(variant=arm["variant"], seed=arm["seed"], run_dir=arm["run_dir"],
                           fitting_sha256=arm["fitting_sha256"], total_camera_updates=arm["total_camera_updates"])
                row.pop("error", None)
                if not completed_training(arm, row):
                    row.update(status="training", train_state="running", eval_state="pending")
                    row.pop("evaluation", None)
                    log = output / f"{arm['id']}_train.log"
                    attempt = {"started_at_unix": time.time(), "command": arm["train_command"], "log": str(log)}
                    row["train_attempts"].append(attempt)
                    save()
                    started = time.monotonic()
                    try:
                        child_runner(arm["train_command"], log)
                        row.update(training_artifacts(arm))
                        row["train_state"] = "completed"
                        attempt["status"] = "completed"
                    except BaseException:
                        attempt["status"] = "failed_or_interrupted"
                        row["train_state"] = "incomplete"
                        raise
                    finally:
                        attempt["elapsed_seconds"] = time.monotonic() - started
                        save()

                if arm["eval_command"] is None:
                    row.update(status="trained_evaluation_skipped", eval_state="skipped")
                    save()
                    continue
                request = evaluation_request(arm)
                request_path = Path(arm["run_dir"]) / "d1_evaluation_request.json"
                cached = read_json(request_path) == request
                if cached:
                    try:
                        row["evaluation"] = evaluation_artifacts(arm)
                    except (KeyError, TypeError, ValueError, RuntimeError):
                        cached = False
                if not cached:
                    row.update(status="evaluating", eval_state="running")
                    row.pop("evaluation", None)
                    log = output / f"{arm['id']}_eval.log"
                    attempt = {"started_at_unix": time.time(), "command": arm["eval_command"], "log": str(log)}
                    row["eval_attempts"].append(attempt)
                    save()
                    started = time.monotonic()
                    try:
                        child_runner(arm["eval_command"], log)
                        row["evaluation"] = evaluation_artifacts(arm)
                        atomic_json_save(request, request_path)
                        attempt["status"] = "completed"
                    except BaseException:
                        attempt["status"] = "failed_or_interrupted"
                        row["eval_state"] = "incomplete"
                        raise
                    finally:
                        attempt["elapsed_seconds"] = time.monotonic() - started
                        save()
                row.update(status="completed", eval_state="completed")
                save()
        except BaseException as exc:
            summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if "row" in locals():
                row.update(status=summary["status"], error=str(exc))
            save()
            raise
        summary["requested_arms_status"] = "completed_without_evaluation" if args.skip_eval else "completed"
        states = {row.get("status", "pending") for row in summary["arms"].values()}
        summary["status"] = ("completed" if states == {"completed"} else
                             "completed_without_evaluation" if states <= {"completed", "trained_evaluation_skipped"}
                             else "partial")
        save()
        return summary


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.status:
        saved = args.output_dir.expanduser().resolve() / "summary.json"
        if not saved.is_file():
            parser.error(f"No suite status: {saved}")
        print(saved.read_text(), end="")
        return

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Suite interrupted; repeat the same --execute command to resume")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_suite(args, build_plan(args))
        if args.execute:
            print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, indent=2))
    except KeyboardInterrupt as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
