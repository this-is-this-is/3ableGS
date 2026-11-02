#!/usr/bin/env python3
"""Record the local rebuild environment without installing or downloading anything.

The optional CUDA smoke test renders two Gaussians with eight material channels
and depth, then checks backward/AbsGrad. It does not load data or train a model.
gsplat may compile its installed extension into the repository-local cache.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "gsplat": "gsplat",
    "nvdiffrast": "nvdiffrast.torch",
    "lpips": "lpips",
    "torchmetrics": "torchmetrics.image.lpip",
    "omegaconf": "omegaconf",
    "Pillow": "PIL.Image",
    "smplx": "smplx",
    "transformers": "transformers",
}


def run_command(argv, timeout=20):
    try:
        completed = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
        return {
            "command": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "returncode": None, "error": str(exc)}


def inspect_packages():
    packages = {}
    for distribution, module_name in PACKAGES.items():
        info = {"module": module_name}
        try:
            info["version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            info["version"] = None
        try:
            module = importlib.import_module(module_name)
            info["import_ok"] = True
            info["import_path"] = getattr(module, "__file__", None)
        except Exception as exc:
            info["import_ok"] = False
            info["error"] = f"{type(exc).__name__}: {exc}"
        packages[distribution] = info
    # Importing LPIPS is sufficient here: constructing pretrained networks may
    # download weights and is deliberately left for the model implementation.
    return packages


def cuda_info():
    import torch

    available = torch.cuda.is_available()
    info = {
        "available": available,
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if available:
        for device_id in range(info["device_count"]):
            props = torch.cuda.get_device_properties(device_id)
            info["devices"].append({
                "id": device_id,
                "name": props.name,
                "compute_capability": list(torch.cuda.get_device_capability(device_id)),
                "total_memory_bytes": props.total_memory,
            })
    return info


def cuda_smoke():
    import torch
    from gsplat import rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the requested GPU smoke test cannot pass.")
    started = time.monotonic()
    device = torch.device("cuda:0")

    def leaf(values):
        return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)

    means = leaf([[-0.2, 0.05, 2.0], [0.3, -0.15, 2.5]])
    quats = leaf([[1.0, 0.0, 0.1, 0.0], [1.0, 0.1, 0.0, 0.0]])
    scales = leaf([[0.12, 0.18, 0.08], [0.15, 0.1, 0.09]])
    opacities = leaf([0.7, 0.6])
    materials = leaf([
        [0.6, 0.3, 0.2, 0.1, 0.5, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.6, 0.2, 0.7, 0.0, 0.0, 1.0],
    ])
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    render, alpha, meta = rasterization(
        means, quats, scales, opacities, materials, viewmats, intrinsics,
        width=32, height=32, packed=False, absgrad=True,
        rasterize_mode="classic", render_mode="RGB+ED",
    )
    if tuple(render.shape) != (1, 32, 32, 9):
        raise RuntimeError(f"Expected eight material channels plus depth, got {render.shape}.")
    if tuple(alpha.shape) != (1, 32, 32, 1):
        raise RuntimeError(f"Unexpected opacity shape: {alpha.shape}.")
    if not torch.isfinite(render).all() or not torch.isfinite(alpha).all():
        raise RuntimeError("Non-finite rasterization output.")
    if not (alpha.max() > 0):
        raise RuntimeError("The smoke test did not rasterize any visible Gaussian.")
    loss = render.square().mean() + alpha.mean()
    loss.backward()
    gradients = {}
    for name, value in {
        "means": means, "quats": quats, "scales": scales,
        "opacities": opacities, "materials": materials,
    }.items():
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError(f"Missing/non-finite gradient: {name}.")
        gradients[name] = float(value.grad.abs().sum().detach().cpu())
    if any(value == 0 for value in gradients.values()):
        raise RuntimeError(f"Unexpected zero gradient in smoke test: {gradients}.")
    absgrad = getattr(meta["means2d"], "absgrad", None)
    if absgrad is None or not torch.isfinite(absgrad).all() or not (absgrad.sum() > 0):
        raise RuntimeError("Missing/non-finite/zero projected-means absolute gradients.")
    torch.cuda.synchronize(device)
    return {
        "status": "passed", "elapsed_seconds": round(time.monotonic() - started, 3),
        "render_shape": list(render.shape), "alpha_shape": list(alpha.shape),
        "gradient_absolute_sums": gradients,
        "means2d_absgrad_sum": float(absgrad.sum().detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/reconstruction/phase1_environment.json")
    parser.add_argument("--cuda-smoke", action="store_true", help="Require CUDA and verify gsplat forward/backward.")
    args = parser.parse_args()
    # Keep any cache produced by imports or the explicit JIT smoke test local.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache/torch_extensions"))
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "git": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short"]),
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "packages": inspect_packages(),
        "cuda_smoke": {"status": "not_requested"},
        "errors": [],
        "notes": ["No pretrained weights were instantiated or downloaded.",
                  "CUDA visibility is recorded; without --cuda-smoke its absence is not an import failure."],
    }
    for name, info in report["packages"].items():
        if not info["import_ok"]:
            report["errors"].append(f"{name}: {info['error']}")
    smplx_path = report["packages"]["smplx"].get("import_path")
    if smplx_path and not Path(smplx_path).resolve().is_relative_to(ROOT / "third_party/smplx"):
        report["errors"].append(f"smplx resolves outside this repository: {smplx_path}")
    try:
        report["cuda"] = cuda_info()
    except Exception as exc:
        report["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        report["errors"].append(f"CUDA inspection: {exc}")
    if args.cuda_smoke:
        try:
            report["cuda_smoke"] = cuda_smoke()
        except Exception as exc:
            report["cuda_smoke"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(f"CUDA smoke: {exc}")
    report["status"] = "failed" if report["errors"] else "passed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out.resolve()),
                      "cuda_available": report["cuda"]["available"],
                      "cuda_smoke": report["cuda_smoke"]["status"],
                      "errors": report["errors"]}, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
