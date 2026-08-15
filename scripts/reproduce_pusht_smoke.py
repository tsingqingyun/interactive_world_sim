#!/usr/bin/env python3
"""Run a deterministic PushT checkpoint smoke test without dataset input."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(checkpoint_path: Path, device: torch.device) -> LatentWorldModel:
    config_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    config.algorithm.load_ae = None
    config.algorithm.metrics = []
    config.algorithm.n_frames = 10

    model = LatentWorldModel(config.algorithm)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if not key.startswith("validation_fvd_model.")
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def make_synthetic_image(resolution: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    x = axis[None, :].expand(resolution, resolution)
    y = axis[:, None].expand(resolution, resolution)
    image = torch.stack((x, y, 0.5 * (x + y)), dim=0)
    return image.unsqueeze(0)


def tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


def save_rgb_tensor(path: Path, tensor: torch.Tensor) -> None:
    image = tensor.detach().float().cpu().clamp(0, 1)[0]
    image = image.permute(1, 2, 0).numpy()
    image = np.rint(image * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pusht_cam1/checkpoints/best.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction_artifacts/pusht_synthetic_smoke"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = load_model(args.checkpoint, device)
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    resolution = 128
    obs_key = model.obs_keys[0]
    input_image = make_synthetic_image(resolution, device)
    normalized_image = model.normalizer[obs_key].normalize(input_image)

    encode_started = time.perf_counter()
    with torch.inference_mode():
        latent = model.encoder_forward(normalized_image)[:, None]
    synchronize(device)
    encode_seconds = time.perf_counter() - encode_started

    action = torch.zeros((1, 2, model.cfg.action_dim), device=device)
    dynamics_started = time.perf_counter()
    with torch.inference_mode():
        predicted_latent = model.dynamics_forward(latent, action)
    synchronize(device)
    dynamics_seconds = time.perf_counter() - dynamics_started

    decode_started = time.perf_counter()
    with torch.inference_mode():
        predicted_image = render_img_cm(
            model,
            predicted_latent[:, -1],
            resolution,
            normalizer=model.normalizer,
            num_views=len(model.obs_keys),
        )
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_started

    input_path = args.output_dir / "synthetic_input.png"
    prediction_path = args.output_dir / "predicted_next_frame.png"
    save_rgb_tensor(input_path, input_image)
    save_rgb_tensor(prediction_path, predicted_image)

    report = {
        "evidence_class": "synthetic checkpoint smoke test; not real dataset evidence",
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "obs_keys": list(model.obs_keys),
        "action": tensor_stats(action),
        "input": tensor_stats(input_image),
        "latent": tensor_stats(latent),
        "predicted_latent": tensor_stats(predicted_latent),
        "prediction": tensor_stats(predicted_image),
        "timings_seconds": {
            "load": load_seconds,
            "encode": encode_seconds,
            "dynamics": dynamics_seconds,
            "decode": decode_seconds,
        },
        "artifacts": {
            "input_png": str(input_path),
            "input_png_sha256": sha256(input_path),
            "prediction_png": str(prediction_path),
            "prediction_png_sha256": sha256(prediction_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    OmegaConf.register_new_resolver("torch", lambda name: getattr(torch, name))
    main()
