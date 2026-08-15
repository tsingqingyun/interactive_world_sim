"""Compare paired RGB-only and RGB+keypoint-marker rollout exports.

Each NPZ must contain identically shaped `predicted_rgb`, `target_rgb`,
`contact_force`, and `marker_mask` arrays. RGB arrays are uint8 E,T,H,W,3;
contact force is E,T,2 in newtons; marker mask is E,T,H,W.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interactive_world_sim.utils.pusht_rollout_metrics import (
    evaluate_rollout,
    paired_bootstrap_mean_delta,
)


def load_export(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"predicted_rgb", "target_rgb", "contact_force", "marker_mask"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: data[key] for key in required}


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb = load_export(args.rgb)
    marker = load_export(args.marker)
    for key in ("target_rgb", "contact_force", "marker_mask"):
        if not np.array_equal(rgb[key], marker[key]):
            raise ValueError(f"Paired-control violation: {key} differs between exports")

    rgb_result = evaluate_rollout(
        predicted_rgb=rgb["predicted_rgb"],
        target_rgb=rgb["target_rgb"],
        contact_force=rgb["contact_force"],
        marker_mask=rgb["marker_mask"],
    )
    marker_result = evaluate_rollout(
        predicted_rgb=marker["predicted_rgb"],
        target_rgb=marker["target_rgb"],
        contact_force=marker["contact_force"],
        marker_mask=marker["marker_mask"],
    )
    rgb_primary = rgb_result["per_episode"]["strong_error_auc"]
    marker_primary = marker_result["per_episode"]["strong_error_auc"]
    bootstrap = paired_bootstrap_mean_delta(rgb_primary, marker_primary)
    rgb_mean = float(np.nanmean(rgb_primary))
    marker_mean = float(np.nanmean(marker_primary))
    relative_reduction = (rgb_mean - marker_mean) / rgb_mean if rgb_mean > 0 else 0.0

    rgb_psnr = np.asarray(rgb_result["per_episode"]["mean_clean_region_psnr_db"])
    marker_psnr = np.asarray(marker_result["per_episode"]["mean_clean_region_psnr_db"])
    psnr_delta = float(np.mean(marker_psnr - rgb_psnr))
    horizon_wins = 0
    for horizon in ("10", "30", "60", "100"):
        rgb_value = rgb_result["summary"]["strong_horizon_iou"][horizon]
        marker_value = marker_result["summary"]["strong_horizon_iou"][horizon]
        if (
            rgb_value is not None
            and marker_value is not None
            and marker_value >= rgb_value
        ):
            horizon_wins += 1

    gates = {
        "strong_error_relative_reduction_ge_10pct": relative_reduction >= 0.10,
        "bootstrap_ci_wholly_below_zero": bootstrap["ci95_high"] < 0.0,
        "strong_horizon_wins_ge_2": horizon_wins >= 2,
        "clean_region_psnr_loss_le_0_5db": psnr_delta >= -0.5,
    }
    report = {
        "evidence_class": "paired MuJoCo model rollout evaluation",
        "rgb": rgb_result,
        "marker": marker_result,
        "paired_primary_bootstrap": bootstrap,
        "strong_error_relative_reduction": relative_reduction,
        "clean_region_psnr_delta_db": psnr_delta,
        "strong_horizon_wins": horizon_wins,
        "gates": gates,
        "marker_advantage_claim": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2) + "\n")
    print(json.dumps(json_ready(report["gates"]), indent=2))


if __name__ == "__main__":
    main()
