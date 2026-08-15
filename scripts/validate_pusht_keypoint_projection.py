"""Validate the eight PushT projections against MuJoCo geometry segmentation."""

# The upstream Gym wrapper exposes MuJoCo state through these private handles.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from gym_aloha.env import AlohaEnv

from interactive_world_sim.utils.pusht_keypoints import (
    make_pusht_marker_observation,
    project_world_points,
    pusht_keypoints_world,
)
from interactive_world_sim.utils.pusht_rollout_metrics import red_t_mask


def crop_mask(
    mask: np.ndarray, output_shape: tuple[int, int] = (128, 128)
) -> np.ndarray:
    height, width = mask.shape
    out_height, out_width = output_shape
    if height / width < out_height / out_width:
        crop_height = height
        crop_width = int(round(height * out_width / out_height))
    else:
        crop_width = width
        crop_height = int(round(width * out_height / out_width))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    cropped = mask[y0 : y0 + crop_height, x0 : x0 + crop_width]
    return cv2.resize(
        cropped.astype(np.uint8),
        (out_width, out_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction_artifacts/pusht_keypoint_projection"),
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--num-seeds", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = AlohaEnv("pusht")
    physics = env._env.physics
    model = physics.model._model
    pusht_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusht")
    corner_distances: list[float] = []
    probe_ious: list[float] = []
    visible_counts: list[int] = []

    for offset in range(args.num_seeds):
        seed = args.seed + offset
        env.reset(seed=seed)
        raw = env._env.task.get_observation(physics)
        image = raw["images"]["top_pov"]
        height, width = image.shape[:2]
        segmentation = physics.render(
            height=height, width=width, camera_id="top_pov", segmentation=True
        )
        t_mask_full = segmentation[..., 0] == pusht_geom_id
        distance = cv2.distanceTransform(
            (~t_mask_full).astype(np.uint8), cv2.DIST_L2, 3
        )

        intrinsics = env.get_cam_intrinsic("top_pov", (height, width))
        extrinsics = env.get_cam_extrinsic("top_pov")
        points_world = pusht_keypoints_world(raw["env_state"])
        uv_full, visible_full = project_world_points(
            points_world, intrinsics, extrinsics, (height, width)
        )
        for point, visible in zip(uv_full, visible_full, strict=False):
            if visible:
                u = int(np.clip(round(point[0]), 0, width - 1))
                v = int(np.clip(round(point[1]), 0, height - 1))
                corner_distances.append(float(distance[v, u]))

        marker_obs = make_pusht_marker_observation(
            image, raw["env_state"], intrinsics, extrinsics
        )
        visible_counts.append(int(marker_obs.keypoints_visible.sum()))
        t_mask = crop_mask(t_mask_full)
        probe_mask = red_t_mask(marker_obs.rgb)
        probe_ious.append(intersection_over_union(t_mask, probe_mask))

        if offset == 0:
            segmentation_rgb = np.zeros_like(marker_obs.rgb)
            segmentation_rgb[t_mask] = [255, 0, 0]
            panel = np.concatenate(
                [marker_obs.rgb, marker_obs.rgb_keypoint_marker, segmentation_rgb],
                axis=1,
            )
            cv2.imwrite(
                str(args.output_dir / "projection_panel.png"),
                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            )

    distances = np.asarray(corner_distances)
    ious = np.asarray(probe_ious)
    report = {
        "environment": "AlohaEnv(pusht)",
        "camera": "top_pov",
        "output_resolution": [128, 128],
        "red_probe": "v2: R>=100, R-G>=90, R-B>=90, largest component",
        "seed_start": args.seed,
        "num_seeds": args.num_seeds,
        "projected_visible_keypoints_per_frame": visible_counts,
        "corner_to_mujoco_mask_distance_px": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances.max()),
        },
        "red_probe_iou": {
            "mean": float(ious.mean()),
            "p05": float(np.percentile(ious, 5)),
            "min": float(ious.min()),
        },
        "gates": {
            "projection_p95_le_2px": bool(np.percentile(distances, 95) <= 2.0),
            "all_eight_visible": bool(all(count == 8 for count in visible_counts)),
            "red_probe_mean_iou_ge_0_90": bool(ious.mean() >= 0.90),
            "red_probe_p05_iou_ge_0_80": bool(np.percentile(ious, 5) >= 0.80),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    physics.free()


if __name__ == "__main__":
    main()
