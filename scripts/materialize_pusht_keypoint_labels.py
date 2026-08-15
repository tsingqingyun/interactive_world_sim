#!/usr/bin/env python3
"""Materialize compact 2-D PushT keypoint labels from recorded MuJoCo state."""

# The upstream Gym wrapper exposes MuJoCo state through this private handle.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from gym_aloha.env import AlohaEnv

from interactive_world_sim.utils.pusht_keypoints import (
    PUSHT_KEYPOINT_NAMES,
    project_world_points,
    pusht_keypoints_world,
)


def episode_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def materialize_split(
    source_dir: Path,
    output_path: Path,
    camera_name: str,
    camera_intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    overwrite: bool,
) -> dict[str, int | str]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    episode_paths = sorted(source_dir.glob("episode_*.hdf5"), key=episode_number)
    if not episode_paths:
        raise FileNotFoundError(f"No episodes found under {source_dir}")

    split_uv = []
    split_visible = []
    episode_ends = []
    total_frames = 0
    for episode_path in episode_paths:
        with h5py.File(episode_path, "r") as episode:
            states = episode["env_state"][:]
            images = episode[f"obs/images/{camera_name}"]
            if states.shape != (len(images), 7):
                raise ValueError(
                    f"{episode_path}: env_state and {camera_name} are not aligned"
                )
            image_shape = tuple(int(value) for value in images.shape[1:3])

        episode_uv = np.empty((len(states), len(PUSHT_KEYPOINT_NAMES), 2), np.float32)
        episode_visible = np.empty(
            (len(states), len(PUSHT_KEYPOINT_NAMES)), dtype=bool
        )
        for frame_idx, state in enumerate(states):
            points_world = pusht_keypoints_world(state)
            uv, visible = project_world_points(
                points_world, camera_intrinsics, world_to_camera, image_shape
            )
            episode_uv[frame_idx] = uv
            episode_visible[frame_idx] = visible
        split_uv.append(episode_uv)
        split_visible.append(episode_visible)
        total_frames += len(states)
        episode_ends.append(total_frames)

    keypoints_uv = np.concatenate(split_uv, axis=0)
    keypoints_visible = np.concatenate(split_visible, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        keypoints_uv=keypoints_uv,
        keypoints_visible=keypoints_visible,
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
    )
    temporary_path.replace(output_path)
    return {
        "file": str(output_path),
        "episodes": len(episode_paths),
        "frames": total_frames,
        "visible_keypoints": int(keypoints_visible.sum()),
        "total_keypoints": int(keypoints_visible.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--camera", default="top_pov")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    env = AlohaEnv("pusht")
    try:
        env.reset(seed=0)
        camera_intrinsics = env.get_cam_intrinsic(args.camera, (128, 128))
        world_to_camera = env.get_cam_extrinsic(args.camera)
        reports = {}
        for split in args.splits:
            reports[split] = materialize_split(
                source_dir=args.source_root / split,
                output_path=args.output_root / f"{split}.npz",
                camera_name=args.camera,
                camera_intrinsics=camera_intrinsics,
                world_to_camera=world_to_camera,
                overwrite=args.overwrite,
            )
    finally:
        env._env.physics.free()

    manifest = {
        "contract": "8 ordered UV keypoints projected from recorded MuJoCo env_state",
        "source_root": str(args.source_root),
        "camera": args.camera,
        "image_shape": [128, 128],
        "coordinate_order": ["u", "v"],
        "keypoint_names": list(PUSHT_KEYPOINT_NAMES),
        "splits": reports,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
