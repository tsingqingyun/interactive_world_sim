#!/usr/bin/env python3
"""Materialize paired PushT RGB and RGB+marker datasets from one trajectory source.

The simulator is never stepped by this script. Dataset A is an exact copy of
the canonical HDF5 episode. Dataset B starts from the same copy and only
replaces ``obs/images/top_pov`` with the eight-keypoint overlay. Consequently,
both datasets use the same observation key and the IWS training pipeline does
not need a marker-specific code path.
"""

# The upstream Gym wrapper exposes MuJoCo state through this private handle.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from gym_aloha.env import AlohaEnv

from interactive_world_sim.utils.pusht_keypoints import (
    draw_keypoint_markers,
    project_world_points,
    pusht_keypoints_world,
)


def update_hash_from_dataset(digest: Any, dataset: h5py.Dataset) -> None:
    """Hash a dataset in bounded slices without loading an episode twice."""
    digest.update(dataset.name.encode())
    digest.update(str(dataset.shape).encode())
    digest.update(str(dataset.dtype).encode())
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes())
        return
    for start in range(0, len(dataset), 64):
        digest.update(np.ascontiguousarray(dataset[start : start + 64]).tobytes())


def shared_payload_sha256(file: h5py.File, excluded_dataset: str) -> str:
    """Hash every dataset except the image array intentionally changed in B."""
    digest = hashlib.sha256()
    excluded_dataset = excluded_dataset.lstrip("/")

    def visit(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset) and name != excluded_dataset:
            update_hash_from_dataset(digest, item)

    file.visititems(visit)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_episode(
    source_path: Path,
    rgb_path: Path,
    marker_path: Path,
    camera_intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    camera_name: str,
    marker_radius: int,
) -> dict[str, object]:
    """Copy one canonical episode into A/B and overlay only B's RGB frames."""
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != rgb_path.resolve():
        shutil.copy2(source_path, rgb_path)
    shutil.copy2(source_path, marker_path)

    changed_pixels = 0
    visible_points = 0
    total_points = 0
    image_key = f"obs/images/{camera_name}"
    with h5py.File(marker_path, "r+") as marker_file:
        images = marker_file[image_key]
        states = marker_file["env_state"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(
                f"Unexpected image contract: {images.shape} {images.dtype}"
            )
        if states.shape != (len(images), 7):
            raise ValueError(
                f"env_state must be (T,7) and aligned with images, got {states.shape}"
            )

        for start in range(0, len(images), 64):
            stop = min(start + 64, len(images))
            image_chunk = images[start:stop]
            state_chunk = states[start:stop]
            marker_chunk = image_chunk.copy()
            for local_idx, (image, state) in enumerate(
                zip(image_chunk, state_chunk, strict=False)
            ):
                points_world = pusht_keypoints_world(state)
                uv, visible = project_world_points(
                    points_world,
                    camera_intrinsics,
                    world_to_camera,
                    image.shape[:2],
                )
                overlay, marker_mask = draw_keypoint_markers(
                    image, uv, visible, radius=marker_radius
                )
                marker_chunk[local_idx] = overlay
                changed_pixels += int(np.count_nonzero(marker_mask))
                visible_points += int(visible.sum())
                total_points += len(visible)
            images[start:stop] = marker_chunk

        marker_file.attrs["pusht_marker_source"] = "oracle MuJoCo env_state"
        marker_file.attrs["pusht_marker_radius_px"] = marker_radius
        marker_file.attrs["pusht_marker_keypoint_count"] = 8

    with (
        h5py.File(rgb_path, "r") as rgb_file,
        h5py.File(marker_path, "r") as marker_file,
    ):
        rgb_trajectory_hash = shared_payload_sha256(rgb_file, image_key)
        marker_trajectory_hash = shared_payload_sha256(marker_file, image_key)
        if rgb_trajectory_hash != marker_trajectory_hash:
            raise RuntimeError(
                "Dataset A/B non-image arrays differ after materialization"
            )

    source_hash = file_sha256(source_path)
    rgb_file_hash = file_sha256(rgb_path)
    if source_hash != rgb_file_hash:
        raise RuntimeError("Dataset A is not an exact byte copy of its source episode")

    return {
        "source": str(source_path),
        "rgb": str(rgb_path),
        "marker": str(marker_path),
        "source_sha256": source_hash,
        "rgb_file_sha256": rgb_file_hash,
        "rgb_trajectory_sha256": rgb_trajectory_hash,
        "marker_trajectory_sha256": marker_trajectory_hash,
        "frames": total_points // 8,
        "visible_keypoints": visible_points,
        "total_keypoints": total_points,
        "marker_mask_pixels": changed_pixels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--camera", default="top_pov")
    parser.add_argument("--marker-radius", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset_a.resolve() == args.dataset_b.resolve():
        raise ValueError("Dataset A and B must be different directories")
    existing = [args.dataset_b] if args.dataset_b.exists() else []
    if (
        args.source_root.resolve() != args.dataset_a.resolve()
        and args.dataset_a.exists()
    ):
        existing.append(args.dataset_a)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing}; pass --overwrite only for "
            "generated outputs"
        )

    env = AlohaEnv("pusht")
    env.reset(seed=0)
    camera_intrinsics = env.get_cam_intrinsic(args.camera, (128, 128))
    world_to_camera = env.get_cam_extrinsic(args.camera)
    episode_reports = []

    for split in args.splits:
        source_dir = args.source_root / split
        source_paths = sorted(
            source_dir.glob("episode_*.hdf5"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not source_paths:
            raise FileNotFoundError(f"No episodes found under {source_dir}")
        for source_path in source_paths:
            episode_reports.append(
                materialize_episode(
                    source_path=source_path,
                    rgb_path=args.dataset_a / split / source_path.name,
                    marker_path=args.dataset_b / split / source_path.name,
                    camera_intrinsics=camera_intrinsics,
                    world_to_camera=world_to_camera,
                    camera_name=args.camera,
                    marker_radius=args.marker_radius,
                )
            )

    env._env.physics.free()
    manifest = {
        "contract": "same source episodes; A=RGB; B=RGB+8 oracle keypoint markers",
        "camera": args.camera,
        "marker_radius_px": args.marker_radius,
        "dataset_a": str(args.dataset_a),
        "dataset_b": str(args.dataset_b),
        "episodes": episode_reports,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (args.dataset_a / "paired_manifest.json").write_text(manifest_text)
    (args.dataset_b / "paired_manifest.json").write_text(manifest_text)
    print(
        json.dumps(
            {
                "episodes": len(episode_reports),
                "dataset_a": str(args.dataset_a),
                "dataset_b": str(args.dataset_b),
                "visible_keypoints": sum(
                    int(report["visible_keypoints"]) for report in episode_reports
                ),
                "total_keypoints": sum(
                    int(report["total_keypoints"]) for report in episode_reports
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
