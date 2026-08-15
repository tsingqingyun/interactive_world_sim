"""Geometry and rendering helpers for the MuJoCo bimanual PushT task.

The eight points are the top-surface corners already used by the scripted
collector, but their dimensions and height are tied to the scaled MuJoCo mesh.
All image coordinates are stored as ``(u, v) == (column, row)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import transforms3d

PUSHT_KEYPOINT_NAMES = (
    "top_left",
    "top_right",
    "top_bar_bottom_left",
    "top_bar_bottom_right",
    "stem_top_left",
    "stem_top_right",
    "stem_bottom_left",
    "stem_bottom_right",
)

# t_shape.stl bounds are x=[-0.10, 0.10], y=[-0.175, 0.025],
# z=[0.0, 0.04] metres. The XML applies scale="0.8 0.8 0.8".
PUSHT_LOCAL_KEYPOINTS = np.array(
    [
        [-0.080, 0.020, 0.032],
        [0.080, 0.020, 0.032],
        [-0.080, -0.020, 0.032],
        [0.080, -0.020, 0.032],
        [-0.020, -0.020, 0.032],
        [0.020, -0.020, 0.032],
        [-0.020, -0.140, 0.032],
        [0.020, -0.140, 0.032],
    ],
    dtype=np.float64,
)

# RGB colours. Red-dominant colours are deliberately excluded so marker pixels
# cannot be mistaken for the red T object by the evaluation segmenter.
PUSHT_MARKER_COLORS_RGB = np.array(
    [
        [0, 255, 0],
        [0, 255, 128],
        [0, 255, 255],
        [0, 128, 255],
        [0, 0, 255],
        [128, 255, 0],
        [128, 255, 128],
        [128, 255, 255],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class PushTMarkerObservation:
    """Paired clean/marker observation and its projection metadata."""

    rgb: np.ndarray
    rgb_keypoint_marker: np.ndarray
    marker_mask: np.ndarray
    keypoints_world: np.ndarray
    keypoints_uv: np.ndarray
    keypoints_visible: np.ndarray


def env_state_to_world_object(env_state: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` to a world-from-object pose."""
    env_state = np.asarray(env_state, dtype=np.float64)
    if env_state.shape != (7,):
        raise ValueError(f"env_state must have shape (7,), got {env_state.shape}")
    quaternion = env_state[3:]
    if not np.isfinite(env_state).all() or np.linalg.norm(quaternion) < 1e-8:
        raise ValueError("env_state must be finite and contain a nonzero quaternion")

    world_from_object = np.eye(4, dtype=np.float64)
    world_from_object[:3, :3] = transforms3d.quaternions.quat2mat(quaternion)
    world_from_object[:3, 3] = env_state[:3]
    return world_from_object


def pusht_keypoints_world(env_state: np.ndarray) -> np.ndarray:
    """Return the ordered eight T top-surface corners in world metres."""
    world_from_object = env_state_to_world_object(env_state)
    points_h = np.concatenate(
        [PUSHT_LOCAL_KEYPOINTS, np.ones((len(PUSHT_LOCAL_KEYPOINTS), 1))], axis=1
    )
    return (world_from_object @ points_h.T).T[:, :3]


def project_world_points(
    points_world: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using OpenCV axes and return UV plus visibility."""
    points_world = np.asarray(points_world, dtype=np.float64)
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"points_world must have shape (N, 3), got {points_world.shape}"
        )
    if camera_intrinsics.shape != (3, 3):
        raise ValueError("camera_intrinsics must have shape (3, 3)")
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape (4, 4)")

    points_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_camera = (world_to_camera @ points_h.T).T[:, :3]
    depth = points_camera[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    in_front = depth > 1e-8
    uv[in_front, 0] = (
        camera_intrinsics[0, 0] * points_camera[in_front, 0] / depth[in_front]
        + camera_intrinsics[0, 2]
    )
    uv[in_front, 1] = (
        camera_intrinsics[1, 1] * points_camera[in_front, 1] / depth[in_front]
        + camera_intrinsics[1, 2]
    )
    height, width = image_shape
    visible = (
        in_front
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, visible


def center_crop_resize_with_points(
    image: np.ndarray,
    uv: np.ndarray,
    output_shape: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the dataset's centre crop/resize to an image and UV points."""
    image = np.asarray(image)
    uv = np.asarray(uv, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (N, 2), got {uv.shape}")

    height, width = image.shape[:2]
    out_height, out_width = output_shape
    if height / width > out_height / out_width:
        crop_width = width
        crop_height = int(round(width * out_height / out_width))
    elif height / width < out_height / out_width:
        crop_height = height
        crop_width = int(round(height * out_width / out_height))
    else:
        crop_height, crop_width = height, width
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2

    cropped = image[y0 : y0 + crop_height, x0 : x0 + crop_width]
    resized = cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_AREA)
    transformed_uv = uv.copy()
    transformed_uv[:, 0] = (transformed_uv[:, 0] - x0) * out_width / crop_width
    transformed_uv[:, 1] = (transformed_uv[:, 1] - y0) * out_height / crop_height
    in_crop = (
        np.isfinite(transformed_uv).all(axis=1)
        & (transformed_uv[:, 0] >= 0)
        & (transformed_uv[:, 0] < out_width)
        & (transformed_uv[:, 1] >= 0)
        & (transformed_uv[:, 1] < out_height)
    )
    return resized, transformed_uv, in_crop


def draw_keypoint_markers(
    image: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the eight ordered markers and return the overlay and binary mask."""
    image = np.asarray(image)
    uv = np.asarray(uv, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    if len(uv) != len(PUSHT_MARKER_COLORS_RGB) or visible.shape != (len(uv),):
        raise ValueError("PushT marker inputs must contain exactly eight points")
    if radius < 1:
        raise ValueError("radius must be positive")

    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for point, is_visible, color in zip(
        uv, visible, PUSHT_MARKER_COLORS_RGB, strict=False
    ):
        if not is_visible:
            continue
        center = tuple(np.rint(point).astype(int).tolist())
        cv2.circle(overlay, center, radius, tuple(int(v) for v in color), thickness=-1)
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return overlay, mask


def make_pusht_marker_observation(
    image: np.ndarray,
    env_state: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    output_shape: tuple[int, int] = (128, 128),
    marker_radius: int = 3,
) -> PushTMarkerObservation:
    """Create synchronized clean RGB and RGB-with-eight-marker observations."""
    points_world = pusht_keypoints_world(env_state)
    uv_full, visible_full = project_world_points(
        points_world, camera_intrinsics, world_to_camera, image.shape[:2]
    )
    rgb, uv, visible_crop = center_crop_resize_with_points(image, uv_full, output_shape)
    visible = visible_full & visible_crop
    overlay, marker_mask = draw_keypoint_markers(rgb, uv, visible, marker_radius)
    return PushTMarkerObservation(
        rgb=rgb,
        rgb_keypoint_marker=overlay,
        marker_mask=marker_mask,
        keypoints_world=points_world,
        keypoints_uv=uv,
        keypoints_visible=visible,
    )
