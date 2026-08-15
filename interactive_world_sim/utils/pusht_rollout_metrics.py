"""Frozen metrics for paired RGB and RGB+marker PushT rollouts."""

from __future__ import annotations

import cv2
import numpy as np


def red_t_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Segment the red T with the v2 probe frozen in the experiment protocol."""
    channels = np.asarray(image_rgb).astype(np.int16)
    if channels.ndim != 3 or channels.shape[-1] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {channels.shape}")
    red = (
        (channels[..., 0] >= 100)
        & (channels[..., 0] - channels[..., 1] >= 90)
        & (channels[..., 0] - channels[..., 2] >= 90)
    ).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    if n_labels <= 1:
        return np.zeros(red.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def simultaneous_strong_contact(
    contact_force: np.ndarray, threshold_newtons: float = 1.0, min_frames: int = 3
) -> np.ndarray:
    """Mark complete runs where both arms contact the T strongly for >=3 frames."""
    contact_force = np.asarray(contact_force)
    if contact_force.ndim != 3 or contact_force.shape[-1] != 2:
        raise ValueError("contact_force must have shape (episodes, time, 2)")
    simultaneous = np.all(contact_force >= threshold_newtons, axis=-1)
    strong = np.zeros_like(simultaneous)
    for episode_idx, row in enumerate(simultaneous):
        padded = np.concatenate([[False], row, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        for start, end in edges.reshape(-1, 2):
            if end - start >= min_frames:
                strong[episode_idx, start:end] = True
    return strong


def _mask_metrics(
    predicted_rgb: np.ndarray, target_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    episodes, steps, height, width, _ = predicted_rgb.shape
    iou = np.empty((episodes, steps), dtype=np.float64)
    centroid_error = np.empty_like(iou)
    missing_error = float(np.hypot(height, width))
    for episode_idx in range(episodes):
        for step_idx in range(steps):
            predicted = red_t_mask(predicted_rgb[episode_idx, step_idx])
            target = red_t_mask(target_rgb[episode_idx, step_idx])
            union = np.logical_or(predicted, target).sum()
            iou[episode_idx, step_idx] = (
                np.logical_and(predicted, target).sum() / union if union else 1.0
            )
            pred_y, pred_x = np.nonzero(predicted)
            target_y, target_x = np.nonzero(target)
            if len(pred_x) == 0 or len(target_x) == 0:
                centroid_error[episode_idx, step_idx] = missing_error
            else:
                centroid_error[episode_idx, step_idx] = np.hypot(
                    pred_x.mean() - target_x.mean(), pred_y.mean() - target_y.mean()
                )
    return iou, centroid_error


def _masked_episode_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for episode_idx in range(values.shape[0]):
        if mask[episode_idx].any():
            result[episode_idx] = values[episode_idx, mask[episode_idx]].mean()
    return result


def _time_to_failure(
    iou: np.ndarray, threshold: float = 0.5, min_frames: int = 3
) -> np.ndarray:
    result = np.full(iou.shape[0], iou.shape[1] + 1, dtype=np.int64)
    failed = iou < threshold
    for episode_idx, row in enumerate(failed):
        for start in range(len(row) - min_frames + 1):
            if row[start : start + min_frames].all():
                result[episode_idx] = start + 1
                break
    return result


def _clean_region_psnr(
    predicted_rgb: np.ndarray, target_rgb: np.ndarray, marker_mask: np.ndarray
) -> np.ndarray:
    result = np.empty(predicted_rgb.shape[:2], dtype=np.float64)
    kernel = np.ones((11, 11), dtype=np.uint8)
    for episode_idx in range(predicted_rgb.shape[0]):
        for step_idx in range(predicted_rgb.shape[1]):
            excluded = cv2.dilate(
                marker_mask[episode_idx, step_idx].astype(np.uint8), kernel
            ).astype(bool)
            error = predicted_rgb[episode_idx, step_idx].astype(
                np.float64
            ) - target_rgb[episode_idx, step_idx].astype(np.float64)
            mse = np.mean(np.square(error[~excluded]))
            result[episode_idx, step_idx] = (
                float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)
            )
    return result


def evaluate_rollout(
    predicted_rgb: np.ndarray,
    target_rgb: np.ndarray,
    contact_force: np.ndarray,
    marker_mask: np.ndarray,
    horizons: tuple[int, ...] = (10, 30, 60, 100),
) -> dict[str, object]:
    """Evaluate one model; arrays use (episodes, time, height, width, channels)."""
    predicted_rgb = np.asarray(predicted_rgb)
    target_rgb = np.asarray(target_rgb)
    contact_force = np.asarray(contact_force)
    marker_mask = np.asarray(marker_mask)
    if predicted_rgb.shape != target_rgb.shape or predicted_rgb.ndim != 5:
        raise ValueError(
            "predicted_rgb and target_rgb must have identical (E,T,H,W,3) shape"
        )
    if predicted_rgb.shape[-1] != 3:
        raise ValueError("RGB arrays must have three channels")
    if contact_force.shape != (*predicted_rgb.shape[:2], 2):
        raise ValueError("contact_force shape does not match RGB episode/time axes")
    if marker_mask.shape != predicted_rgb.shape[:4]:
        raise ValueError(
            "marker_mask shape does not match RGB episode/time/spatial axes"
        )
    if max(horizons) > predicted_rgb.shape[1]:
        raise ValueError("rollout is shorter than the requested maximum horizon")

    iou, centroid_error = _mask_metrics(predicted_rgb, target_rgb)
    strong = simultaneous_strong_contact(contact_force)
    no_contact = np.all(contact_force < 1.0, axis=-1)
    clean_psnr = _clean_region_psnr(predicted_rgb, target_rgb, marker_mask)
    error = 1.0 - iou

    horizon_iou = {}
    strong_horizon_iou = {}
    for horizon in horizons:
        values = iou[:, horizon - 1]
        horizon_iou[str(horizon)] = float(values.mean())
        selected = values[strong[:, horizon - 1]]
        strong_horizon_iou[str(horizon)] = (
            float(selected.mean()) if len(selected) else None
        )

    return {
        "per_episode": {
            "strong_error_auc": _masked_episode_mean(error, strong),
            "no_contact_error_auc": _masked_episode_mean(error, no_contact),
            "mean_centroid_error_px": centroid_error.mean(axis=1),
            "mean_clean_region_psnr_db": clean_psnr.mean(axis=1),
            "time_to_failure_steps": _time_to_failure(iou),
        },
        "summary": {
            "strong_frame_count": int(strong.sum()),
            "no_contact_frame_count": int(no_contact.sum()),
            "horizon_iou": horizon_iou,
            "strong_horizon_iou": strong_horizon_iou,
            "mean_centroid_error_px": float(centroid_error.mean()),
            "mean_clean_region_psnr_db": float(clean_psnr.mean()),
            "median_time_to_failure_steps": float(np.median(_time_to_failure(iou))),
        },
    }


def paired_bootstrap_mean_delta(
    rgb_values: np.ndarray,
    marker_values: np.ndarray,
    seed: int = 20260814,
    samples: int = 10000,
) -> dict[str, float | int]:
    """Bootstrap paired episode deltas, defined as marker minus RGB."""
    rgb_values = np.asarray(rgb_values, dtype=np.float64)
    marker_values = np.asarray(marker_values, dtype=np.float64)
    valid = np.isfinite(rgb_values) & np.isfinite(marker_values)
    delta = marker_values[valid] - rgb_values[valid]
    if len(delta) < 2:
        raise ValueError("At least two paired finite episodes are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "paired_episodes": int(len(delta)),
        "mean_delta_marker_minus_rgb": float(delta.mean()),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
    }
