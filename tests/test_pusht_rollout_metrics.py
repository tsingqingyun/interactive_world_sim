import numpy as np

from interactive_world_sim.utils.pusht_rollout_metrics import (
    evaluate_rollout,
    paired_bootstrap_mean_delta,
    red_t_mask,
    simultaneous_strong_contact,
)


def test_red_probe_ignores_wood_like_background() -> None:
    image = np.full((32, 32, 3), [220, 170, 110], dtype=np.uint8)
    image[10:20, 12:18] = [255, 20, 20]
    mask = red_t_mask(image)
    assert mask.sum() == 60


def test_strong_contact_requires_complete_three_frame_run() -> None:
    force = np.zeros((1, 8, 2), dtype=np.float32)
    force[0, 1:3] = 2.0
    force[0, 4:7] = 2.0
    assert simultaneous_strong_contact(force).tolist() == [
        [False, False, False, False, True, True, True, False]
    ]


def test_perfect_100_step_rollout_has_zero_state_error() -> None:
    target = np.zeros((2, 100, 16, 16, 3), dtype=np.uint8)
    target[:, :, 5:11, 6:10, 0] = 255
    force = np.full((2, 100, 2), 2.0, dtype=np.float32)
    marker_mask = np.zeros((2, 100, 16, 16), dtype=np.uint8)
    result = evaluate_rollout(target.copy(), target, force, marker_mask)
    np.testing.assert_allclose(result["per_episode"]["strong_error_auc"], 0.0)
    assert result["summary"]["horizon_iou"]["100"] == 1.0


def test_paired_bootstrap_reports_marker_minus_rgb() -> None:
    result = paired_bootstrap_mean_delta(
        np.array([0.4, 0.5, 0.6]), np.array([0.2, 0.3, 0.4]), samples=1000
    )
    assert result["mean_delta_marker_minus_rgb"] < 0
    assert result["ci95_high"] < 0
