import cv2
import numpy as np

from interactive_world_sim.utils.pusht_keypoints import (
    PUSHT_LOCAL_KEYPOINTS,
    center_crop_resize_with_points,
    make_pusht_marker_observation,
    project_world_points,
    pusht_keypoints_world,
)


def test_local_keypoints_match_scaled_mesh_bounds() -> None:
    np.testing.assert_allclose(PUSHT_LOCAL_KEYPOINTS.min(axis=0), [-0.08, -0.14, 0.032])
    np.testing.assert_allclose(PUSHT_LOCAL_KEYPOINTS.max(axis=0), [0.08, 0.02, 0.032])


def test_keypoints_follow_full_object_pose() -> None:
    # 90 degrees around world z, with wxyz quaternion ordering.
    state = np.array([1.0, 2.0, 0.07, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    points = pusht_keypoints_world(state)
    expected_xy = np.stack(
        [-PUSHT_LOCAL_KEYPOINTS[:, 1], PUSHT_LOCAL_KEYPOINTS[:, 0]], axis=1
    )
    np.testing.assert_allclose(
        points[:, :2], expected_xy + [1.0, 2.0], atol=1e-8  # noqa: RUF005
    )
    np.testing.assert_allclose(points[:, 2], 0.102, atol=1e-8)


def test_projection_uses_world_to_camera_without_inversion() -> None:
    points = np.array([[1.0, 2.0, 4.0], [-1.0, 0.0, -1.0]])
    intrinsics = np.array([[10.0, 0.0, 20.0], [0.0, 10.0, 30.0], [0.0, 0.0, 1.0]])
    world_to_camera = np.eye(4)
    uv, visible = project_world_points(points, intrinsics, world_to_camera, (100, 100))
    np.testing.assert_allclose(uv[0], [22.5, 35.0])
    assert visible.tolist() == [True, False]
    assert np.isnan(uv[1]).all()


def test_crop_resize_matches_repository_center_crop_geometry() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    uv = np.array([[80.0, 0.0], [560.0, 480.0], [320.0, 240.0]])
    resized, transformed, visible = center_crop_resize_with_points(image, uv)
    assert resized.shape == (128, 128, 3)
    np.testing.assert_allclose(transformed, [[0.0, 0.0], [128.0, 128.0], [64.0, 64.0]])
    assert visible.tolist() == [True, False, True]


def test_marker_pair_differs_only_inside_saved_mask() -> None:
    image = np.full((480, 640, 3), 73, dtype=np.uint8)
    state = np.array([0.0, 0.0, 0.07, 1.0, 0.0, 0.0, 0.0])
    intrinsics = np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    world_to_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.685],
            [0, 0, 0, 1],
        ]
    )
    obs = make_pusht_marker_observation(image, state, intrinsics, world_to_camera)
    changed = np.any(obs.rgb != obs.rgb_keypoint_marker, axis=-1)
    assert obs.rgb.shape == obs.rgb_keypoint_marker.shape == (128, 128, 3)
    assert obs.keypoints_visible.all()
    assert np.array_equal(changed, obs.marker_mask > 0)
    assert cv2.countNonZero(obs.marker_mask) > 0
