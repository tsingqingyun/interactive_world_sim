"""Usage:

Scripted Data Collection with Auto-Recording:
- By default, episodes are automatically recorded every 3 seconds
- Press "Q" to exit program
- Press "A" to toggle auto-recording on/off
- Manual controls (when auto-recording is OFF):
  - Press "C" to start recording
  - Press "S" to stop recording
  - Press "Backspace" to delete the previously recorded episode

The system uses a scripted policy that generates 4 types of motions:
1. Linear pushes (30%): Coordinated pushing in different directions
2. Rotations (30%): Rotating the T-shape by gripping corners
3. Random contact (30%): Random contact points with collision avoidance
4. Random exploration (10%): Non-contact exploration motions
"""

# The upstream Gym wrapper exposes MuJoCo state through these private handles.
# ruff: noqa: SLF001

# %%
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

import click
import cv2
import mujoco
import numpy as np
import transforms3d
from gym_aloha.env import AlohaEnv
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.utils.draw_utils import plot_single_3d_pos_traj
from interactive_world_sim.utils.motion_planner import (
    MotionPlanner,
    TShapeInfo,
    WorkspaceConstraints,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert
from interactive_world_sim.utils.pusht_keypoints import (
    PUSHT_LOCAL_KEYPOINTS,
    draw_keypoint_markers,
    make_pusht_marker_observation,
    project_world_points,
    pusht_keypoints_world,
)


def env_state_to_mat(env_state: np.ndarray) -> np.ndarray:
    """Convert environment state to matrix."""
    position = env_state[:3]
    quaternion = env_state[3:]
    rot_matrix = transforms3d.quaternions.quat2mat(quaternion)
    world_t_obj = np.eye(4)
    world_t_obj[:3, :3] = rot_matrix
    world_t_obj[:3, 3] = position
    return world_t_obj


def extract_t_info(env_state: np.ndarray) -> TShapeInfo:
    """Extract T-shape information from environment state."""
    # env_state contains [x, y, z, qw, qx, qy, qz] for the T-shape
    world_t_obj = env_state_to_mat(env_state)
    position = world_t_obj[:3, 3]
    rot_matrix = world_t_obj[:3, :3]

    rotation_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # -pi/2 to pi/2

    # Exact scaled mesh bounds: width=0.16 m, height=0.16 m,
    # thickness=0.04 m and top surface z=0.032 m in the object frame.
    width = 0.16
    height = 0.16
    thickness = 0.04
    keypoints = PUSHT_LOCAL_KEYPOINTS.copy()

    return TShapeInfo(
        center=position[:2],  # Only use x, y
        rotation=rotation_angle,
        keypoints=keypoints,
        width=width,
        height=height,
        thickness=thickness,
        pose=world_t_obj,
    )


def get_current_arm_positions(
    obs: dict, kin_helper: KinHelper, world_t_bases: np.ndarray
) -> np.ndarray:
    """Get current end-effector positions for both arms in left base frame."""
    # Extract joint positions
    left_qpos = obs["qpos"][:7]
    right_qpos = obs["qpos"][7:]

    # Compute forward kinematics
    left_qpos_extended = np.concatenate([left_qpos, left_qpos[6:7]])
    right_qpos_extended = np.concatenate([right_qpos, right_qpos[6:7]])

    left_ee_pose = kin_helper.compute_fk_from_link_idx(
        left_qpos_extended, [kin_helper.sapien_eef_idx]
    )[0]
    right_ee_pose = kin_helper.compute_fk_from_link_idx(
        right_qpos_extended, [kin_helper.sapien_eef_idx]
    )[0]
    world_t_left_ee = world_t_bases[0] @ left_ee_pose
    world_t_right_ee = world_t_bases[1] @ right_ee_pose

    # Extract XY positions
    left_xy = world_t_left_ee[:2, 3]
    right_xy = world_t_right_ee[:2, 3]

    return np.concatenate([left_xy, right_xy])


def trajectory_to_joint_actions(
    target_xy: np.ndarray,
    world_t_bases: np.ndarray,
    kin_helper: KinHelper,
    curr_puppet_joint: np.ndarray,
    curr_vel: np.ndarray,
    dt: float,
    k_p: float = 50,
    k_v: float = 10,
    acc_lim: float = 1.0,
    vel_lim: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2D trajectory to joint actions using the existing PID controller."""
    assert target_xy.shape == (4,), f"action shape must be (4,), got {target_xy.shape}"
    puppet_target_state = np.zeros(14)
    target_xy_clip = target_xy.copy()

    for rob_i in range(2):  # left and right arms
        # Target end-effector pose
        world_t_ee_pose = np.eye(4)
        world_t_ee_pose[:2, 3] = target_xy[rob_i * 2 : rob_i * 2 + 2]
        world_t_ee_pose[2, 3] = 0.02

        theta = np.pi * 5.0 / 12.0
        target_ee_pose = np.linalg.inv(world_t_bases[rob_i]) @ world_t_ee_pose
        target_ee_pose[:3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )
        target_ee_pose[0, 3] = np.clip(target_ee_pose[0, 3], 0.25, 0.6)
        target_ee_pose[1, 3] = np.clip(target_ee_pose[1, 3], -0.2, 0.2)
        target_ee_pose[2, 3] = 0.02

        world_t_ee_pose_clip = world_t_bases[rob_i] @ target_ee_pose
        target_xy_clip[rob_i * 2 : rob_i * 2 + 2] = world_t_ee_pose_clip[:2, 3]

        # Get current joint positions
        ik_init_joint = np.concatenate(
            [curr_puppet_joint[7 * rob_i : 7 * rob_i + 6], np.zeros(2)]
        )

        # Compute current end-effector pose
        curr_ee_pose = kin_helper.compute_fk_from_link_idx(
            ik_init_joint, [kin_helper.sapien_eef_idx]
        )[0]

        # PID control
        pid_ee_pose = target_ee_pose.copy()
        acceleration = k_p * (target_ee_pose[:3, 3] - curr_ee_pose[:3, 3]) + k_v * (
            np.zeros(3) - curr_vel[rob_i * 3 : rob_i * 3 + 3]
        )
        acceleration = np.clip(acceleration, -acc_lim, acc_lim)
        next_vel = acceleration * dt + curr_vel[rob_i * 3 : rob_i * 3 + 3]
        next_vel = np.clip(next_vel, -vel_lim, vel_lim)
        avg_vel = (next_vel + curr_vel[rob_i * 3 : rob_i * 3 + 3]) / 2.0
        pid_ee_pose[:3, 3] = curr_ee_pose[:3, 3] + avg_vel * dt
        curr_vel[rob_i * 3 : rob_i * 3 + 3] = next_vel

        # Inverse kinematics
        ik_joint = kin_helper.compute_ik_from_mat(ik_init_joint, pid_ee_pose)
        puppet_target_state[7 * rob_i : 7 * rob_i + 6] = ik_joint[:6]
        puppet_target_state[7 * rob_i + 6] = 0.0  # Gripper closed

    return puppet_target_state, target_xy_clip


def init_episode() -> dict:
    episode = {
        "robot_bases": [],
        "env_state": [],
        "obs": {
            "joint_pos": [],
            "ee_pos": [],
            "pusht_arm_contact_force": [],
            "images": {},
            "keypoints": {},
        },
        "action": [],
    }
    return episode


def dict_list_to_np(episode: dict) -> dict:
    for key in list(episode.keys()):
        if isinstance(episode[key], list):
            episode[key] = np.stack(episode[key], axis=0)
        elif isinstance(episode[key], dict):
            episode[key] = dict_list_to_np(episode[key])
    return episode


def save_episode(episode: dict, output_dir: str, episode_id: int) -> None:
    ### create config dict
    config_dict: dict = {
        "obs": {
            "images": {},
            "keypoints": {},
            "pusht_arm_contact_force": {"dtype": "float32"},
        },
    }
    episode = dict_list_to_np(episode)

    for key, value in episode["obs"]["keypoints"].items():
        config_dict["obs"]["keypoints"][key] = {"dtype": str(value.dtype)}

    # Get first camera name for video saving
    cam_names = list(episode["obs"]["images"].keys())
    first_cam_name = cam_names[0] if cam_names else None

    for cam_name in cam_names:
        cam_height, cam_width = episode["obs"]["images"][cam_name].shape[1:3]
        color_save_kwargs = {
            "chunks": (1, cam_height, cam_width, 3),  # (1, 480, 640, 3)
            # "compression": "gzip",
            # "compression_opts": 9,
            "dtype": "uint8",
        }
        config_dict["obs"]["images"][cam_name] = color_save_kwargs

    ### save episode data
    episode_path = os.path.join(output_dir, f"episode_{episode_id}.hdf5")
    attr_dict = {
        "pusht_keypoint_order": (
            "top_left,top_right,top_bar_bottom_left,top_bar_bottom_right,"
            "stem_top_left,stem_top_right,stem_bottom_left,stem_bottom_right"
        ),
        "pusht_keypoint_frame": "world_xyz_m; image_uv_col_row_px",
        "pusht_marker_radius_px": 3,
        "pusht_marker_observation": "top_pov_keypoint_marker",
    }
    save_dict_to_hdf5(episode, config_dict, str(episode_path), attr_dict=attr_dict)

    ### save video
    if first_cam_name is not None:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, f"episode_{episode_id}.mp4")
        video_frames = episode["obs"]["images"][first_cam_name]

        # Get video dimensions
        num_frames, height, width, channels = video_frames.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        # Write frames to video
        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    print(f"Episode {episode_id} saved!")


def visualize_t_keypoints_on_camera(
    img: np.ndarray, obs: dict, env: AlohaEnv, camera_name: str
) -> np.ndarray:
    """Visualize T-shape 3D keypoints projected onto camera image."""
    keypoints_world = pusht_keypoints_world(obs["env_state"])
    img_shape = img.shape[:2]  # (H, W)
    cam_intrinsics = env.get_cam_intrinsic(camera_name, img_shape)
    cam_extrinsics = env.get_cam_extrinsic(camera_name)
    keypoints_uv, visible = project_world_points(
        keypoints_world, cam_intrinsics, cam_extrinsics, img_shape
    )
    overlay, _ = draw_keypoint_markers(img, keypoints_uv, visible, radius=5)
    return overlay


def get_pusht_arm_contact_force(env: AlohaEnv) -> np.ndarray:
    """Return summed T-contact normal force for the left and right arm in newtons."""
    physics = env._env.physics
    model = physics.model._model
    data = physics.data._data
    pusht_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    arm_force = np.zeros(2, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        if body_1 == pusht_body_id:
            arm_body = body_2
        elif body_2 == pusht_body_id:
            arm_body = body_1
        else:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, arm_body)
        if body_name is None:
            continue
        if body_name.startswith("left/"):
            arm_idx = 0
        elif body_name.startswith("right/"):
            arm_idx = 1
        else:
            continue
        mujoco.mj_contactForce(model, data, contact_idx, contact_force)
        arm_force[arm_idx] += abs(contact_force[0])
    return arm_force.astype(np.float32)


def vis_obs(
    obs: dict,
    episode_id: int,
    is_recording: bool,
    env: Optional[AlohaEnv] = None,
    trajectory: Optional[np.ndarray] = None,
) -> np.ndarray:
    third_views = ["top_pov"]

    # Create visualization with T-shape keypoints overlay
    for view_name in third_views:
        img = obs["images"][view_name].copy()

        # Add T-shape keypoint visualization for top_pov camera
        if view_name == "top_pov" and env is not None:
            img = visualize_t_keypoints_on_camera(img, obs, env, "top_pov")
            if trajectory is not None:
                trajectory = trajectory.reshape(-1, 2, 2)
                trajectory = np.transpose(trajectory, (1, 0, 2))
                traj_3d = np.concatenate(
                    [
                        trajectory,
                        0.02 * np.ones((trajectory.shape[0], trajectory.shape[1], 1)),
                    ],
                    axis=-1,
                )
                cam_intrinsics = env.get_cam_intrinsic(view_name, img.shape[:-1])
                cam_extrinsics = env.get_cam_extrinsic(view_name)
                cx = cam_intrinsics[0, 2]
                cy = cam_intrinsics[1, 2]
                fx = cam_intrinsics[0, 0]
                fy = cam_intrinsics[1, 1]
                img = plot_single_3d_pos_traj(
                    img=img.copy(),
                    cam_intrisics=(cx, cy, fx, fy),
                    world_t_cam=cam_extrinsics,
                    trajs=traj_3d,
                    radius=5,
                )

        images_with_keypoints = img

    vis_img = images_with_keypoints
    text = f"Episode: {episode_id}"
    if is_recording:
        text += ", Recording!"
    vis_img = cv2.putText(
        vis_img.copy(),
        text,
        (10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
        color=(255, 255, 255),
    )
    return vis_img


def update_episode(
    episode: dict,
    obs: dict,
    action: np.ndarray,
    kin_helper: KinHelper,
    env: AlohaEnv,
) -> dict:
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    world_t_left_base = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_right_base = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    all_bases = np.stack([world_t_left_base, world_t_right_base])

    episode["robot_bases"].append(all_bases)
    episode["env_state"].append(obs["env_state"])
    episode["obs"]["joint_pos"].append(obs["qpos"])
    episode["obs"]["pusht_arm_contact_force"].append(get_pusht_arm_contact_force(env))

    # compute FK for obs
    left_qpos = obs["qpos"][:7]
    left_qpos = np.concatenate([left_qpos, left_qpos[6:7]])
    right_qpos = obs["qpos"][7:]
    right_qpos = np.concatenate([right_qpos, right_qpos[6:7]])
    left_base_t_left_eef = kin_helper.compute_fk_from_link_idx(
        left_qpos, [kin_helper.sapien_eef_idx]
    )[0]
    right_base_t_right_eef = kin_helper.compute_fk_from_link_idx(
        right_qpos, [kin_helper.sapien_eef_idx]
    )[0]
    world_t_left_eef = world_t_left_base @ left_base_t_left_eef
    world_t_right_eef = world_t_right_base @ right_base_t_right_eef
    world_t_all_eef = np.stack([world_t_left_eef, world_t_right_eef])
    episode["obs"]["ee_pos"].append(world_t_all_eef)

    for camera_name in list(obs["images"].keys()):
        if camera_name not in episode["obs"]["images"]:
            episode["obs"]["images"][camera_name] = []
        img = obs["images"][camera_name]
        if camera_name == "top_pov":
            marker_obs = make_pusht_marker_observation(
                image=img,
                env_state=obs["env_state"],
                camera_intrinsics=env.get_cam_intrinsic(camera_name, img.shape[:2]),
                world_to_camera=env.get_cam_extrinsic(camera_name),
            )
            marker_name = f"{camera_name}_keypoint_marker"
            episode["obs"]["images"].setdefault(marker_name, []).append(
                marker_obs.rgb_keypoint_marker
            )
            episode["obs"]["images"][camera_name].append(marker_obs.rgb)
            episode["obs"]["keypoints"].setdefault(f"{camera_name}_uv", []).append(
                marker_obs.keypoints_uv.astype(np.float32)
            )
            episode["obs"]["keypoints"].setdefault(f"{camera_name}_visible", []).append(
                marker_obs.keypoints_visible.astype(np.uint8)
            )
            episode["obs"]["keypoints"].setdefault(f"{camera_name}_world", []).append(
                marker_obs.keypoints_world.astype(np.float32)
            )
            episode["obs"]["keypoints"].setdefault(
                f"{camera_name}_marker_mask", []
            ).append(marker_obs.marker_mask)
        else:
            crop_img = center_crop(img, (128, 128))
            resize_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_AREA)
            episode["obs"]["images"][camera_name].append(resize_img)

    episode["action"].append(action)

    return episode


def generate_random_init_action(world_t_bases: np.ndarray) -> np.ndarray:
    rand_init_action = np.random.uniform(0, 1, size=(4,))
    rand_init_action[0] = rand_init_action[0] * 0.2 + 0.05
    rand_init_action[1] = rand_init_action[1] * 0.4 - 0.2
    rand_init_action[2] = rand_init_action[2] * 0.2 + 0.05
    rand_init_action[3] = rand_init_action[3] * 0.4 - 0.2
    left_base_t_left_action = np.eye(4)
    left_base_t_left_action[:2, 3] = rand_init_action[:2]
    right_base_t_right_action = np.eye(4)
    right_base_t_right_action[:2, 3] = rand_init_action[2:]
    world_t_left_action = world_t_bases[0] @ left_base_t_left_action
    world_t_right_action = world_t_bases[1] @ right_base_t_right_action
    action = np.concatenate([world_t_left_action[:2, 3], world_t_right_action[:2, 3]])
    return action


def task_reset(
    env: AlohaEnv,
    reset_seed: int,
    kin_helper: KinHelper,
    k_p: float,
    k_v: float,
    acc_lim: float,
    vel_lim: float,
    headless: bool,
) -> None:
    env.reset(seed=reset_seed)
    obs = env._env.task.get_observation(env._env.physics)
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    left_base_mat = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    right_base_mat = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_bases = np.stack([left_base_mat, right_base_mat])
    rand_init_action = generate_random_init_action(world_t_bases)
    curr_vel = np.zeros(6)
    curr_puppet_joint = obs["qpos"][:14]
    dt = 1 / 10.0
    for _ in range(100):
        obs = env._env.task.get_observation(env._env.physics)
        curr_puppet_joint = obs["qpos"][:14]
        joint_actions, _ = trajectory_to_joint_actions(
            rand_init_action,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )
        env.step(joint_actions)
        obs = env._env.task.get_observation(env._env.physics)
        if not headless:
            vis_img = vis_obs(obs, reset_seed, False, env)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Aloha Dataset Collection", vis_img)
            cv2.waitKey(1)


@click.command()
@click.option(
    "--output_dir", "-o", default=".", help="Directory to save demonstration dataset."
)
@click.option("--motion_type", "-mt", default="random_no_contact", help="Motion type.")
@click.option("--headless", "-h", is_flag=True, help="Run in headless mode.")
@click.option("--seed", type=int, default=20260814, show_default=True)
@click.option(
    "--num_episodes",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of successful episodes to save; 0 keeps collecting.",
)
@click.option(
    "--max_trials",
    type=click.IntRange(min=1),
    default=10000,
    show_default=True,
    help="Fail instead of silently retrying forever.",
)
@click.option(
    "--max_plan_attempts",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Maximum initial-state replans for one trajectory.",
)
def main(
    output_dir: str,
    motion_type: str = "random_no_contact",
    headless: bool = False,
    seed: int = 20260814,
    num_episodes: int = 0,
    max_trials: int = 10000,
    max_plan_attempts: int = 100,
) -> None:
    frequency = 10.0
    dt = 1 / frequency
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    random.seed(seed)
    kin_helper = KinHelper(robot_name="trossen_vx300s")

    curr_vel = np.zeros(6)
    k_p, k_v = 50, 10  # PD control
    acc_lim = 10.0
    vel_lim = 0.04
    env = AlohaEnv("pusht")
    cv2.setNumThreads(1)

    time.sleep(1.0)
    print("Ready!")
    existing_ids = [
        int(path.stem.split("_")[-1])
        for path in Path(output_dir).glob("episode_*.hdf5")
    ]
    episode_id = max(existing_ids, default=-1) + 1
    failed_dir = Path(output_dir) / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    existing_failed_ids = [
        int(path.stem.split("_")[-1]) for path in failed_dir.glob("episode_*.hdf5")
    ]
    failed_episode_id = max(existing_failed_ids, default=-1) + 1
    init_episode_id = episode_id
    stop = False
    is_recording = False
    episode = init_episode()

    # sample to a random init action
    reset_idx = 0
    task_reset(env, seed + reset_idx, kin_helper, k_p, k_v, acc_lim, vel_lim, headless)
    reset_idx += 1

    # Initialize scripted policy components
    workspace_constraints = WorkspaceConstraints(
        x_min=-0.25,
        x_max=0.25,
        y_min=-0.2,
        y_max=0.2,
    )
    motion_planner = MotionPlanner(workspace_constraints)
    current_trajectory = None
    success_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], bool]] = None
    trajectory_step = 0

    # Auto-recording setup
    auto_record = True  # Set to True for automatic data collection
    episode_start_time = None

    # compute world_t_bases
    obs = env._env.task.get_observation(env._env.physics)
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    left_base_mat = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    right_base_mat = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_bases = np.stack([left_base_mat, right_base_mat])
    trial_num = 0
    trajectory_steps = 300

    while not stop:
        # pump obs
        obs = env._env.task.get_observation(env._env.physics)

        # Auto-recording logic
        if auto_record and not is_recording:
            is_recording = True
            episode_start_time = time.monotonic()
            current_trajectory = None  # Force new trajectory generation
            trajectory_step = 0
            print(f"Auto-started recording episode {episode_id}")

        if auto_record and is_recording and episode_start_time is not None:
            # Check if episode duration elapsed
            if trajectory_step >= trajectory_steps:
                # Auto-save episode
                init_pose = env_state_to_mat(episode["env_state"][0])
                final_pose = env_state_to_mat(episode["env_state"][-1])
                actions = np.stack(episode["action"])
                if success_fn is None:
                    raise RuntimeError(
                        "Trajectory completed before a success function was set"
                    )
                episode_success = success_fn(init_pose, final_pose, actions)
                if episode_success:
                    print(f"Episode {episode_id} was successful!")
                    save_episode(episode, output_dir, episode_id)
                    episode_id += 1
                else:
                    print(f"Episode {episode_id} was NOT successful!")
                    save_episode(episode, str(failed_dir), failed_episode_id)
                    failed_episode_id += 1
                trial_num += 1
                print(
                    "Current success rate: ",
                    float(episode_id - init_episode_id) / float(trial_num),
                )
                if num_episodes and episode_id - init_episode_id >= num_episodes:
                    break
                if trial_num >= max_trials:
                    env._env.physics.free()
                    raise RuntimeError(
                        f"Reached max_trials={max_trials} after saving "
                        f"{episode_id - init_episode_id} successful episodes"
                    )
                episode = init_episode()
                task_reset(
                    env,
                    seed + reset_idx,
                    kin_helper,
                    k_p,
                    k_v,
                    acc_lim,
                    vel_lim,
                    headless,
                )
                reset_idx += 1
                is_recording = False
                print(
                    f"One episode takes {time.monotonic() - episode_start_time} seconds"
                )
                episode_start_time = None

        if not headless:
            vis_img = vis_obs(obs, episode_id, is_recording, env, current_trajectory)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Aloha Dataset Collection", vis_img)
            cv2.waitKey(1)

        # Generate scripted policy actions
        puppet_target_state = np.zeros(14)

        # Check if we need to generate a new trajectory
        if current_trajectory is None:
            success = False
            plan_attempt = 0
            while not success and plan_attempt < max_plan_attempts:
                episode = init_episode()
                task_reset(
                    env,
                    seed + reset_idx,
                    kin_helper,
                    k_p,
                    k_v,
                    acc_lim,
                    vel_lim,
                    headless,
                )
                reset_idx += 1
                plan_attempt += 1
                obs = env._env.task.get_observation(env._env.physics)

                # Extract T-shape information
                t_info = extract_t_info(obs["env_state"])

                # Get current arm positions
                current_arm_pos = get_current_arm_positions(
                    obs, kin_helper, world_t_bases
                )

                # Plan new trajectory
                current_trajectory, success, success_fn, trajectory_steps = (
                    motion_planner.plan_episode(t_info, current_arm_pos, motion_type)
                )
            if not success:
                env._env.physics.free()
                raise RuntimeError(
                    f"Failed to plan {motion_type!r} after "
                    f"max_plan_attempts={max_plan_attempts}"
                )
            trajectory_step = 0

        # Get target positions from trajectory
        if current_trajectory is not None and trajectory_step < len(current_trajectory):
            target_xy = current_trajectory[trajectory_step]
        else:
            # Fallback to current position
            target_xy = get_current_arm_positions(obs, kin_helper, world_t_bases)

        # Convert to joint actions using the existing PID control logic
        curr_puppet_joint = obs["qpos"][:14]
        puppet_target_state, target_xy_clip = trajectory_to_joint_actions(
            target_xy,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )

        trajectory_step += 1

        if is_recording:
            episode = update_episode(episode, obs, target_xy_clip, kin_helper, env)

        # execute teleop command
        env.step(puppet_target_state)

    env._env.physics.free()


# %%
if __name__ == "__main__":
    main()
