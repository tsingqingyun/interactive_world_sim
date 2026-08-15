import concurrent.futures
import copy
import gc
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import psutil
import torch
import zarr
import zarr.storage
from filelock import FileLock
from imgaug import augmenters as iaa
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset

register_codecs()


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            # count total steps
            episode_length = file["action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            if "action" not in lowdim_data_dict:
                lowdim_data_dict["action"] = list()
            this_data = file["obs"]["ee_pos"][()]  # (T, 2, 4, 4)
            action_data = np.concatenate(
                [this_data[:, 0, :2, 3], this_data[:, 1, :2, 3]], axis=1
            )  # (T, 4)
            lowdim_data_dict["action"].append(action_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)
                assert imgs[0].shape == (h, w, c)
                rgb_data_dict[key].append(imgs)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h, w, c)
                depth_data_dict[key].append(imgs)

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def load_replay_buffer(
    dataset_dir: str, use_cache: bool, shape_meta: dict
) -> ReplayBuffer:
    replay_buffer = None
    if use_cache:
        cache_info_str = ""
        cache_zarr_path = os.path.join(dataset_dir, f"cache{cache_info_str}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    replay_buffer = _convert_real_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_real_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
        )
    return replay_buffer


def load_keypoint_labels(label_dir: str, split: str) -> ReplayBuffer:
    """Load compact, frame-aligned PushT keypoint labels into memory."""
    label_path = os.path.join(label_dir, f"{split}.npz")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Missing keypoint labels: {label_path}")
    with np.load(label_path) as labels:
        keypoints_uv = labels["keypoints_uv"].astype(np.float32)
        keypoints_visible = labels["keypoints_visible"].astype(bool)
        episode_ends = labels["episode_ends"].astype(np.int64)
    if keypoints_uv.ndim != 3 or keypoints_uv.shape[1:] != (8, 2):
        raise ValueError(f"Expected keypoints_uv shape (T,8,2), got {keypoints_uv.shape}")
    if keypoints_visible.shape != keypoints_uv.shape[:2]:
        raise ValueError(
            "keypoints_visible must match the first two keypoints_uv dimensions"
        )
    return ReplayBuffer(
        {
            "data": {
                "keypoints_uv": keypoints_uv,
                "keypoints_visible": keypoints_visible,
            },
            "meta": {"episode_ends": episode_ends},
        }
    )


class SimAlohaDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = cfg.aug_mode
        if cfg.aug_mode == "img_aug":
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif cfg.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {cfg.aug_mode}")

        train_dir = os.path.join(dataset_dir, "train")
        self.replay_buffer = load_replay_buffer(train_dir, use_cache, shape_meta)

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution
        self.target_dataset_dir = (
            str(cfg.target_dataset_dir)
            if "target_dataset_dir" in cfg and cfg.target_dataset_dir
            else None
        )
        self.keypoint_label_dir = (
            str(cfg.keypoint_label_dir)
            if "keypoint_label_dir" in cfg and cfg.keypoint_label_dir
            else None
        )

        self.target_replay_buffer = None
        self.target_sampler = None
        if self.target_dataset_dir is not None:
            target_train_dir = os.path.join(self.target_dataset_dir, "train")
            self.target_replay_buffer = load_replay_buffer(
                target_train_dir, use_cache, shape_meta
            )
            self._check_episode_alignment(
                self.replay_buffer, self.target_replay_buffer, "prediction target"
            )
            self.target_sampler = self._make_auxiliary_sampler(
                self.target_replay_buffer, horizon, keys=self.rgb_keys
            )

        self.keypoint_replay_buffer = None
        self.keypoint_sampler = None
        if self.keypoint_label_dir is not None:
            self.keypoint_replay_buffer = load_keypoint_labels(
                self.keypoint_label_dir, "train"
            )
            self._check_episode_alignment(
                self.replay_buffer, self.keypoint_replay_buffer, "keypoint labels"
            )
            self.keypoint_sampler = self._make_auxiliary_sampler(
                self.keypoint_replay_buffer,
                horizon,
                keys=["keypoints_uv", "keypoints_visible"],
            )

    @staticmethod
    def _check_episode_alignment(
        source: ReplayBuffer, paired: ReplayBuffer, paired_name: str
    ) -> None:
        if not np.array_equal(source.episode_ends[:], paired.episode_ends[:]):
            raise ValueError(
                f"{paired_name} episode boundaries do not match the input dataset"
            )

    def _make_auxiliary_sampler(
        self, replay_buffer: ReplayBuffer, horizon: int, keys: list[str]
    ) -> SequenceSampler:
        return SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=np.ones((replay_buffer.n_episodes,), dtype=bool),
            goal_sample=self.goal_sample,
            keys=keys,
            skip_frame=self.skip_frame,
        )

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            # the number of episodes in the validation set
            return self.replay_buffer.n_episodes // self.skip_idx
        else:
            return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_dir = os.path.join(self.dataset_dir, "val")
        shape_meta = self.shape_meta
        use_cache = self.use_cache
        val_set.replay_buffer = load_replay_buffer(val_dir, use_cache, shape_meta)
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask

        if self.target_dataset_dir is not None:
            target_val_dir = os.path.join(self.target_dataset_dir, "val")
            val_set.target_replay_buffer = load_replay_buffer(
                target_val_dir, use_cache, shape_meta
            )
            self._check_episode_alignment(
                val_set.replay_buffer,
                val_set.target_replay_buffer,
                "validation prediction target",
            )
            val_set.target_sampler = val_set._make_auxiliary_sampler(
                val_set.target_replay_buffer,
                self.val_horizon,
                keys=self.rgb_keys,
            )

        if self.keypoint_label_dir is not None:
            val_set.keypoint_replay_buffer = load_keypoint_labels(
                self.keypoint_label_dir, "val"
            )
            self._check_episode_alignment(
                val_set.replay_buffer,
                val_set.keypoint_replay_buffer,
                "validation keypoint labels",
            )
            val_set.keypoint_sampler = val_set._make_auxiliary_sampler(
                val_set.keypoint_replay_buffer,
                self.val_horizon,
                keys=["keypoints_uv", "keypoints_visible"],
            )
        return val_set

    def _sample_replay_sequence(
        self,
        idx: int,
        replay_buffer: ReplayBuffer,
        sampler: SequenceSampler,
    ) -> Dict[str, np.ndarray]:
        """Sample the same temporal indices from an aligned replay buffer."""
        if not self.is_val:
            return sampler.sample_sequence(idx)

        epi_idx = idx * self.skip_idx
        epi_start = replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
        epi_end = replay_buffer.episode_ends[epi_idx]
        seq_end = min(epi_end, epi_start + self.val_horizon)
        sample: Dict[str, np.ndarray] = {}
        for key in sampler.keys:
            value = replay_buffer[key][epi_start:seq_end]
            if value.shape[0] < self.val_horizon:
                pad_len = self.val_horizon - value.shape[0]
                value = np.concatenate(
                    [value, np.repeat(value[-1:], pad_len, axis=0)], axis=0
                )
            if key in sampler.keys_to_keep_intermediate:
                inter_frames = value.shape[0] // self.skip_frame
                value_shape = list(value.shape[1:])
                value_shape[0] *= self.skip_frame
                value = value.reshape(
                    inter_frames, self.skip_frame, *value.shape[1:]
                ).reshape(-1, *value_shape)
            else:
                value = value[:: self.skip_frame]
            sample[key] = value
            sample[f"{key}_final"] = value[-1]
        sample["is_early_stop"] = np.asarray(False)
        sample["rel_stop_idx"] = np.asarray(self.val_horizon - 1)
        return sample

    def _target_sample_to_obs(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        target_obs = {}
        for key in self.rgb_keys:
            images = sample[key].astype(np.uint8)
            images = np.moveaxis(images, -1, 1).astype(np.float32) / 255.0
            target_obs[key] = torch.from_numpy(images)
        return target_obs

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()

        # Apply augmentation with 0.2 probability
        apply_aug = np.random.random() < 0.2 if self.aug_mode == "img_aug" else False

        # skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_images = sample[key].astype(np.uint8)
            final_images = sample[f"{key}_final"].astype(np.uint8)

            # Apply augmentation if probability condition is met
            if apply_aug:
                aug_det = self.aug.to_deterministic()
                combined = [*obs_images, final_images]
                # apply the deterministic augmenter separately to each image
                combined_aug = [aug_det.augment_image(img) for img in combined]
                obs_images = np.stack(combined_aug[:-1], axis=0)
                final_images = combined_aug[-1]  # Last image

            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(sample[f"{key}_final"], -1, 0).astype(np.float32) / 1000.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        # action_dim = actions.shape[-1]
        # downsample_horizon = actions.shape[0] // self.skip_frame - 1
        # action_len = downsample_horizon * self.skip_frame
        # action_start = skip_start - self.skip_frame
        # actions = actions[action_start : action_start + action_len]
        # actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        # actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._sample_replay_sequence(idx, self.replay_buffer, self.sampler)
        data = self._sample_to_data(sample)

        if self.target_replay_buffer is not None and self.target_sampler is not None:
            target_sample = self._sample_replay_sequence(
                idx, self.target_replay_buffer, self.target_sampler
            )
            data["target_obs"] = self._target_sample_to_obs(target_sample)

        if self.keypoint_replay_buffer is not None and self.keypoint_sampler is not None:
            keypoint_sample = self._sample_replay_sequence(
                idx, self.keypoint_replay_buffer, self.keypoint_sampler
            )
            data["keypoints_uv"] = torch.from_numpy(
                keypoint_sample["keypoints_uv"].astype(np.float32)
            )
            data["keypoints_visible"] = torch.from_numpy(
                keypoint_sample["keypoints_visible"].astype(bool)
            )
        return data


def test_sim_aloha_dataset() -> None:
    config_path = "configurations/dataset/sim_aloha_dataset.yaml"
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/single_arm_transfer_cube_0407_v3"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_test_0414"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_0407"  # noqa
    cfg.dataset_dir = "data/scripted_sim_aloha_10000"
    cfg.horizon = 10
    cfg.shape_meta.action.shape = (4,)
    cfg.skip_frame = 1
    cfg.skip_idx = 4
    cfg.val_horizon = 200
    cfg.goal_sample = "aggressive"
    cfg.resolution = 128
    dataset = SimAlohaDataset(cfg)
    print(len(dataset))

    p = psutil.Process(os.getpid())

    def rss() -> float:
        return p.memory_info().rss / 1e9

    print(f"START RSS: {rss():.3f} GB")
    k = min(2000, len(dataset))  # enough iterations to see creep
    for i in range(k):
        _ = dataset[i]  # exercises __getitem__
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    cpu_count = os.cpu_count()
    assert cpu_count is not None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=min(cpu_count, 16),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=1,
    )
    i = 0
    for _ in dataloader:
        i += 1
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    val_dataset = dataset.get_validation_dataset()
    print(len(val_dataset))
    for i in range(len(val_dataset)):
        data = val_dataset[i]
    print(data)
    print("validation dataset success!")


if __name__ == "__main__":
    test_sim_aloha_dataset()
