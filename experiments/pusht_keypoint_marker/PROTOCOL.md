# IWS MuJoCo PushT: RGB vs RGB + keypoint marker

Frozen on 2026-08-14, before training either comparison arm.

## One-sentence question

With identical MuJoCo states, bimanual actions, episode splits, IWS capacity and
training budget, does drawing the T object's eight known top-surface corners on
the RGB input reduce object-state drift during simultaneous two-arm contact and
100-step autoregressive rollout?

## Facts and representation contract

- Source environment: the repository's `AlohaEnv("pusht")`, top camera, 10 Hz.
- The visual mesh is `t_shape.stl` at XML scale 0.8. Its scaled bounds are
  x=[-0.08, 0.08], y=[-0.14, 0.02], z=[0, 0.032] metres.
- Eight ordered top-surface corners are transformed by the saved 7-DoF T pose,
  projected with MuJoCo's OpenCV-style world-to-camera matrix, centre-cropped,
  and resized to 128 x 128.
- RGB-only uses `obs/images/top_pov`. RGB+marker uses
  `obs/images/top_pov_keypoint_marker`. Both are three-channel images and use
  one IWS view. A separate marker view/channel is prohibited because IWS derives
  latent width and model capacity from the number of views.
- The paired images, actions, T poses, end-effector poses, UV coordinates,
  visibility flags, marker masks and per-arm T-contact normal forces are saved at
  the same simulator step. Marker radius is 3 px; the fixed palette has no
  red-dominant colour.

## Dataset and split

Use one canonical download of the official IWS MuJoCo dataset. With selection
seed `20260814`, sample 1000 training and 200 held-out test episode IDs without
replacement from the 10,000 official training episodes, and retain all 100
official validation episodes. This resource-bounded subset is fixed before
training and is not a claim of full-dataset paper reproduction.

Dataset A is that canonical subset without modification. Dataset B is copied
from A and only `obs/images/top_pov` is replaced by the eight-marker overlay
computed from the aligned `env_state`. Every non-image HDF5 array is hashed
after materialization. A hash mismatch aborts generation. The datasets must not
be collected in separate simulator runs.

## Strong-interaction and rollout definitions

- A frame is a simultaneous-contact frame when both saved arm-to-T normal
  forces are at least 1.0 N. This scale equals approximately the weight of the
  0.1 kg T object and is fixed before looking at model results.
- A strong-interaction segment contains at least three consecutive
  simultaneous-contact frames. No-contact frames are retained as a control.
- Evaluation uses one observed context frame followed by the same ground-truth
  action sequence for both models. Horizons are 10, 30, 60 and 100 steps
  (1, 3, 6 and 10 seconds at 10 Hz).
- All eligible test starts are evaluated; starts are never selected from model
  performance.

## Training control

Run the repository's three IWS stages once for each representation with seed
`20260814`, for six runs total. Architecture, optimizer, batch ordering,
augmentation, step counts, checkpoint selection rule and action normalization
are identical. Stage 1 reconstruction must pass before dynamics training:
held-out T-mask IoU >= 0.95 and clean-region PSNR >= 30 dB. A failed seed is a
failed run, not a reason to increase only that arm's budget.

## Metrics and gates

The red T mask is extracted from each prediction using probe v2
(`R>=100`, `R-G>=90`, `R-B>=90`, then largest connected component). Probe v1
used a 35-level margin and failed before training because connected wood-grain
shadows were included; v2 was calibrated only against MuJoCo geometry labels,
not against either model's output. Before model comparison, this probe
must match MuJoCo geometry segmentation on 200 held-out rendered frames with
mean IoU >= 0.90 and fifth-percentile IoU >= 0.80. If it fails, no object-state
claim is made.

Primary metric: paired per-start area under `1 - T-mask IoU` through 100 steps,
reported separately for strong-interaction and no-contact starts. Lower is
better. Secondary metrics are T-mask centroid error, IoU at each frozen horizon,
time to three consecutive frames below 0.50 IoU, and clean-region PSNR outside
the dilated ground-truth marker mask.

An advantage is claimed only if RGB+marker:

1. reduces strong-interaction primary error by at least 10%;
2. has a paired episode-bootstrap 95% confidence interval wholly below zero;
3. improves or ties at least two of the four frozen horizons; and
4. loses no more than 0.5 dB clean-region PSNR.

Results outside these gates are reported as neutral or negative. Synthetic
smoke tests and projection checks validate plumbing only; they are not the
model comparison.

## Commands

Download the frozen official subset as Dataset A, then derive Dataset B without
stepping the simulator:

```bash
python scripts/download_pusht_official_subset.py \
  --output-root data/mujoco/pusht_pair/rgb \
  --train-episodes 1000 --val-episodes 100 --test-episodes 200 \
  --seed 20260814

PYTHONPATH=external/gym-aloha MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/materialize_pusht_marker_datasets.py \
  --source-root data/mujoco/pusht_pair/rgb \
  --dataset-a data/mujoco/pusht_pair/rgb \
  --dataset-b data/mujoco/pusht_pair/keypoint_marker
```

Train both arms with `dataset=sim_aloha_dataset` and `obs_keys=[top_pov]`.
Change only `dataset.dataset_dir` between Dataset A and Dataset B; all other
overrides must be shared, including `algorithm.action_dim=4` and
`algorithm.num_views=1`.

## Current execution status

The local source, MuJoCo assets, pretrained RGB checkpoint and one official
300-frame episode are available. Its A/B materialization passed the trajectory
hash check, and all six stage entrypoints completed a one-step CPU smoke run.
The remaining official subset is not present locally, outbound download is
currently blocked, and CUDA is unavailable to Torch. Therefore the equal-budget
formal training result remains pending; smoke checkpoints are plumbing evidence
only.
