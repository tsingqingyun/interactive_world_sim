# Paired IWS training runbook

Dataset A and B contain identical episode IDs, states, actions and split
membership. Both expose the image as `obs/images/top_pov`; only the pixel values
inside the eight marker disks differ. Both runs use the repository's original
`dataset=sim_aloha_dataset`; only `dataset.dataset_dir` changes.

Use the same seed and the same command overrides for each A/B pair. The seeded
launcher initializes Python, NumPy and Torch before executing the unchanged IWS
`main.py` entrypoint.

Materialize the frozen subset first:

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

## Stage 1

Run once with `DATASET_ROOT=data/mujoco/pusht_pair/rgb` and once with
`DATASET_ROOT=data/mujoco/pusht_pair/keypoint_marker`:

```bash
python scripts/run_seeded_iws.py --seed 20260814 main.py \
  +name=pusht_PAIR_stage1 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_aloha_dataset \
  dataset.dataset_dir=DATASET_ROOT \
  dataset.horizon=1 dataset.val_horizon=1 \
  experiment.training.batch_size=1 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1 \
  wandb.mode=offline wandb.entity=local
```

## Stage 2

Use the checkpoint from the matching representation's Stage 1 run. Never load
the RGB autoencoder into the marker arm or vice versa.

```bash
python scripts/run_seeded_iws.py --seed 20260814 main.py \
  +name=pusht_PAIR_stage2 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_aloha_dataset \
  dataset.dataset_dir=DATASET_ROOT \
  dataset.horizon=10 dataset.val_horizon=100 \
  experiment.training.batch_size=4 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  "algorithm.load_ae='STAGE1_CHECKPOINT'" \
  algorithm.training_stage=2 \
  wandb.mode=offline wandb.entity=local
```

## Stage 3

Use the checkpoint from the matching representation's Stage 2 run.

```bash
python scripts/run_seeded_iws.py --seed 20260814 main.py \
  +name=pusht_PAIR_stage3 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_aloha_dataset \
  dataset.dataset_dir=DATASET_ROOT \
  dataset.horizon=1 dataset.val_horizon=100 \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  "algorithm.load_ae='STAGE2_CHECKPOINT'" \
  algorithm.training_stage=3 \
  wandb.mode=offline wandb.entity=local
```

This protocol runs each representation/stage pair once with seed `20260814`, as
requested: A1, B1, A2, B2, A3 and B3. Checkpoint selection must use the same
validation rule in both arms. The outer double quotes above are required when a
Lightning checkpoint filename contains `=`.
