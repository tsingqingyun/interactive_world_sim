# Execution report: PushT RGB vs keypoint-marker IWS

Date: 2026-08-15

## Conclusion

Dataset A/B materialization and all six Stage 1/2/3 training entrypoints are
runnable and validated on one official 300-frame episode, but no RGB-vs-marker
model result is claimed: the frozen official subset is not downloaded and CUDA
is currently unavailable to Torch.

## Earliest issues found

1. The collector's function named `visualize_t_keypoints_on_camera` drew the
   dense 2 mm contact outline, not the eight saved T keypoints.
2. It forced those points to world `z=0.02 m`. The actual STL top is local
   `z=0.032 m` after the XML's 0.8 scale and must then be transformed by the
   object's 7-DoF pose.
3. Marker visualization was preview-only, so RGB and marker observations could
   not be trained from the same saved states.
4. A second marker view would increase IWS latent width and model capacity,
   confounding the requested representation comparison.
5. Separate simulator collection would not guarantee identical trajectories.
   A deterministic post-processing pass over one canonical dataset is required.
6. Marker-specific dataset config filenames change Hydra's dataset registry key
   and fail before training. Both arms must retain the upstream
   `dataset=sim_aloha_dataset` config and differ only in `dataset.dataset_dir`.

## Minimal changes

- One geometry/projection implementation now owns the ordered eight points,
  world transform, OpenCV projection, centre crop, visibility and marker draw.
- A materializer copies each canonical HDF5 episode into A/B, modifies only B's
  `obs/images/top_pov`, and aborts if the shared trajectory hash changes.
- Both roots use the original `sim_aloha_dataset` config and the same
  three-channel `top_pov` key; only pixel values and dataset root differ.
- A deterministic downloader freezes 1000 train, 100 validation and 200 held-out
  test episodes with selection seed `20260814`.
- A seeded launcher initializes Python, NumPy and Torch before the unchanged
  IWS `main.py` entrypoint.
- The only training-framework compatibility change permits `num_workers=0` by
  omitting PyTorch's multiprocessing-only `prefetch_factor` in that case.
- A paired rollout evaluator checks identical targets/contact metadata before
  computing strong-contact and no-contact metrics through 100 steps.

## Measured validation evidence

- Eight-point projection on 200 deterministic MuJoCo initial states:
  corner-to-geometry-mask distance mean `0.632 px`, p95 `1.369 px`.
- Visibility: `1599/1600` projected points remained in the 128 x 128 crop; the
  one naturally cropped point is represented by its saved visibility flag.
- Frozen red-object probe v2 versus MuJoCo geometry segmentation: mean IoU
  `0.9110`, fifth percentile `0.8834`, minimum `0.8672`. The original probe v1
  failed at mean `0.5331` because wood-grain shadows joined the T component.
- Real one-frame MuJoCo HDF5 fixture: both images are `(1,128,128,3) uint8`, UV
  is `(1,8,2) float32`, visibility and mask are uint8, contact force is
  `(1,2) float32`; clean and marker images are exactly equal outside the saved
  marker mask.
- Both dataset configurations converted the same two fixture states to
  `(2,128,128,3) uint8`, with bit-identical `(2,4)` action arrays and one view.
- Official episode 0 probe: `300` frames, `2400/2400` keypoints visible, and
  `69,600` marker-mask pixels. Dataset A and B have the identical trajectory
  SHA-256 `a561d77ddb0e165c4eed0f267cd23ccdc3a61ab00bb9e8cfb81c11b133d72221`.
- Stage 1, Stage 2 and Stage 3 each completed one CPU optimization step for both
  A and B. All six checkpoints report `global_step=1`, 780 state-dict entries
  and the expected matching stage-to-stage checkpoint dependency.
- The one-episode smoke fixture deliberately reuses the same episode for train
  and validation, so its losses and checkpoints are not comparison evidence.

## Unchanged components

IWS encoder, decoder, latent dynamics, optimizer, loss, action representation,
MuJoCo XML/mesh and model capacity were not changed for this comparison. No
keypoint loss, prediction head, extra input channel or second camera view was
added. The prior CPU-safe attention fix and collector/evaluator work remain
separate changes.

## Still unresolved

- Only one official episode is locally available. The frozen 1000/100/200 subset
  cannot be downloaded while outbound network access is disabled.
- `torch.cuda.is_available()` is false, so the six equal-budget 1,000,005-step
  runs of the 33.4M-parameter model are not feasible in the current session.
- Consequently there are no paired model predictions, bootstrap interval or
  defensible marker-advantage result yet. Projection/HDF5 fixtures are plumbing
  evidence only.

## Formal next run

Download and materialize the frozen 1000/100/200 split from `PROTOCOL.md`, then
run the six commands in `TRAINING_RUNBOOK.md`. Export paired prediction NPZ
files with `predicted_rgb`, shared clean `target_rgb`, `contact_force`, and
`marker_mask`, then run:

```bash
python scripts/evaluate_pusht_rollout_comparison.py \
  --rgb outputs/pusht_rgb/rollouts.npz \
  --marker outputs/pusht_marker/rollouts.npz \
  --output reproduction_artifacts/pusht_keypoint_comparison/report.json
```
