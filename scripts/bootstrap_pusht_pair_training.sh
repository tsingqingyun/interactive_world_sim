#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[IWS bootstrap] failed at line ${LINENO}" >&2' ERR

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

SEED=${IWS_SEED:-20260814}
ENV_NAME=${IWS_ENV_NAME:-iws}
DATA_ROOT=${IWS_DATA_ROOT:-/data/jingchen/interactive_world_sim/pusht_pair}
STORAGE_ROOT=${IWS_STORAGE_ROOT:-/data/jingchen/interactive_world_sim}
RUN_ROOT=${IWS_RUN_ROOT:-$STORAGE_ROOT/outputs/pusht_pair_quick_seed${SEED}}
RGB_GPU=${IWS_RGB_GPU:-0}
MARKER_GPU=${IWS_MARKER_GPU:-2}
EXPLICIT_GPU=${IWS_EXPLICIT_GPU:-4}
MARKER_TO_RGB_GPU=${IWS_MARKER_TO_RGB_GPU:-5}
WANDB_RUN_MODE=${IWS_WANDB_MODE:-online}
WANDB_ENTITY=${IWS_WANDB_ENTITY:-}
WANDB_PROJECT=${IWS_WANDB_PROJECT:-interactive-world-sim}
MIN_FREE_GIB=${IWS_MIN_FREE_GIB:-80}
MIN_RAM_GIB=${IWS_MIN_RAM_GIB:-64}

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONHASHSEED=$SEED
export WANDB_MODE=$WANDB_RUN_MODE
export WANDB_DIR=${IWS_WANDB_DIR:-$RUN_ROOT}

echo "[IWS bootstrap] repository: $REPO_ROOT"
echo "[IWS bootstrap] seed: $SEED"
echo "[IWS bootstrap] data: $DATA_ROOT"
echo "[IWS bootstrap] GPUs: rgb=$RGB_GPU marker=$MARKER_GPU explicit=$EXPLICIT_GPU marker_to_rgb=$MARKER_TO_RGB_GPU"

GPU_LIST=("$RGB_GPU" "$MARKER_GPU" "$EXPLICIT_GPU" "$MARKER_TO_RGB_GPU")
if [[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -ne 4 ]]; then
    echo "The four training pipelines require four different GPUs." >&2
    exit 1
fi
if [[ "$WANDB_RUN_MODE" == online && -z "$WANDB_ENTITY" ]]; then
    echo "Set IWS_WANDB_ENTITY to the logged-in W&B entity." >&2
    exit 1
fi

command -v conda >/dev/null || {
    echo "conda is required; install Miniforge first." >&2
    exit 1
}
command -v mamba >/dev/null || {
    echo "mamba is required by the repository installation procedure." >&2
    exit 1
}
command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable; install the NVIDIA driver first." >&2
    exit 1
}

mkdir -p "$RUN_ROOT"
AVAILABLE_KIB=$(df -Pk "$STORAGE_ROOT" | awk 'NR == 2 {print $4}')
REQUIRED_KIB=$((MIN_FREE_GIB * 1024 * 1024))
if (( AVAILABLE_KIB < REQUIRED_KIB )); then
    echo "At least ${MIN_FREE_GIB} GiB free disk is required; found $((AVAILABLE_KIB / 1024 / 1024)) GiB." >&2
    exit 1
fi

TOTAL_RAM_KIB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
REQUIRED_RAM_KIB=$((MIN_RAM_GIB * 1024 * 1024))
if (( TOTAL_RAM_KIB < REQUIRED_RAM_KIB )); then
    echo "At least ${MIN_RAM_GIB} GiB RAM is required to build the in-memory replay buffers; found $((TOTAL_RAM_KIB / 1024 / 1024)) GiB." >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader
git submodule update --init --recursive

CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[IWS bootstrap] creating conda environment $ENV_NAME"
    mamba env create -n "$ENV_NAME" -f conda_env.yaml
else
    echo "[IWS bootstrap] reusing conda environment $ENV_NAME"
fi
# Conda package activation hooks may probe unset toolchain variables. Temporarily
# disable nounset while those third-party hooks run, then restore strict mode.
set +u
conda activate "$ENV_NAME"
set -u

uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128/
uv pip install -e .

for gpu in "${GPU_LIST[@]}"; do
CUDA_VISIBLE_DEVICES="$gpu" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access CUDA after environment setup")
device = torch.device("cuda:0")
# Run an actual kernel so an unsupported GPU architecture fails before data
# download or a long training job begins.
result = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
torch.cuda.synchronize(device)
if not torch.equal(result.cpu(), torch.full((2, 2), 2.0)):
    raise SystemExit("Torch CUDA kernel smoke test returned an invalid result")
print(
    "[IWS bootstrap] torch CUDA ready:",
    torch.__version__,
    torch.version.cuda,
    torch.cuda.get_device_name(device),
)
PY
done

RGB_ROOT="$DATA_ROOT/rgb"
MARKER_ROOT="$DATA_ROOT/keypoint_marker"
KEYPOINT_LABEL_ROOT="$DATA_ROOT/keypoint_labels"

python scripts/download_pusht_official_subset.py \
    --output-root "$RGB_ROOT" \
    --train-episodes 1000 \
    --val-episodes 100 \
    --test-episodes 200 \
    --seed "$SEED"

if [[ ! -f "$MARKER_ROOT/paired_manifest.json" ]]; then
    PYTHONPATH=external/gym-aloha python \
        scripts/materialize_pusht_marker_datasets.py \
        --source-root "$RGB_ROOT" \
        --dataset-a "$RGB_ROOT" \
        --dataset-b "$MARKER_ROOT" \
        --overwrite
else
    echo "[IWS bootstrap] paired marker dataset already materialized"
fi

if [[ ! -f "$KEYPOINT_LABEL_ROOT/manifest.json" ]]; then
    PYTHONPATH=external/gym-aloha python \
        scripts/materialize_pusht_keypoint_labels.py \
        --source-root "$RGB_ROOT" \
        --output-root "$KEYPOINT_LABEL_ROOT"
else
    echo "[IWS bootstrap] keypoint coordinate labels already materialized"
fi

latest_checkpoint() {
    local stage_dir=$1
    local checkpoint
    checkpoint=$(
        find "$stage_dir/checkpoints" -maxdepth 1 -type f -name '*.ckpt' \
            -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
    )
    if [[ -z "$checkpoint" ]]; then
        echo "No checkpoint found under $stage_dir/checkpoints" >&2
        return 1
    fi
    realpath "$checkpoint"
}

LAST_CHECKPOINT=
run_stage() {
    local pair=$1
    local stage=$2
    local dataset_root=$3
    local gpu=$4
    local load_checkpoint=${5:-}
    shift 5
    local extra_args=("$@")
    local output_dir="$RUN_ROOT/$pair/stage$stage"
    local resume_checkpoint=

    if [[ -f "$output_dir/.complete" ]]; then
        echo "[IWS bootstrap] reusing completed $pair Stage $stage"
        LAST_CHECKPOINT=$(latest_checkpoint "$output_dir")
        return
    fi
    if [[ -e "$output_dir" ]]; then
        if resume_checkpoint=$(latest_checkpoint "$output_dir"); then
            echo "[IWS bootstrap] resuming $pair Stage $stage: $resume_checkpoint"
        else
            echo "Incomplete output has no checkpoint at $output_dir; inspect it before retrying." >&2
            return 1
        fi
    fi

    local args=(
        scripts/run_seeded_iws.py --seed "$SEED" main.py
        "+name=pusht_${pair}_quick_stage${stage}"
        algorithm=latent_world_model
        experiment=exp_latent_dyn
        dataset=sim_aloha_dataset
        "dataset.dataset_dir=$dataset_root"
        algorithm.latent_dim=512
        algorithm.action_dim=4
        "algorithm.training_stage=$stage"
        "wandb.mode=$WANDB_RUN_MODE"
        "wandb.entity=$WANDB_ENTITY"
        "wandb.project=$WANDB_PROJECT"
        "hydra.run.dir=$output_dir"
    )

    case "$stage" in
        1)
            args+=(
                dataset.horizon=1 dataset.val_horizon=1
                experiment.training.batch_size=1
                experiment.training.max_steps=30000
                experiment.training.log_every_n_steps=100
                experiment.validation.limit_batch=1.0
                experiment.validation.batch_size=10
                experiment.validation.val_every_n_step=6000
            )
            ;;
        2)
            args+=(
                dataset.horizon=10 dataset.val_horizon=100
                experiment.training.batch_size=4
                experiment.training.max_steps=50000
                experiment.training.log_every_n_steps=100
                experiment.validation.limit_batch=1.0
                experiment.validation.batch_size=2
                experiment.validation.val_every_n_step=10000
                experiment.training.checkpointing.every_n_train_steps=10000
                experiment.training.data.num_workers=4
                experiment.validation.data.num_workers=4
                algorithm.noise_scheduler.loss_weighting=uniform
                algorithm.sampling_strategy=terminal_only
            )
            ;;
        3)
            args+=(
                dataset.horizon=1 dataset.val_horizon=100
                experiment.training.batch_size=16
                experiment.training.max_steps=20000
                experiment.training.log_every_n_steps=100
                experiment.validation.limit_batch=1.0
                experiment.validation.batch_size=2
                experiment.validation.val_every_n_step=10000
                experiment.training.checkpointing.every_n_train_steps=10000
                experiment.training.data.num_workers=4
                experiment.validation.data.num_workers=4
                algorithm.noise_scheduler.loss_weighting=uniform
                algorithm.sampling_strategy=terminal_only
            )
            ;;
        *)
            echo "Invalid training stage: $stage" >&2
            return 1
            ;;
    esac

    if [[ -n "$load_checkpoint" ]]; then
        args+=("algorithm.load_ae='$load_checkpoint'")
    fi
    if [[ -n "$resume_checkpoint" ]]; then
        args+=("load=$resume_checkpoint")
    fi
    args+=("${extra_args[@]}")

    echo "[IWS bootstrap] starting $pair Stage $stage on physical GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python "${args[@]}"
    touch "$output_dir/.complete"
    LAST_CHECKPOINT=$(latest_checkpoint "$output_dir")
    echo "[IWS bootstrap] completed $pair Stage $stage: $LAST_CHECKPOINT"
}

run_pair() {
    local pair=$1
    local dataset_root=$2
    local gpu=$3
    shift 3
    local extra_args=("$@")
    local stage1_checkpoint
    local stage2_checkpoint

    LAST_CHECKPOINT=
    run_stage "$pair" 1 "$dataset_root" "$gpu" "" "${extra_args[@]}"
    stage1_checkpoint=$LAST_CHECKPOINT
    run_stage "$pair" 2 "$dataset_root" "$gpu" "$stage1_checkpoint" "${extra_args[@]}"
    stage2_checkpoint=$LAST_CHECKPOINT
    run_stage "$pair" 3 "$dataset_root" "$gpu" "$stage2_checkpoint" "${extra_args[@]}"
}

run_pair rgb "$RGB_ROOT" "$RGB_GPU" &
RGB_PID=$!
run_pair marker "$MARKER_ROOT" "$MARKER_GPU" &
MARKER_PID=$!
run_pair marker_explicit "$MARKER_ROOT" "$EXPLICIT_GPU" \
    "dataset.keypoint_label_dir=$KEYPOINT_LABEL_ROOT" \
    algorithm.keypoint_loss_weight=0.1 &
EXPLICIT_PID=$!
run_pair marker_to_rgb "$MARKER_ROOT" "$MARKER_TO_RGB_GPU" \
    "dataset.target_dataset_dir=$RGB_ROOT" &
MARKER_TO_RGB_PID=$!

PAIR_PIDS=("$RGB_PID" "$MARKER_PID" "$EXPLICIT_PID" "$MARKER_TO_RGB_PID")
while (( ${#PAIR_PIDS[@]} > 0 )); do
    FINISHED_PID=
    if wait -n -p FINISHED_PID "${PAIR_PIDS[@]}"; then
        STATUS=0
    else
        STATUS=$?
    fi
    REMAINING_PIDS=()
    for pid in "${PAIR_PIDS[@]}"; do
        [[ "$pid" == "$FINISHED_PID" ]] || REMAINING_PIDS+=("$pid")
    done
    PAIR_PIDS=("${REMAINING_PIDS[@]}")
    if (( STATUS != 0 )); then
        echo "Training pipeline $FINISHED_PID failed with status $STATUS" >&2
        if (( ${#PAIR_PIDS[@]} > 0 )); then
            kill "${PAIR_PIDS[@]}" 2>/dev/null || true
            wait "${PAIR_PIDS[@]}" 2>/dev/null || true
        fi
        exit "$STATUS"
    fi
done

echo "[IWS bootstrap] all twelve training runs completed"
echo "[IWS bootstrap] outputs: $RUN_ROOT"
