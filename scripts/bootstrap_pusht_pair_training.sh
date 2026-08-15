#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[IWS bootstrap] failed at line ${LINENO}" >&2' ERR

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

SEED=${IWS_SEED:-20260814}
ENV_NAME=${IWS_ENV_NAME:-iws}
DATA_ROOT=${IWS_DATA_ROOT:-data/mujoco/pusht_pair}
RUN_ROOT=${IWS_RUN_ROOT:-outputs/pusht_pair_seed${SEED}}
MIN_FREE_GIB=${IWS_MIN_FREE_GIB:-80}
MIN_RAM_GIB=${IWS_MIN_RAM_GIB:-64}

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONHASHSEED=$SEED
export WANDB_MODE=${WANDB_MODE:-offline}

echo "[IWS bootstrap] repository: $REPO_ROOT"
echo "[IWS bootstrap] seed: $SEED"
echo "[IWS bootstrap] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

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

AVAILABLE_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')
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
conda activate "$ENV_NAME"

uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126/
uv pip install -e .

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access CUDA after environment setup")
print(
    "[IWS bootstrap] torch CUDA ready:",
    torch.__version__,
    torch.cuda.get_device_name(0),
)
PY

RGB_ROOT="$DATA_ROOT/rgb"
MARKER_ROOT="$DATA_ROOT/keypoint_marker"

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
    local load_checkpoint=${4:-}
    local output_dir="$RUN_ROOT/$pair/stage$stage"

    if [[ -f "$output_dir/.complete" ]]; then
        echo "[IWS bootstrap] reusing completed $pair Stage $stage"
        LAST_CHECKPOINT=$(latest_checkpoint "$output_dir")
        return
    fi
    if [[ -e "$output_dir" ]]; then
        echo "Incomplete output already exists at $output_dir; inspect it before retrying." >&2
        return 1
    fi

    local args=(
        scripts/run_seeded_iws.py --seed "$SEED" main.py
        "+name=pusht_${pair}_stage${stage}"
        algorithm=latent_world_model
        experiment=exp_latent_dyn
        dataset=sim_aloha_dataset
        "dataset.dataset_dir=$dataset_root"
        algorithm.latent_dim=512
        algorithm.action_dim=4
        "algorithm.training_stage=$stage"
        wandb.mode=offline
        wandb.entity=local
        "hydra.run.dir=$output_dir"
    )

    case "$stage" in
        1)
            args+=(
                dataset.horizon=1 dataset.val_horizon=1
                experiment.training.batch_size=1
                experiment.training.max_steps=1000005
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
                experiment.training.max_steps=1000005
                experiment.training.log_every_n_steps=100
                experiment.validation.limit_batch=1.0
                experiment.validation.batch_size=2
                experiment.validation.val_every_n_step=30000
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
                experiment.training.max_steps=1000005
                experiment.training.log_every_n_steps=100
                experiment.validation.limit_batch=1.0
                experiment.validation.batch_size=2
                experiment.validation.val_every_n_step=30000
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

    echo "[IWS bootstrap] starting $pair Stage $stage"
    python "${args[@]}"
    touch "$output_dir/.complete"
    LAST_CHECKPOINT=$(latest_checkpoint "$output_dir")
    echo "[IWS bootstrap] completed $pair Stage $stage: $LAST_CHECKPOINT"
}

run_stage rgb 1 "$RGB_ROOT"
RGB_STAGE1=$LAST_CHECKPOINT
run_stage marker 1 "$MARKER_ROOT"
MARKER_STAGE1=$LAST_CHECKPOINT
run_stage rgb 2 "$RGB_ROOT" "$RGB_STAGE1"
RGB_STAGE2=$LAST_CHECKPOINT
run_stage marker 2 "$MARKER_ROOT" "$MARKER_STAGE1"
MARKER_STAGE2=$LAST_CHECKPOINT
run_stage rgb 3 "$RGB_ROOT" "$RGB_STAGE2"
run_stage marker 3 "$MARKER_ROOT" "$MARKER_STAGE2"

echo "[IWS bootstrap] all six training runs completed"
echo "[IWS bootstrap] outputs: $RUN_ROOT"
