#!/bin/bash
#SBATCH --job-name=ravdess-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[015-016,021,025-027,028],gpuk[005-006]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_grid_%j.out
#SBATCH --error=slurm/slurm_grid_%j.err

set -xeuo pipefail

# -------------------------
# 0) Directories & grid setup
# -------------------------
PROJECT_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection
GRID_ROOT=${PROJECT_ROOT}/grid_sweep_results

mkdir -p "${PROJECT_ROOT}/slurm"
mkdir -p "${GRID_ROOT}"

# Small grid: adjust as needed
LRS=(0.0005 0.001 0.002)
MODEL_DROPS=(0.0 0.1)
MODALITY_DROPS=(0.0 0.05)

# -------------------------
# 1) Modules
# -------------------------
module purge
module use /projects/community/modulefiles
module load gcc/10.2.0/openmpi
module load cuda/12.1

# -------------------------
# 2) Go to project root
# -------------------------
cd "${PROJECT_ROOT}"

# ------------------------
# 3) uv + venv setup
# -------------------------
if [ ! -d ".venv" ]; then
  echo "No .venv found, creating a new one with uv venv..."
  uv venv .venv
fi

source .venv/bin/activate

echo "Syncing environment with uv..."
uv sync

# SSL certs
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# -------------------------
# 4) Env vars
# -------------------------
export RAVDESS_DATA_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-dataset
export PYTHONPATH="${PYTHONPATH:-}:src"
export CUDA_VISIBLE_DEVICES=0

echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Working Dir:     $(pwd)"
echo "RAVDESS Root:    $RAVDESS_DATA_ROOT"

nvidia-smi

# -------------------------
# 5) Grid loop
# -------------------------
for LR in "${LRS[@]}"; do
  for MDROP in "${MODALITY_DROPS[@]}"; do
    for MD in "${MODEL_DROPS[@]}"; do

      # Make a tag safe for filenames (replace '.' with 'p')
      LR_TAG=${LR/./p}
      MD_TAG=${MD/./p}
      MMD_TAG=${MDROP/./p}
      TAG="lr${LR_TAG}_drop${MD_TAG}_mDrop${MMD_TAG}"

      EXP_NAME="ravdess_sweep_${TAG}"
      echo "============================================================"
      echo "Starting sweep run: ${TAG}"
      echo "  LR               = ${LR}"
      echo "  model.dropout    = ${MD}"
      echo "  modality_dropout = ${MDROP}"
      echo "  experiment.name  = ${EXP_NAME}"
      echo "============================================================"

      # -------------------------
      # 5a) Run training with overrides
      # -------------------------
      uv run python src/train.py \
        experiment.name="${EXP_NAME}" \
        dataset.name=ravdess \
        dataset.data_dir="${RAVDESS_DATA_ROOT}" \
        dataset.modalities='[audio, video]' \
        dataset.num_classes=8 \
        \
        model.output_dim=256 \
        model.hidden_dim=512 \
        model.dropout="${MD}" \
        \
        model.encoders.audio.hidden_dim=512 \
        model.encoders.audio.output_dim=256 \
        model.encoders.audio.num_layers=3 \
        model.encoders.audio.dropout=0.1 \
        \
        model.encoders.video.hidden_dim=512 \
        model.encoders.video.output_dim=256 \
        model.encoders.video.dropout=0.1 \
        \
        training.max_epochs=80 \
        training.early_stopping_patience=15 \
        training.learning_rate="${LR}" \
        training.augmentation.modality_dropout="${MDROP}"

      echo "Finished training for ${TAG}"

      # -------------------------
      # 5b) Collect results into grid_sweep_results/<TAG>/
      # -------------------------
      SAVE_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
      RUN_OUT="${GRID_ROOT}/${TAG}"

      mkdir -p "${RUN_OUT}"

      # Copy results.json if present
      if [ -f "${SAVE_DIR}/results.json" ]; then
        cp "${SAVE_DIR}/results.json" "${RUN_OUT}/"
      fi

      # Copy confusion matrix artifacts if present
      if [ -f "${SAVE_DIR}/confusion_matrix.png" ]; then
        cp "${SAVE_DIR}/confusion_matrix.png" "${RUN_OUT}/"
      fi
      if [ -f "${SAVE_DIR}/confusion_matrix.npy" ]; then
        cp "${SAVE_DIR}/confusion_matrix.npy" "${RUN_OUT}/"
      fi

      # Copy best checkpoint if present
      if [ -f "${SAVE_DIR}/best.ckpt" ]; then
        cp "${SAVE_DIR}/best.ckpt" "${RUN_OUT}/"
      fi

      # Copy latest metrics.csv from CSVLogger (if any)
      # Typically: outputs/<EXP_NAME>/csv_logs/version_*/metrics.csv
      METRICS_SRC_DIR=$(ls -td "${SAVE_DIR}/csv_logs"/version_* 2>/dev/null | head -n 1 || true)
      if [ -n "${METRICS_SRC_DIR}" ] && [ -f "${METRICS_SRC_DIR}/metrics.csv" ]; then
        cp "${METRICS_SRC_DIR}/metrics.csv" "${RUN_OUT}/metrics.csv"
      fi

      # Write a small manifest of hyperparams for this run
      cat > "${RUN_OUT}/hyperparams.txt" <<EOF
experiment.name = ${EXP_NAME}
learning_rate   = ${LR}
model.dropout   = ${MD}
modality_dropout= ${MDROP}
model.output_dim= 256
model.hidden_dim= 512
audio.hidden_dim= 512
audio.output_dim= 256
audio.num_layers= 3
video.hidden_dim= 512
video.output_dim= 256
EOF

      echo "Collected results for ${TAG} into ${RUN_OUT}"
      echo
    done
  done
done

echo "All grid sweep runs completed at: $(date)"
