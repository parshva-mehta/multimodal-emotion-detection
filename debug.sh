#!/bin/bash
#SBATCH --job-name=ravdess-debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[015-016,021,025-027,028],gpuk[005-006]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu

# logs go into debugs/
#SBATCH --output=/scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection/debugs/slurm_%j.out
#SBATCH --error=/scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection/debugs/slurm_%j.err

set -xeuo pipefail

# -----------------------------------------------------------------------------
# 0) Directories
# -----------------------------------------------------------------------------
PROJECT_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection
DEBUG_DIR=${PROJECT_ROOT}/debugs

mkdir -p "${DEBUG_DIR}"

# -----------------------------------------------------------------------------
# 1) Modules
# -----------------------------------------------------------------------------
module purge
module use /projects/community/modulefiles
module load gcc/10.2.0/openmpi
module load cuda/12.1

# -----------------------------------------------------------------------------
# 2) Go to project root
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# 3) uv + venv setup
# -----------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "No .venv found, creating a new one with uv venv..."
  uv venv .venv
fi

source .venv/bin/activate

echo "Syncing environment with uv..."
uv sync

export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# -----------------------------------------------------------------------------
# 4) Env vars
# -----------------------------------------------------------------------------
export RAVDESS_DATA_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-dataset
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0

echo "Job ID:      $SLURM_JOB_ID"
echo "Job Name:    $SLURM_JOB_NAME"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Working Dir: $(pwd)"
echo "RAVDESS Root:${RAVDESS_DATA_ROOT}"

nvidia-smi

# -----------------------------------------------------------------------------
# 5) Run debug.py via uv
# -----------------------------------------------------------------------------
echo "Starting debug run..."
uv run python src/debug.py \
  experiment.name=ravdess_debug \
  dataset.name=ravdess \
  dataset.data_dir="${RAVDESS_DATA_ROOT}" \
  dataset.modalities='[audio, video]'

echo "Debug job completed at: $(date)"
