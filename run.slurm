#!/bin/bash
set -xeuo pipefail
#SBATCH --job-name=ravdess-mm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[015-016,021,025-027,028],gpuk[005-006]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err

echo "SLURM script started at: $(date)"

# Ensure SLURM log directory exists
mkdir -p slurm

# -------------------------
# 1) Load modules
# -------------------------
module purge
module use /projects/community/modulefiles
module load gcc/10.2.0/openmpi
module load cuda/12.1

# -------------------------
# 2) Go to project root
# -------------------------
cd /scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection

# -------------------------
# 3) uv + venv setup
# -------------------------
if [ ! -d ".venv" ]; then
  echo "No .venv found, creating a new one with uv venv..."
  uv venv .venv
fi

source .venv/bin/activate

echo "Syncing environment with uv..."
uv sync

# Ensure SSL certs work inside cluster
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# -------------------------
# 4) Project-specific env / paths
# -------------------------
# adjust this if your dataset is actually under a different tree
export RAVDESS_DATA_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-dataset
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=0

# -------------------------
# 5) Print job information
# -------------------------
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Node:            $SLURM_NODELIST"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Working Dir:     $(pwd)"
echo "Start Time:      $(date)"
echo "Python Path:     $PYTHONPATH"
echo "RAVDESS Root:    $RAVDESS_DATA_ROOT"

nvidia-smi

# -------------------------
# 6) (Optional) Run preprocessing on the cluster
# -------------------------
# echo "Running RAVDESS preprocessing..."
# uv run python src/dataprocessing.py \
#   --audio_root "$RAVDESS_DATA_ROOT/Audio_Speech_Actors_01-24" \
#   --video_root "$RAVDESS_DATA_ROOT/Video_Speech_Actors_01-24" \
#   --out_root "$RAVDESS_DATA_ROOT" \
#   --val_size 0.1 \
#   --test_size 0.1

# -------------------------
# 7) Run training (via uv)
# -------------------------
echo "Starting RAVDESS multimodal training..."
uv run python src/train.py \
  experiment.name=ravdess_audio_video_baseline \
  dataset.name=ravdess \
  dataset.data_dir="$RAVDESS_DATA_ROOT" \
  dataset.modalities='[audio, video]' \
  dataset.num_classes=8

echo "Job completed at: $(date)"

echo "Outputs:"
ls -la checkpoints/ 2>/dev/null || true
ls -la outputs/ 2>/dev/null || true
