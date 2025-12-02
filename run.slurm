#!/bin/bash
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

# Ensure SLURM log directory exists
mkdir -p slurm

# -------------------------
# 1) Load modules (same style as a1 job)
# -------------------------
module purge
module use /projects/community/modulefiles
module load gcc/10.2.0/openmpi
module load cuda/12.1
# module load fa2/2.4   # not needed here unless you really want it

# -------------------------
# 2) Go to project root
# -------------------------
cd /scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection

# ------------------------
# 3) uv + venv setup
#    - use local .venv as in your a1 script
# -------------------------
# If .venv doesn't exist yet, create a uv-managed venv once
if [ ! -d ".venv" ]; then
  echo "No .venv found, creating a new one with uv venv..."
  uv venv .venv
fi

# Activate venv
source .venv/bin/activate

# Sync dependencies from pyproject.toml
echo "Syncing environment with uv..."
uv sync

# Ensure SSL certs work inside cluster
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# -------------------------
# 4) Project-specific env / paths
# -------------------------
# (adjust if you want a different data root)
export RAVDESS_DATA_ROOT=/scratch/pbm52/emotion-detection-mm/multimodal-dataset
export PYTHONPATH=$PYTHONPATH:/src
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

# Check GPU availability
nvidia-smi

# -------------------------
# 6) (Optional) Run preprocessing on the cluster
# -------------------------
# If you haven't run dataprocessing.py on Amarel yet, uncomment this block once:
# echo "Running RAVDESS preprocessing..."
# uv run python mutlimodal-variant/src/dataprocessing.py \
#   --audio_root multimodal-dataset/Audio_Speech_Actors_01-24 \
#   --video_root multimodal-dataset/Video_Speech_Actors_01-24 \
#   --out_root multimodal-dataset \
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

# Optionally list some outputs
echo "Outputs:"
ls -la checkpoints/ 2>/dev/null || true
ls -la outputs/ 2>/dev/null || true
