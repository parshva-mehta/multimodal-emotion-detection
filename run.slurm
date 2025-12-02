#!/bin/bash
#SBATCH --job-name=multimodal-emotion-detection
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

# -------------------------
# 1) Load modules
# -------------------------
module purge
module load anaconda3/2024.05

# -------------------------
# 2) Activate conda env
# -------------------------
source ~/.bashrc
conda env create -f environment.yml -n multimodal-emotion-detection
conda activate multimodal-emotion-detection

# -------------------------
# 3) Go to project root
# -------------------------
cd /scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection


# -------------------------
# 4) Print job info
# -------------------------
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE"
echo "Working Dir:   $(pwd)"
echo "Python:        $(which python)"
python --version
echo "PYTHONPATH:    $PYTHONPATH"
echo "Start Time:    $(date)"

# -------------------------
# 5) (Optional) Preprocess RAVDESS on Amarel
# -------------------------
# Uncomment this block if you need to regenerate the .npy splits on the cluster.
# echo "Running RAVDESS preprocessing..."
# python mutlimodal-variant/src/dataprocessing.py \
#   --audio_root multimodal-dataset/Audio_Speech_Actors_01-24 \
#   --video_root multimodal-dataset/Video_Speech_Actors_01-24 \
#   --out_root multimodal-dataset \
#   --val_size 0.1 \
#   --test_size 0.1

# -------------------------
# 6) Run training
# -------------------------
echo "Starting multimodal RAVDESS training..."
python /scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection/src/train.py \
  experiment.name=ravdess_audio_video_baseline \
  dataset.name=ravdess \
  dataset.data_dir=multimodal-dataset \
  dataset.modalities='[audio, video]' \
  dataset.num_classes=8

echo "End Time: $(date)"
