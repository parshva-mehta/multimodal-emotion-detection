#!/bin/bash
#
#SBATCH -J ravdess_fusion           # Job name
#SBATCH -p main                     # Partition (queue) to use
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 8                        # CPU cores per task
#SBATCH --mem=32G                   # Memory per node
#SBATCH -t 24:00:00                 # Walltime (hh:mm:ss)
#SBATCH -o logs/ravdess_%j.out      # STDOUT (%j = job ID)
#SBATCH -e logs/ravdess_%j.err      # STDERR
#SBATCH --mail-type=END,FAIL        # Email on end/fail
#SBATCH --mail-user=<NETID>@rutgers.edu  # <-- change to your Rutgers email

# ------------------------------------------------------------------
# 1) Load Anaconda on Amarel and enable conda
# ------------------------------------------------------------------
module use /projects/community/modulefiles
module load anaconda/2020.07-gc563

# Make sure conda is available in non-interactive shells
source ~/.bashrc

# ------------------------------------------------------------------
# 2) Activate your conda environment
# ------------------------------------------------------------------
conda env create -f environment.yml -n multimodal-emotion-detection
conda activate multimodal-emotion-detection

# ------------------------------------------------------------------
# 3) Go to your project directory
# ------------------------------------------------------------------
cd /scratch/pbm52/emotion-detection-mm/multimodal-emotion-detection   # <-- change path to your repo root

# Ensure logs directory exists (sbatch won't create nested dirs automatically)
mkdir -p logs

# ------------------------------------------------------------------
# 4) Run training
# ------------------------------------------------------------------
python /src/train.py \
  experiment.name=ravdess_audio_video_baseline \
  dataset.name=ravdess \
  dataset.data_dir=multimodal-dataset \
  dataset.modalities='[audio, video]' \
  dataset.num_classes=8
