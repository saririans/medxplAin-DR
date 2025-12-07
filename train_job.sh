#!/bin/bash
#SBATCH --job-name=liver_seg_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ufl.edu

# Load modules
module load conda

# Activate environment
source activate liver-segmentation

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run training
python scripts/train.py --max_epochs 100 --gpus 1

echo "Training completed at $(date)"

