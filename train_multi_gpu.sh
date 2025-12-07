#!/bin/bash
#SBATCH --job-name=liver_seg_multi
#SBATCH --output=logs/train_multi_%j.out
#SBATCH --error=logs/train_multi_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128gb
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ufl.edu

module load conda
source activate liver-segmentation
cd $SLURM_SUBMIT_DIR

# Note: Update config.yaml to use gpus: 2
python scripts/train.py --max_epochs 100 --gpus 2

echo "Multi-GPU training completed at $(date)"

