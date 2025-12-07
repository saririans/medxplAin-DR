#!/bin/bash
#SBATCH --job-name=liver_seg_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ufl.edu

module load conda
source activate liver-segmentation
cd $SLURM_SUBMIT_DIR

# Test with specific checkpoint (update path as needed)
python scripts/test.py --checkpoint weights/epoch=97-step=25773.ckpt --gpus 1

echo "Testing completed at $(date)"

