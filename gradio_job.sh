#!/bin/bash
#SBATCH --job-name=liver_gradio
#SBATCH --output=logs/gradio_%j.out
#SBATCH --error=logs/gradio_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ufl.edu

# Load modules
module load conda

# Activate environment
source activate liver-segmentation

# Navigate to project directory
# Handle both running from project root and subdirectories
if [ -f "$SLURM_SUBMIT_DIR/gradio_app.py" ]; then
    cd $SLURM_SUBMIT_DIR
elif [ -f "$SLURM_SUBMIT_DIR/../gradio_app.py" ]; then
    cd $SLURM_SUBMIT_DIR/..
else
    # Try to find the project root
    cd /home/saririans/gongbcmc/gongbcmc
fi

# Set port (use a high port number to avoid conflicts)
export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME=0.0.0.0

# Get the hostname
HOSTNAME=$(hostname)
echo "Gradio will be available on: ${HOSTNAME}:${GRADIO_SERVER_PORT}"
echo "To access from your local machine, create an SSH tunnel:"
echo "ssh -L ${GRADIO_SERVER_PORT}:${HOSTNAME}:${GRADIO_SERVER_PORT} $USER@hpg.rc.ufl.edu"
echo "Then open: http://localhost:${GRADIO_SERVER_PORT}"
echo "Current directory: $(pwd)"

# Run Gradio app
python gradio_app.py

echo "Gradio app stopped at $(date)"

