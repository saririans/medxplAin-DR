# Running on HiPerGator

This guide provides instructions for running the 3D Liver Tumor Segmentation project on HiPerGator (UF's supercomputing cluster).

## Prerequisites

1. **HiPerGator Account**: Ensure you have an active HiPerGator account
2. **Access**: SSH access to HiPerGator login nodes
3. **Data**: Upload your dataset to HiPerGator (or use existing data)

## Initial Setup

### 1. Connect to HiPerGator

```bash
ssh your_username@hpg.rc.ufl.edu
```

### 2. Navigate to Your Workspace

```bash
cd /blue/your_group/your_username  # or your preferred directory
```

### 3. Clone/Upload Project

If using git:
```bash
git clone <repository-url>
cd gongbcmc
```

Or upload your project files using `scp` or `rsync`:
```bash
# From your local machine
rsync -avz /path/to/local/gongbcmc your_username@hpg.rc.ufl.edu:/blue/your_group/your_username/
```

### 4. Create Conda Environment

```bash
# Load conda module
module load conda

# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate liver-segmentation
```

### 5. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## Data Preparation

### Upload Dataset

If your dataset is not already on HiPerGator:

```bash
# From your local machine
rsync -avz --progress Task03_Liver_rs/ your_username@hpg.rc.ufl.edu:/blue/your_group/your_username/gongbcmc/Task03_Liver_rs/
```

### Verify Data Structure

```bash
# On HiPerGator
cd gongbcmc
ls -la Task03_Liver_rs/imagesTr/ | head -5
ls -la Task03_Liver_rs/labelsTr/ | head -5
```

## Running Training Jobs

### Option 1: Interactive GPU Session (Testing)

For quick testing or debugging:

```bash
# Request interactive GPU node
srun --partition=gpu --gres=gpu:1 --mem=32gb --time=4:00:00 --pty bash

# Activate environment
conda activate liver-segmentation

# Run training
cd gongbcmc
python scripts/train.py --max_epochs 2 --gpus 1
```

### Option 2: SLURM Job Script (Recommended)

Create a job script for training:

**File: `train_job.sh`**
```bash
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
cd $SLURM_SUBMIT_DIR/gongbcmc

# Run training
python scripts/train.py --max_epochs 100 --gpus 1

echo "Training completed at $(date)"
```

**Submit the job:**
```bash
sbatch train_job.sh
```

### Option 3: Multi-GPU Training

For faster training with multiple GPUs:

**File: `train_multi_gpu.sh`**
```bash
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
cd $SLURM_SUBMIT_DIR/gongbcmc

# Note: Update config.yaml to use gpus: 2
python scripts/train.py --max_epochs 100 --gpus 2
```

**Update config.yaml:**
```yaml
training:
  gpus: 2  # Change from 1 to 2
```

## Running Gradio Web Interface

### Setup

The Gradio interface allows you to upload CT scans and get segmentation results through a web browser.

### Submit Gradio Job

**File: `gradio_job.sh`** (already created)

```bash
# Update email in the script first
sed -i 's/your_email@ufl.edu/YOUR_EMAIL@ufl.edu/g' gradio_job.sh

# Submit the job
sbatch gradio_job.sh
```

### Access Gradio Interface

1. **Check job output for hostname and port:**
```bash
tail -f logs/gradio_<job_id>.out
```

You'll see output like:
```
Gradio will be available on: gpu123:7860
To access from your local machine, create an SSH tunnel:
ssh -L 7860:gpu123:7860 $USER@hpg.rc.ufl.edu
```

2. **Create SSH tunnel from your local machine:**
```bash
ssh -L 7860:gpu123:7860 your_username@hpg.rc.ufl.edu
```
(Replace `gpu123` with the actual hostname from the job output)

3. **Open browser:**
```
http://localhost:7860
```

### Using the Interface

1. Upload a CT scan in NIfTI format (.nii.gz or .nii)
2. Click "Run Segmentation"
3. View results: input slice, segmentation mask, and overlay
4. Download the full 3D segmentation as NIfTI file

### Interactive Session (Alternative)

For testing or debugging:

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=32gb --time=4:00:00 --pty bash

# Activate environment
conda activate liver-segmentation

# Set port
export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME=0.0.0.0

# Run Gradio
cd /path/to/gongbcmc
python gradio_app.py
```

### Notes

- The Gradio app uses checkpoint: `weights/epoch=62-step=16568.ckpt`
- Update `CHECKPOINT_PATH` in `gradio_app.py` if using a different checkpoint
- The interface processes one volume at a time
- Processing time depends on volume size (typically 1-5 minutes per scan)

## Running Testing/Evaluation

### Create Test Job Script

**File: `test_job.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=liver_seg_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

module load conda
source activate liver-segmentation
cd $SLURM_SUBMIT_DIR/gongbcmc

# Test with specific checkpoint
python scripts/test.py --checkpoint weights/epoch=97-step=25773.ckpt --gpus 1
```

**Submit test job:**
```bash
sbatch test_job.sh
```

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View detailed job information
scontrol show job <job_id>

# View job output in real-time
tail -f logs/train_<job_id>.out
```

### Cancel a Job

```bash
scancel <job_id>
```

### View Resource Usage

```bash
# After job completes, check efficiency
seff <job_id>
```

## TensorBoard on HiPerGator

### Option 1: Port Forwarding (Recommended)

1. **On HiPerGator login node:**
```bash
# Start TensorBoard (use a high port number)
tensorboard --logdir outputs/logs --port 6006 --host 0.0.0.0
```

2. **On your local machine:**
```bash
# Create SSH tunnel
ssh -L 6006:localhost:6006 your_username@hpg.rc.ufl.edu

# Then open browser to: http://localhost:6006
```

### Option 2: Jupyter Notebook with TensorBoard

```bash
# Request interactive session
srun --partition=gpu --gres=gpu:1 --mem=16gb --time=2:00:00 --pty bash

# Start Jupyter
jupyter notebook --no-browser --port=8888
```

## Common HiPerGator Commands

### Check Available GPUs

```bash
# View GPU partitions
sinfo -p gpu

# Check GPU availability
sinfo -o "%P %G %C" | grep gpu
```

### Check Quota and Storage

```bash
# Check disk usage
quota -s

# Check storage
df -h /blue/your_group
```

### Module Commands

```bash
# List available modules
module avail

# Load specific modules
module load cuda/11.8
module load python/3.10

# List loaded modules
module list
```

## Configuration for HiPerGator

### Update config.yaml for HiPerGator

Recommended settings for HiPerGator GPU nodes:

```yaml
training:
  batch_size: 4  # Increase if GPU memory allows
  learning_rate: 0.0001
  max_epochs: 100
  gpus: 1  # or 2 for multi-GPU
  num_workers: 8  # Match cpus-per-task
  max_length: 40
  samples_per_volume: 5
```

### Memory Considerations

- **Single GPU**: Use `--mem=64gb` for batch_size=2-4
- **Multi-GPU**: Use `--mem=128gb` for batch_size=4-8
- **Large models**: May need `--mem=256gb`

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 1  # Reduce from 2

# Or request more memory
#SBATCH --mem=128gb
```

### CUDA Errors

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version compatibility
nvidia-smi
```

### TorchIO Batch Transfer Errors

If you see `TypeError: The path argument cannot be a dictionary`:

This has been fixed in the code with a custom `transfer_batch_to_device` method. If you still encounter this:

1. **Ensure you have the latest code** with the fix
2. **Check PyTorch Lightning version**: Should be >= 2.0.0
3. **Verify TorchIO version**: Should be >= 0.18.0

The fix automatically handles TorchIO's nested batch structure when moving data to GPU.

### Module Not Found Errors

```bash
# Ensure environment is activated
conda activate liver-segmentation

# Reinstall if needed
pip install -r requirements.txt
```

### Data Loading Issues

```bash
# Check data paths are correct
ls -la Task03_Liver_rs/imagesTr/
ls -la Task03_Liver_rs/labelsTr/

# Verify file permissions
chmod -R 755 Task03_Liver_rs/
```

## Example Workflow

### Complete Training Workflow

```bash
# 1. Connect to HiPerGator
ssh your_username@hpg.rc.ufl.edu

# 2. Navigate to project
cd /blue/your_group/your_username/gongbcmc

# 3. Activate environment
conda activate liver-segmentation

# 4. Create logs directory
mkdir -p logs

# 5. Submit training job
sbatch train_job.sh

# 6. Monitor job
squeue -u $USER
tail -f logs/train_<job_id>.out

# 7. After training, submit test job
sbatch test_job.sh

# 8. Check results
ls -la outputs/checkpoints/
ls -la outputs/test_*/predictions/
```

## Quick Reference

| Task | Command |
|------|---------|
| Submit training job | `sbatch train_job.sh` |
| Check job status | `squeue -u $USER` |
| View job output | `tail -f logs/train_<job_id>.out` |
| Cancel job | `scancel <job_id>` |
| Check GPU availability | `sinfo -p gpu` |
| Activate environment | `conda activate liver-segmentation` |
| Start TensorBoard | `tensorboard --logdir outputs/logs --port 6006` |
| Submit Gradio job | `sbatch gradio_job.sh` |
| Access Gradio | Create SSH tunnel and open `http://localhost:7860` |

## Notes

- **Time Limits**: GPU partitions typically have 24-48 hour limits
- **Queue Times**: GPU jobs may have longer wait times during peak hours
- **Storage**: Use `/blue/` for persistent storage, `/orange/` for scratch
- **Backup**: Regularly backup checkpoints and important outputs
- **Email Notifications**: Configure `--mail-user` in SLURM scripts for job completion alerts

## Support

For HiPerGator-specific issues:
- **HiPerGator Documentation**: https://help.rc.ufl.edu/
- **Support Email**: support@rc.ufl.edu
- **Slack**: UF Research Computing Slack channel

For project-specific issues, refer to the main README.md file.

