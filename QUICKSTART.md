# Quick Start Guide

## Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify your data structure:**
Make sure your data is in the correct location:
```
Task03_Liver_rs/
├── imagesTr/          # Training images (liver_*.nii.gz)
└── labelsTr/          # Training labels (liver_*.nii.gz)
```

## Training

### Basic Training (Default Settings)

Simply run:
```bash
python scripts/train.py
```

This will:
- Use configuration from `configs/config.yaml`
- Train for 100 epochs (default)
- Use batch size of 2
- Save checkpoints to `outputs/checkpoints/`
- Save logs to `outputs/logs/` and `logs/`
- Use 1 GPU (if available)

### Custom Training Parameters

You can override config settings via command line:

```bash
# Train with custom batch size and epochs
python scripts/train.py --batch_size 4 --max_epochs 50

# Train on CPU only
python scripts/train.py --gpus 0

# Train with multiple GPUs
python scripts/train.py --gpus 2

# Use a custom config file
python scripts/train.py --config configs/my_custom_config.yaml
```

### Training Options

All available command-line options:
```bash
python scripts/train.py --help
```

Options:
- `--config`: Path to config file (default: `configs/config.yaml`)
- `--batch_size`: Batch size (overrides config)
- `--max_epochs`: Maximum epochs (overrides config)
- `--gpus`: Number of GPUs to use (default: 1, use 0 for CPU)

### Monitor Training

1. **TensorBoard:**
```bash
tensorboard --logdir outputs/logs
```
Then open http://localhost:6006 in your browser

2. **Log Files:**
Check `logs/training_*.log` for detailed training logs

3. **Checkpoints:**
Best models are saved in `outputs/checkpoints/` with names like:
```
epoch=XX-step=XXXX-val_loss=X.XXXX.ckpt
```

## Testing/Evaluation

### Basic Testing

Test with the most recent checkpoint:
```bash
python scripts/test.py
```

### Test with Specific Checkpoint

```bash
python scripts/test.py --checkpoint weights/epoch=97-step=25773.ckpt
```

Or use a checkpoint from training:
```bash
python scripts/test.py --checkpoint outputs/checkpoints/epoch=XX-step=XXXX-val_loss=X.XXXX.ckpt
```

### Testing Options

```bash
# Use custom config
python scripts/test.py --config configs/config.yaml --checkpoint path/to/checkpoint.ckpt

# Test on CPU
python scripts/test.py --checkpoint path/to/checkpoint.ckpt --gpus 0
```

### Test Outputs

After testing, you'll find:

1. **Predictions:** `outputs/test_*/predictions/prediction_*.nii.gz`
   - NIfTI files with predicted segmentations
   - Can be loaded in medical imaging software

2. **Visualizations:** `outputs/test_*/visualizations/visualization_*.gif`
   - Animated GIFs showing predictions vs ground truth
   - Useful for quick visual inspection

3. **Logs:** `logs/testing_*.log`
   - Detailed testing logs

## Configuration

Edit `configs/config.yaml` to customize:

- **Data settings:** patch sizes, train/val split, augmentation
- **Training settings:** batch size, learning rate, epochs, GPU usage
- **Model settings:** architecture parameters
- **Output paths:** where to save results

## Common Issues

### Out of Memory (OOM)

Reduce batch size:
```bash
python scripts/train.py --batch_size 1
```

Or edit `configs/config.yaml`:
```yaml
training:
  batch_size: 1
  max_length: 20  # Reduce queue size
  samples_per_volume: 3  # Fewer patches per volume
```

### CPU Training

If you don't have a GPU:
```bash
python scripts/train.py --gpus 0
```

### Slow Training

- Reduce `num_workers` in config if CPU is bottleneck
- Reduce `samples_per_volume` to process fewer patches
- Use smaller `patch_size_training`

## Example Workflow

1. **Start training:**
```bash
python scripts/train.py --batch_size 2 --max_epochs 100
```

2. **Monitor in another terminal:**
```bash
tensorboard --logdir outputs/logs
```

3. **After training completes, test:**
```bash
python scripts/test.py --checkpoint outputs/checkpoints/best_model.ckpt
```

4. **View results:**
- Check `outputs/test_*/predictions/` for segmentation files
- Check `outputs/test_*/visualizations/` for visualizations
- Check TensorBoard for metrics

## Tips

- Start with fewer epochs to test the pipeline: `--max_epochs 5`
- Use `--gpus 0` for CPU-only training/testing
- Check logs if something goes wrong: `logs/training_*.log`
- Best model is automatically saved based on validation loss
- Predictions are saved in NIfTI format for easy viewing in medical imaging tools

