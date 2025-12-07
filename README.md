# 3D Liver Tumor Segmentation

A PyTorch-based implementation for 3D liver and liver tumor segmentation using U-Net architecture.

## Project Structure

```
3D-Liver-Tumor-Segmentation/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py              # 3D U-Net model definition
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training pipeline with PyTorch Lightning
│   ├── testing/
│   │   ├── __init__.py
│   │   └── evaluator.py         # Testing and evaluation pipeline
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
├── scripts/
│   ├── train.py                 # Training script
│   └── test.py                  # Testing script
├── configs/
│   └── config.yaml              # Configuration file
├── logs/                        # Training logs
├── outputs/                     # Model outputs, predictions, visualizations
│   ├── checkpoints/             # Model checkpoints
│   ├── predictions/             # Predicted segmentations
│   └── visualizations/          # Visualization animations
├── weights/                     # Pre-trained model weights
└── Task03_Liver_rs/            # Dataset directory
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 3D-Liver-Tumor-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Basic training (default settings)
python scripts/train.py

# Custom parameters
python scripts/train.py --batch_size 4 --max_epochs 50 --gpus 1
```

### 3. Test the Model
```bash
# Test with specific checkpoint
python scripts/test.py --checkpoint weights/epoch=97-step=25773.ckpt

# Test with most recent checkpoint
python scripts/test.py
```

### 4. Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir outputs/logs
```

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)

## Usage

### Training

Train the model using the default configuration:

```bash
python scripts/train.py
```

Train with custom parameters:

```bash
python scripts/train.py --config configs/config.yaml --batch_size 4 --max_epochs 50 --gpus 1
```

**Available options:**
- `--config`: Path to config file (default: `configs/config.yaml`)
- `--batch_size`: Batch size (overrides config)
- `--max_epochs`: Maximum epochs (overrides config)
- `--gpus`: Number of GPUs (default: 1, use 0 for CPU)

### Testing/Evaluation

Test the model with a checkpoint:

```bash
python scripts/test.py --checkpoint weights/epoch=97-step=25773.ckpt
```

Test with default checkpoint (uses most recent):

```bash
python scripts/test.py
```

**Test outputs:**
- Predictions: `outputs/test_*/predictions/prediction_*.nii.gz`
- Visualizations: `outputs/test_*/visualizations/visualization_*.gif`
- Logs: `logs/testing_*.log`

### Configuration

Edit `configs/config.yaml` to customize:
- Data paths and splits
- Model hyperparameters
- Training parameters (batch size, learning rate, epochs)
- Output directories

## Features

- **Modular Code Structure**: Separated into data, models, training, and testing modules
- **Logging**: Comprehensive logging to both files and console
- **Output Management**: Automatic saving of:
  - Model checkpoints
  - Training logs (TensorBoard)
  - Predictions (NIfTI format)
  - Visualizations (GIF animations)
- **Command Line Interface**: Easy-to-use scripts for training and testing
- **Configuration Management**: YAML-based configuration system
- **Patch-based Training**: Efficient training on large 3D volumes using patch sampling
- **Patch Aggregation**: Full volume prediction using grid sampling and aggregation

## Data

The dataset should be organized as:
```
Task03_Liver_rs/
├── imagesTr/          # Training images
│   └── liver_*.nii.gz
└── labelsTr/          # Training labels
    └── liver_*.nii.gz
```

## Outputs

- **Checkpoints**: Saved in `outputs/checkpoints/` with best models based on validation loss
- **Logs**: TensorBoard logs in `outputs/logs/` and text logs in `logs/`
- **Predictions**: NIfTI files in `outputs/test_*/predictions/`
- **Visualizations**: GIF animations in `outputs/test_*/visualizations/`

## Model Architecture

The model uses a 3D U-Net architecture with:
- 3 downsampling and 3 upsampling layers
- Skip connections between encoder and decoder
- 3 output channels (background, liver, tumor)
- Trilinear upsampling

## License

Data License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
