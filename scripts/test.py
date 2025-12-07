#!/usr/bin/env python3
"""
Testing script for 3D Liver Tumor Segmentation
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchio as tio
from src.utils.config import load_config, parse_args
from src.utils.logger import setup_logger
from src.data.dataset import (
    create_subjects,
    create_transforms
)
from src.testing.evaluator import (
    load_model,
    evaluate_dataset
)


def main():
    """Main testing function"""
    args = parse_args()
    
    # Load configuration
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Use best checkpoint from weights directory
        weights_dir = project_root / "weights"
        checkpoints = list(weights_dir.glob("*.ckpt"))
        if not checkpoints:
            raise ValueError("No checkpoint found. Please specify --checkpoint")
        checkpoint_path = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    # Setup logging
    output_dir = project_root / config['output']['output_dir']
    logs_dir = output_dir / config['output']['logs_dir']
    logger = setup_logger('testing', logs_dir)
    
    logger.info("Starting testing...")
    logger.info(f"Using checkpoint: {checkpoint_path}")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['training']['gpus'] > 0 else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(checkpoint_path, config['training'], device)
    
    # Create data directory path
    data_dir = project_root / config['data']['data_dir']
    
    # Create subjects
    logger.info("Loading data...")
    subjects = create_subjects(data_dir)
    logger.info(f"Loaded {len(subjects)} subjects")
    
    # Create transforms (no augmentation for testing)
    val_transform = create_transforms(
        patch_size=tuple(config['data']['patch_size']),
        augmentation=False
    )
    
    # Create validation dataset
    # Use same split as training
    split_idx = int(len(subjects) * config['data']['train_split'])
    val_subjects = subjects[split_idx:]
    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
    
    logger.info(f"Test samples: {len(val_dataset)}")
    
    # Create output directory for this test run
    test_output_dir = output_dir / f"test_{checkpoint_path.stem}"
    
    # Evaluate
    logger.info("Starting evaluation...")
    evaluate_dataset(
        model=model,
        dataset=val_dataset,
        output_dir=test_output_dir,
        config=config,
        device=device,
        num_samples=config['testing'].get('num_samples')
    )
    
    logger.info("Testing completed!")
    logger.info(f"Results saved to: {test_output_dir}")


if __name__ == "__main__":
    main()

