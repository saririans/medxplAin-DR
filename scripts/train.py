#!/usr/bin/env python3
"""
Training script for 3D Liver Tumor Segmentation
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.config import load_config, parse_args
from src.utils.logger import setup_logger
from src.data.dataset import (
    create_subjects,
    create_transforms,
    create_datasets,
    create_sampler,
    create_patch_queues,
    create_dataloaders
)
from src.training.trainer import Segmenter, train_model


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    if args.gpus is not None:
        config['training']['gpus'] = args.gpus
    
    # Setup logging
    output_dir = project_root / config['output']['output_dir']
    logs_dir = output_dir / config['output']['logs_dir']
    logger = setup_logger('training', logs_dir)
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Create data directory path
    data_dir = project_root / config['data']['data_dir']
    
    # Create subjects
    logger.info("Loading data...")
    subjects = create_subjects(data_dir)
    logger.info(f"Loaded {len(subjects)} subjects")
    
    # Create transforms
    train_transform = create_transforms(
        patch_size=tuple(config['data']['patch_size']),
        augmentation=config['data']['augmentation']
    )
    val_transform = create_transforms(
        patch_size=tuple(config['data']['patch_size']),
        augmentation=False
    )
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(
        subjects=subjects,
        train_split=config['data']['train_split'],
        train_transform=train_transform,
        val_transform=val_transform
    )
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create sampler
    sampler = create_sampler(
        patch_size=tuple(config['data']['patch_size_training']),
        sampler_type=config['data']['sampler_type'],
        label_probabilities=config['data']['label_probabilities']
    )
    
    # Create patch queues
    train_queue, val_queue = create_patch_queues(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        sampler=sampler,
        max_length=config['training']['max_length'],
        samples_per_volume=config['training']['samples_per_volume'],
        num_workers=config['training']['num_workers']
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_queue=train_queue,
        val_queue=val_queue,
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    logger.info("Creating model...")
    model = Segmenter(config['training'])
    
    # Train
    logger.info("Starting training...")
    trainer, checkpoint_callback = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        output_dir=output_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

