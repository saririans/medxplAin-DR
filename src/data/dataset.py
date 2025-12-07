"""
Dataset creation and data loading utilities
"""
from pathlib import Path
import torchio as tio
import torch
from typing import List, Tuple


def change_img_to_label_path(path: Path) -> Path:
    """
    Replace imagesTr with labelsTr in the filepath
    
    Args:
        path: Path to image file
        
    Returns:
        Path to corresponding label file
    """
    parts = list(path.parts)
    if "imagesTr" in parts:
        parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)


def create_subjects(data_dir: Path) -> List[tio.Subject]:
    """
    Create torchio subjects from data directory
    
    Args:
        data_dir: Path to directory containing imagesTr and labelsTr
        
    Returns:
        List of torchio subjects
    """
    images_path = data_dir / "imagesTr"
    subjects_paths = list(images_path.glob("liver_*"))
    subjects = []
    
    for subject_path in subjects_paths:
        label_path = change_img_to_label_path(subject_path)
        if label_path.exists():
            subject = tio.Subject({
                "CT": tio.ScalarImage(subject_path),
                "Label": tio.LabelMap(label_path)
            })
            subjects.append(subject)
    
    return subjects


def create_transforms(
    patch_size: Tuple[int, int, int] = (256, 256, 200),
    augmentation: bool = True
) -> tio.Compose:
    """
    Create preprocessing and augmentation transforms
    
    Args:
        patch_size: Target patch size (H, W, D)
        augmentation: Whether to apply augmentation
        
    Returns:
        Transform composition
    """
    process = tio.Compose([
        tio.CropOrPad(patch_size),
        tio.RescaleIntensity((-1, 1))
    ])
    
    if augmentation:
        augmentation_transform = tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=(-10, 10)
        )
        return tio.Compose([process, augmentation_transform])
    
    return process


def create_datasets(
    subjects: List[tio.Subject],
    train_split: float = 0.9,
    train_transform: tio.Compose = None,
    val_transform: tio.Compose = None
) -> Tuple[tio.SubjectsDataset, tio.SubjectsDataset]:
    """
    Create train and validation datasets
    
    Args:
        subjects: List of subjects
        train_split: Fraction of data for training
        train_transform: Transform for training data
        val_transform: Transform for validation data
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    split_idx = int(len(subjects) * train_split)
    train_subjects = subjects[:split_idx]
    val_subjects = subjects[split_idx:]
    
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
    
    return train_dataset, val_dataset


def create_sampler(
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    sampler_type: str = "label",
    label_probabilities: dict = None
):
    """
    Create sampler for patch extraction
    
    Args:
        patch_size: Size of patches to extract
        sampler_type: Type of sampler ('label' or 'uniform')
        label_probabilities: Probability distribution for label sampler
        
    Returns:
        Sampler instance
    """
    if sampler_type == "label":
        if label_probabilities is None:
            label_probabilities = {0: 0.2, 1: 0.3, 2: 0.5}
        return tio.data.LabelSampler(
            patch_size=patch_size,
            label_name="Label",
            label_probabilities=label_probabilities
        )
    else:
        return tio.data.UniformSampler(patch_size=patch_size)


def create_patch_queues(
    train_dataset: tio.SubjectsDataset,
    val_dataset: tio.SubjectsDataset,
    sampler,
    max_length: int = 40,
    samples_per_volume: int = 5,
    num_workers: int = 4
) -> Tuple[tio.Queue, tio.Queue]:
    """
    Create patch queues for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        sampler: Sampler for patch extraction
        max_length: Maximum queue length
        samples_per_volume: Number of patches per volume
        num_workers: Number of workers for queue
        
    Returns:
        Tuple of (train_queue, val_queue)
    """
    train_queue = tio.Queue(
        train_dataset,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    
    val_queue = tio.Queue(
        val_dataset,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    
    return train_queue, val_queue


def create_dataloaders(
    train_queue: tio.Queue,
    val_queue: tio.Queue,
    batch_size: int = 2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders from queues
    
    Args:
        train_queue: Training queue
        val_queue: Validation queue
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_queue,
        batch_size=batch_size,
        num_workers=0  # Must be 0 for queue-based loaders
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_queue,
        batch_size=batch_size,
        num_workers=0
    )
    
    return train_loader, val_loader

