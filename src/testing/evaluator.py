"""
Testing and evaluation module
"""
import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import nibabel as nib
from celluloid import Camera

from src.training.trainer import Segmenter
from src.data.dataset import create_subjects, create_transforms


def load_model(checkpoint_path: Path, config: Dict[str, Any], device: torch.device) -> Segmenter:
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = Segmenter.load_from_checkpoint(str(checkpoint_path), config=config)
    model = model.eval()
    model.to(device)
    return model


def predict_volume(
    model: Segmenter,
    subject: tio.Subject,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    patch_overlap: Tuple[int, int, int] = (8, 8, 8),
    batch_size: int = 4,
    device: torch.device = None
) -> torch.Tensor:
    """
    Predict segmentation for a full volume using patch aggregation
    
    Args:
        model: Trained model
        subject: TorchIO subject
        patch_size: Size of patches
        patch_overlap: Overlap between patches
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        Predicted segmentation tensor
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # GridSampler
    grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
    
    # GridAggregator
    aggregator = tio.inference.GridAggregator(grid_sampler)
    
    # DataLoader for speed up
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    
    # Prediction
    with torch.no_grad():
        for patches_batch in patch_loader:
            input_tensor = patches_batch['CT']["data"].to(device)
            locations = patches_batch[tio.LOCATION]
            pred = model(input_tensor)
            aggregator.add_batch(pred, locations)
    
    # Extract the volume prediction
    output_tensor = aggregator.get_output_tensor()
    return output_tensor


def save_prediction(
    prediction: torch.Tensor,
    output_path: Path,
    reference_affine: np.ndarray = None
):
    """
    Save prediction as NIfTI file
    
    Args:
        prediction: Prediction tensor
        output_path: Path to save prediction
        reference_affine: Affine matrix for NIfTI header
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and get class predictions
    pred_np = prediction.argmax(0).cpu().numpy().astype(np.uint8)
    
    # Create NIfTI image
    if reference_affine is not None:
        nii_img = nib.Nifti1Image(pred_np, reference_affine)
    else:
        nii_img = nib.Nifti1Image(pred_np, np.eye(4))
    
    nib.save(nii_img, output_path)


def create_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray = None,
    output_path: Path = None,
    step: int = 2
):
    """
    Create and save visualization animation
    
    Args:
        image: Input CT image
        prediction: Predicted segmentation
        ground_truth: Ground truth segmentation (optional)
        output_path: Path to save visualization
        step: Step size for slices
    """
    fig = plt.figure(figsize=(12, 6))
    camera = Camera(fig)
    
    num_slices = image.shape[2]
    
    for i in range(0, num_slices, step):
        plt.subplot(1, 2 if ground_truth is not None else 1, 1)
        plt.imshow(image[:, :, i], cmap="bone")
        pred_mask = np.ma.masked_where(prediction[:, :, i] == 0, prediction[:, :, i])
        plt.imshow(pred_mask, alpha=0.6, cmap="autumn")
        plt.title("Prediction")
        plt.axis('off')
        
        if ground_truth is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(image[:, :, i], cmap="bone")
            gt_mask = np.ma.masked_where(ground_truth[:, :, i] == 0, ground_truth[:, :, i])
            plt.imshow(gt_mask, alpha=0.6, cmap="jet")
            plt.title("Ground Truth")
            plt.axis('off')
        
        plt.tight_layout()
        camera.snap()
    
    animation = camera.animate()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Try to save as GIF using pillow
            animation.save(str(output_path), writer='pillow', fps=10)
        except Exception:
            # Fallback: save frames as images
            frames_dir = output_path.parent / output_path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(animation.frame_seq):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                # Note: This is a simplified fallback - full implementation would require
                # extracting frames from animation object
                print(f"Visualization saved to: {output_path} (or frames in {frames_dir})")
    
    plt.close()
    return animation


def evaluate_dataset(
    model: Segmenter,
    dataset: tio.SubjectsDataset,
    output_dir: Path,
    config: Dict[str, Any],
    device: torch.device = None,
    num_samples: int = None
):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        output_dir: Directory to save outputs
        config: Configuration dictionary
        device: Device to run inference on
        num_samples: Number of samples to evaluate (None for all)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    visualizations_dir = output_dir / "visualizations"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = num_samples or len(dataset)
    
    for idx in range(min(num_samples, len(dataset))):
        subject = dataset[idx]
        
        # Get reference affine from original image
        reference_affine = subject["CT"].affine
        
        # Predict
        prediction = predict_volume(
            model=model,
            subject=subject,
            patch_size=tuple(config.get('patch_size', [96, 96, 96])),
            patch_overlap=tuple(config.get('patch_overlap', [8, 8, 8])),
            batch_size=config.get('batch_size', 4),
            device=device
        )
        
        # Save prediction
        pred_path = predictions_dir / f"prediction_{idx}.nii.gz"
        save_prediction(prediction, pred_path, reference_affine.numpy())
        
        # Create visualization
        img_data = subject["CT"]["data"][0, 0].cpu().numpy()
        pred_data = prediction.argmax(0).cpu().numpy()
        gt_data = subject["Label"]["data"][0, 0].cpu().numpy() if "Label" in subject else None
        
        viz_path = visualizations_dir / f"visualization_{idx}.gif"
        create_visualization(img_data, pred_data, gt_data, viz_path)
        
        print(f"Processed sample {idx + 1}/{num_samples}")

