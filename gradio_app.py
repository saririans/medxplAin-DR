#!/usr/bin/env python3
"""
Gradio interface for 3D Liver Tumor Segmentation
"""
import sys
from pathlib import Path
import os

# Add project root to path
# Handle both running from root and from subdirectories
script_path = Path(__file__).resolve()
script_name = script_path.name

# Find project root by looking for configs directory
current = script_path.parent
while current != current.parent:
    if (current / "configs" / "config.yaml").exists():
        project_root = current
        break
    current = current.parent
else:
    # Fallback: assume script is in project root
    project_root = script_path.parent

# Insert project root at the beginning
sys.path.insert(0, str(project_root))

# CRITICAL: Remove src from path BEFORE any imports to avoid circular import
# This prevents importing local gradio.py instead of the installed gradio library
src_path = str(project_root / "src")
if src_path in sys.path:
    sys.path.remove(src_path)

os.chdir(project_root)

# Set custom temp directory for file uploads (fix permission issues on shared systems)
temp_dir = project_root / "gradio_temp"
temp_dir.mkdir(exist_ok=True, parents=True)
os.environ["GRADIO_TEMP_DIR"] = str(temp_dir)
# Also patch tempfile to use our directory
import tempfile
original_gettempdir = tempfile.gettempdir
def custom_gettempdir():
    return str(temp_dir)
tempfile.gettempdir = custom_gettempdir

# Now import everything - gradio will be imported from the installed package
import torch
import torchio as tio
import numpy as np
import nibabel as nib
import gradio as gr
from typing import Tuple, Optional

from src.training.trainer import Segmenter
from src.testing.evaluator import predict_volume
from src.utils.config import load_config
from src.data.dataset import create_transforms


class ModelInference:
    """Wrapper class for model inference"""
    
    def __init__(self, checkpoint_path: str, config_path: str = "configs/config.yaml"):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        
        # Load config
        self.config = load_config(self.config_path)
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from {self.checkpoint_path}...")
        self.model = Segmenter.load_from_checkpoint(
            str(self.checkpoint_path),
            config=self.config['training']
        )
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Get patch settings from config
        self.patch_size = tuple(self.config['data']['patch_size_training'])
        self.patch_overlap = tuple(self.config['data']['patch_overlap'])
        self.batch_size = self.config['testing']['batch_size']
    
    def predict(self, nifti_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict segmentation for uploaded NIfTI file
        
        Returns:
            Tuple of (input_slice, prediction_slice, overlay_slice)
        """
        if nifti_file is None:
            return None, None, None
        
        try:
            # Load NIfTI file
            nii_img = nib.load(nifti_file.name)
            volume_data = nii_img.get_fdata()
            affine = nii_img.affine
            
            # Create TorchIO subject
            subject = tio.Subject({
                "CT": tio.ScalarImage(tensor=torch.from_numpy(volume_data).float().unsqueeze(0))
            })
            
            # Apply preprocessing transform
            transform = create_transforms(
                patch_size=tuple(self.config['data']['patch_size']),
                augmentation=False
            )
            subject = transform(subject)
            
            # Run prediction
            print("Running inference...")
            prediction = predict_volume(
                model=self.model,
                subject=subject,
                patch_size=self.patch_size,
                patch_overlap=self.patch_overlap,
                batch_size=self.batch_size,
                device=self.device
            )
            
            # Get class predictions
            pred_classes = prediction.argmax(0).cpu().numpy()
            
            # Get input volume
            input_volume = subject["CT"]["data"][0, 0].cpu().numpy()
            
            print(f"DEBUG: Input volume shape: {input_volume.shape}, range: [{input_volume.min():.2f}, {input_volume.max():.2f}]")
            print(f"DEBUG: Prediction shape: {pred_classes.shape}, range: [{pred_classes.min()}, {pred_classes.max()}]")
            
            # Handle different volume shapes (2D or 3D)
            if len(input_volume.shape) == 3:
                # 3D volume: select middle slice along z-axis
                mid_slice = input_volume.shape[2] // 2
                input_slice = input_volume[:, :, mid_slice]
                print(f"DEBUG: Extracted 3D slice {mid_slice}, input_slice shape: {input_slice.shape}")
                if len(pred_classes.shape) == 3:
                    pred_slice = pred_classes[:, :, mid_slice]
                else:
                    pred_slice = pred_classes
            elif len(input_volume.shape) == 2:
                # 2D volume: use as-is
                input_slice = input_volume
                print(f"DEBUG: Using 2D volume as-is, shape: {input_slice.shape}")
                if len(pred_classes.shape) == 3:
                    # If prediction is 3D but input is 2D, take middle slice
                    mid_slice = pred_classes.shape[2] // 2
                    pred_slice = pred_classes[:, :, mid_slice]
                else:
                    pred_slice = pred_classes
            else:
                # For other shapes, try to extract a 2D slice
                # Take the middle slice along the last dimension
                if len(input_volume.shape) >= 3:
                    mid_slice = input_volume.shape[-1] // 2
                    input_slice = input_volume[:, :, mid_slice]
                else:
                    input_slice = input_volume
                
                if len(pred_classes.shape) >= 3:
                    mid_slice = pred_classes.shape[-1] // 2
                    pred_slice = pred_classes[:, :, mid_slice]
                else:
                    pred_slice = pred_classes
            
            print(f"DEBUG: After extraction - input_slice shape: {input_slice.shape}, range: [{input_slice.min():.2f}, {input_slice.max():.2f}]")
            print(f"DEBUG: After extraction - pred_slice shape: {pred_slice.shape}, range: [{pred_slice.min()}, {pred_slice.max()}]")
            
            # Ensure both are 2D arrays
            # Flatten any extra dimensions
            if len(input_slice.shape) > 2:
                input_slice = input_slice.reshape(-1, input_slice.shape[-1])[:, 0] if input_slice.shape[-1] == 1 else input_slice[:, :, 0]
            if len(pred_slice.shape) > 2:
                pred_slice = pred_slice.reshape(-1, pred_slice.shape[-1])[:, 0] if pred_slice.shape[-1] == 1 else pred_slice[:, :, 0]
            
            # Ensure both are 2D
            if len(input_slice.shape) == 1:
                # If 1D, try to reshape (this shouldn't happen but handle it)
                side_len = int(np.sqrt(len(input_slice)))
                input_slice = input_slice.reshape(side_len, side_len) if side_len * side_len == len(input_slice) else input_slice.reshape(1, -1)
            if len(pred_slice.shape) == 1:
                side_len = int(np.sqrt(len(pred_slice)))
                pred_slice = pred_slice.reshape(side_len, side_len) if side_len * side_len == len(pred_slice) else pred_slice.reshape(1, -1)
            
            # Ensure shapes match - take minimum dimensions
            min_h = min(input_slice.shape[0], pred_slice.shape[0])
            min_w = min(input_slice.shape[1], pred_slice.shape[1])
            input_slice = input_slice[:min_h, :min_w]
            pred_slice = pred_slice[:min_h, :min_w]
            
            # Normalize input for display
            # Data might be in range [-1, 1] from RescaleIntensity transform
            print(f"DEBUG: Before normalization - min: {input_slice.min():.4f}, max: {input_slice.max():.4f}, mean: {input_slice.mean():.4f}")
            
            # Handle different data ranges
            if input_slice.min() < 0:
                # Data is in range [-1, 1] or similar, shift to [0, 1]
                input_slice_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)
            elif input_slice.max() > 1:
                # Data is in original Hounsfield units or similar, normalize to [0, 1]
                input_slice_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)
            elif input_slice.max() > input_slice.min():
                # Data is already in [0, 1] range, use as-is
                input_slice_norm = input_slice
            else:
                # If all values are the same, set to middle gray
                input_slice_norm = np.ones_like(input_slice) * 0.5
                print("DEBUG: Warning - all input values are the same, using 0.5")
            
            # Ensure values are in [0, 1] range
            input_slice_norm = np.clip(input_slice_norm, 0, 1)
            
            print(f"DEBUG: After normalization - min: {input_slice_norm.min():.4f}, max: {input_slice_norm.max():.4f}")
            
            # Convert to uint8 (0-255) for proper display in Gradio
            input_slice_display = (input_slice_norm * 255).astype(np.uint8)
            print(f"DEBUG: After uint8 conversion - min: {input_slice_display.min()}, max: {input_slice_display.max()}, mean: {input_slice_display.mean():.2f}")
            
            # Verify we have non-zero values
            if input_slice_display.max() == 0:
                print("DEBUG: ERROR - All values are zero after conversion!")
                # Use a test pattern to verify display works
                input_slice_display = np.ones_like(input_slice_display) * 128  # Gray
            
            # Create colored prediction mask
            # Background=0 (black), Liver=1 (green), Tumor=2 (red)
            pred_colored = np.zeros((*pred_slice.shape, 3), dtype=np.uint8)
            pred_colored[pred_slice == 1] = [0, 255, 0]  # Green for liver
            pred_colored[pred_slice == 2] = [255, 0, 0]  # Red for tumor
            
            # Create overlay (grayscale input + colored prediction)
            overlay = np.stack([input_slice_display, input_slice_display, input_slice_display], axis=-1)
            # Blend prediction colors with input
            liver_mask = (pred_slice == 1)
            tumor_mask = (pred_slice == 2)
            overlay[liver_mask] = overlay[liver_mask] * 0.5 + np.array([0, 255, 0]) * 0.5  # Green overlay
            overlay[tumor_mask] = overlay[tumor_mask] * 0.5 + np.array([255, 0, 0]) * 0.5  # Red overlay
            overlay = overlay.astype(np.uint8)
            
            print(f"Inference completed - Input shape: {input_slice_display.shape}, Range: [{input_slice_display.min()}, {input_slice_display.max()}]")
            
            return input_slice_display, pred_colored, overlay
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def save_prediction(self, nifti_file, output_dir: Path = None) -> Optional[str]:
        """
        Save prediction as NIfTI file
        
        Returns:
            Path to saved prediction file
        """
        if nifti_file is None:
            return None
        
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load original NIfTI for affine
            nii_img = nib.load(nifti_file.name)
            volume_data = nii_img.get_fdata()
            affine = nii_img.affine
            
            # Create TorchIO subject
            subject = tio.Subject({
                "CT": tio.ScalarImage(tensor=torch.from_numpy(volume_data).float().unsqueeze(0))
            })
            
            # Apply preprocessing
            transform = create_transforms(
                patch_size=tuple(self.config['data']['patch_size']),
                augmentation=False
            )
            subject = transform(subject)
            
            # Run prediction
            prediction = predict_volume(
                model=self.model,
                subject=subject,
                patch_size=self.patch_size,
                patch_overlap=self.patch_overlap,
                batch_size=self.batch_size,
                device=self.device
            )
            
            # Get class predictions
            pred_classes = prediction.argmax(0).cpu().numpy().astype(np.uint8)
            
            # Save as NIfTI
            output_path = output_dir / "prediction.nii.gz"
            nii_pred = nib.Nifti1Image(pred_classes, affine)
            nib.save(nii_pred, output_path)
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None


# Initialize model
CHECKPOINT_PATH = "/home/saririans/gongbcmc/gongbcmc/weights/epoch=70-step=18672.ckpt"
CONFIG_PATH = project_root / "configs" / "config.yaml"

# Check if checkpoint exists, if not use relative path
if not Path(CHECKPOINT_PATH).exists():
    CHECKPOINT_PATH = project_root / "weights" / "epoch=70-step=18672.ckpt"
else:
    CHECKPOINT_PATH = Path(CHECKPOINT_PATH)

# Convert to string for ModelInference
CHECKPOINT_PATH = str(CHECKPOINT_PATH)
CONFIG_PATH = str(CONFIG_PATH)

try:
    model_inference = ModelInference(CHECKPOINT_PATH, CONFIG_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL_LOADED = False
    model_inference = None


def predict_and_visualize(nifti_file):
    """Gradio prediction function"""
    if not MODEL_LOADED:
        return None, None, None, "Error: Model not loaded. Please check checkpoint path."
    
    if nifti_file is None:
        return None, None, None, "Please upload a NIfTI file."
    
    input_slice, pred_slice, overlay = model_inference.predict(nifti_file)
    
    if input_slice is None:
        return None, None, None, "Error during prediction. Check console for details."
    
    # Create visualization text
    if pred_slice is not None:
        liver_pixels = np.sum(pred_slice == 1)
        tumor_pixels = np.sum(pred_slice == 2)
        total_pixels = pred_slice.size
        
        info_text = f"""
        Segmentation Results:
        - Liver pixels: {liver_pixels} ({liver_pixels/total_pixels*100:.2f}%)
        - Tumor pixels: {tumor_pixels} ({tumor_pixels/total_pixels*100:.2f}%)
        - Total pixels: {total_pixels}
        """
    else:
        info_text = "Prediction completed."
    
    return input_slice, pred_slice, overlay, info_text


def download_prediction(nifti_file):
    """Save and return prediction file"""
    if not MODEL_LOADED or nifti_file is None:
        return None
    
    output_path = model_inference.save_prediction(nifti_file)
    if output_path:
        return output_path
    return None


# Create Gradio interface
with gr.Blocks(title="3D Liver Tumor Segmentation") as demo:
    gr.Markdown("# 3D Liver and Tumor Segmentation")
    gr.Markdown("Upload a CT scan in NIfTI format (.nii.gz) to get automated segmentation of liver and tumors.")
    
    if not MODEL_LOADED:
        gr.Markdown("## Warning: Model not loaded. Please check the checkpoint path.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload CT Scan (NIfTI .nii.gz)",
                file_types=[".nii.gz", ".nii"],
                type="filepath"
            )
            predict_btn = gr.Button("Run Segmentation", variant="primary")
            info_output = gr.Textbox(label="Information", lines=5)
        
        with gr.Column(scale=2):
            input_image = gr.Image(label="Input CT Slice", type="numpy", image_mode="L")
            pred_image = gr.Image(label="Segmentation (Green=Liver, Red=Tumor)", type="numpy")
            overlay_image = gr.Image(label="Overlay", type="numpy")
    
    with gr.Row():
        download_btn = gr.Button("Download Prediction (NIfTI)")
        download_file = gr.File(label="Download Segmentation")
    
    # Connect functions
    predict_btn.click(
        fn=predict_and_visualize,
        inputs=[file_input],
        outputs=[input_image, pred_image, overlay_image, info_output]
    )
    
    download_btn.click(
        fn=download_prediction,
        inputs=[file_input],
        outputs=[download_file]
    )
    
    gr.Markdown("""
    ## Instructions:
    1. Upload a CT scan in NIfTI format (.nii.gz or .nii)
    2. Click "Run Segmentation" to process the scan
    3. View the results: input slice, segmentation mask, and overlay
    4. Click "Download Prediction" to save the full 3D segmentation as NIfTI file
    
    ## Segmentation Classes:
    - Background (black)
    - Liver (green)
    - Tumor (red)
    """)


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    
    print(f"Starting Gradio server on {server_name}:{port}")
    print(f"Using temp directory: {os.environ.get('GRADIO_TEMP_DIR', 'default')}")
    demo.launch(
        server_name=server_name, 
        server_port=port, 
        share=False
    )

