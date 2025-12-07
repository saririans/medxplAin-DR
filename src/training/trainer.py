"""
Training module using PyTorch Lightning
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.models.unet import UNet


class Segmenter(pl.LightningModule):
    """
    PyTorch Lightning module for 3D segmentation
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = UNet()
        
        # Optimize for Tensor Cores on NVIDIA GPUs (L4, A100, etc.)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        # Debug: Print first batch info
        if batch_idx == 0:
            print(f"DEBUG: Processing first training batch, batch_idx={batch_idx}")
            print(f"DEBUG: Image shape: {batch['CT']['data'].shape}")
            print(f"DEBUG: Label shape: {batch['Label']['data'].shape}")
        
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:, 0]  # Remove single channel
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Train Loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % self.config.get('log_images_every', 50) == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        
        if batch_idx == 0:
            print(f"DEBUG: First batch completed, loss={loss.item():.4f}")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:, 0]
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % self.config.get('log_images_every', 50) == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")
        
        return loss

    def log_images(self, img, pred, mask, name):
        """Log images to tensorboard"""
        results = []
        pred = torch.argmax(pred, 1)  # Take the output with the highest value
        axial_slice = 50  # Always plot slice 50 of the 96 slices
        
        fig, axis = plt.subplots(1, 2, figsize=(10, 5))
        axis[0].imshow(img[0][0][:, :, axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:, :, axial_slice] == 0, mask[0][:, :, axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        axis[0].axis('off')
        
        axis[1].imshow(img[0][0][:, :, axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:, :, axial_slice] == 0, pred[0][:, :, axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")
        axis[1].axis('off')
        
        plt.tight_layout()
        if self.logger:
            self.logger.experiment.add_figure(
                f"{name} Prediction vs Label",
                fig,
                self.global_step
            )
        plt.close(fig)
    
    def configure_optimizers(self):
        return [self.optimizer]
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Custom batch transfer for TorchIO data structures.
        TorchIO Subjects need special handling when moving to GPU.
        """
        import torchio as tio
        
        # TorchIO batches are dictionaries with nested structures
        # We need to move only the tensor data, not the entire TorchIO objects
        if isinstance(batch, dict):
            new_batch = {}
            for key, value in batch.items():
                # Skip location data (used by TorchIO for patch tracking)
                if key == tio.LOCATION:
                    new_batch[key] = value
                elif isinstance(value, dict) and "data" in value:
                    # Extract tensor data and move to device
                    new_batch[key] = {
                        "data": value["data"].to(device),
                        "affine": value.get("affine", None)
                    }
                elif isinstance(value, torch.Tensor):
                    # Direct tensors (if any)
                    new_batch[key] = value.to(device)
                else:
                    # Keep other values as-is (strings, paths, etc.)
                    new_batch[key] = value
            return new_batch
        else:
            # Fallback to default behavior
            return super().transfer_batch_to_device(batch, device, dataloader_idx)


def train_model(
    model: Segmenter,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    output_dir: Path
):
    """
    Train the segmentation model
    
    Args:
        model: Segmenter model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        output_dir: Directory to save outputs
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor='Val Loss',
        save_top_k=config.get('save_top_k', 10),
        mode='min',
        filename='epoch={epoch}-step={step}-val_loss={Val Loss:.4f}'
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir),
        name="segmentation"
    )
    
    # Create trainer
    num_gpus = config.get('gpus', 0)
    if num_gpus > 0 and torch.cuda.is_available():
        accelerator = 'gpu'
        devices = num_gpus
    else:
        accelerator = 'cpu'
        devices = 1  # CPU requires devices to be an int > 0
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        log_every_n_steps=config.get('log_every_n_steps', 1),
        callbacks=[checkpoint_callback],
        max_epochs=config.get('max_epochs', 100),
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Debug: Print info before training
    print("DEBUG: About to start training.fit()")
    print(f"DEBUG: Train loader length: {len(train_loader)}")
    print(f"DEBUG: Val loader length: {len(val_loader)}")
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return trainer, checkpoint_callback

