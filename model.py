import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv3D(nn.Module):
    """Double 3D Convolution with GroupNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),  # Better than BatchNorm for small batches
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet3D(nn.Module):
    """3D U-Net model for hierarchical medical image segmentation."""
    def __init__(self, in_channels: int = 1, out_channels: int = 3, features: list = [64, 128, 256, 512]):
        super().__init__()
        
        self.features = features
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # First encoder block
        self.encoder_blocks.append(DoubleConv3D(in_channels, features[0]))
        
        # Remaining encoder blocks
        for i in range(1, len(features)):
            self.encoder_blocks.append(DoubleConv3D(features[i-1], features[i]))
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        # Build decoder in reverse
        reversed_features = features[::-1]  # [512, 256, 128, 64]
        
        for i in range(len(reversed_features)):
            if i == 0:
                # First decoder block (from bottleneck)
                self.upconv_blocks.append(nn.ConvTranspose3d(reversed_features[i] * 2, reversed_features[i], kernel_size=2, stride=2))
                self.decoder_blocks.append(DoubleConv3D(reversed_features[i] * 2, reversed_features[i]))
            else:
                # Subsequent decoder blocks
                self.upconv_blocks.append(nn.ConvTranspose3d(reversed_features[i-1], reversed_features[i], kernel_size=2, stride=2))
                self.decoder_blocks.append(DoubleConv3D(reversed_features[i-1] + reversed_features[i], reversed_features[i]))
        
        # Final convolution for multi-label output
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            x = upconv(x)
            
            # Handle size mismatch if it occurs
            if x.shape != skip_connections[i].shape:
                x = F.interpolate(x, size=skip_connections[i].shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Sigmoid for multi-label classification
        return torch.sigmoid(x)

class MultiLabelDiceLoss(nn.Module):
    """Dice loss for multi-label hierarchical segmentation."""
    def __init__(self, smooth: float = 1.0, class_weights: list = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights or [1.0, 1.0, 2.0]  # Weight tumor higher
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss for a single channel."""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice_coef = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice_coef
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted multi-label dice loss.
        
        Args:
            pred: Predictions [B, 3, D, H, W]
            target: Ground truth [B, 3, D, H, W]
        """
        total_loss = 0
        
        for i in range(pred.shape[1]):  # For each of the 3 channels
            channel_loss = self.dice_loss(pred[:, i], target[:, i])
            total_loss += self.class_weights[i] * channel_loss
        
        return total_loss / sum(self.class_weights)

class MultiLabelDiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for multi-label segmentation."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, 
                 class_weights: list = None, pos_weight: torch.Tensor = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = MultiLabelDiceLoss(class_weights=class_weights)
        
        # BCE with positive class weighting for imbalanced data
        if pos_weight is not None:
            self.bce_loss = nn.BCELoss(reduction='none')
            self.pos_weight = pos_weight
        else:
            self.bce_loss = nn.BCELoss()
            self.pos_weight = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predictions [B, 3, D, H, W] (after sigmoid)
            target: Ground truth [B, 3, D, H, W]
        """
        # Dice loss
        dice_loss_val = self.dice_loss(pred, target)
        
        # BCE loss
        if self.pos_weight is not None:
            bce_loss_val = self.bce_loss(pred, target)
            # Apply positive class weighting
            bce_loss_val = (bce_loss_val * (target * self.pos_weight.to(pred.device) + (1 - target))).mean()
        else:
            bce_loss_val = self.bce_loss(pred, target)
        
        return self.dice_weight * dice_loss_val + self.bce_weight * bce_loss_val

def calculate_dice_scores(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calculate Dice scores for each hierarchical level.
    
    Args:
        pred: Predictions [B, 3, D, H, W]
        target: Ground truth [B, 3, D, H, W]
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with dice scores for each level
    """
    pred_binary = (pred > threshold).float()
    dice_scores = {}
    
    channel_names = ['kidney_tumor_cyst', 'tumor_cyst', 'tumor_only']
    
    for i, name in enumerate(channel_names):
        pred_channel = pred_binary[:, i].contiguous().view(-1)
        target_channel = target[:, i].contiguous().view(-1)
        
        intersection = (pred_channel * target_channel).sum()
        union = pred_channel.sum() + target_channel.sum()
        
        if union == 0:
            dice_score = 1.0  # Perfect score if both are empty
        else:
            dice_score = (2.0 * intersection / union).item()
        
        dice_scores[name] = dice_score
    
    # Calculate overall dice score
    dice_scores['overall'] = np.mean(list(dice_scores.values()))
    
    return dice_scores

def calculate_class_weights(dataloader) -> torch.Tensor:
    """
    Calculate positive class weights for BCE loss based on class frequencies.
    
    Args:
        dataloader: Training dataloader
        
    Returns:
        Tensor of positive class weights [3]
    """
    class_counts = torch.zeros(3)
    total_voxels = 0
    
    print("Calculating class weights...")
    for batch in dataloader:
        masks = batch['mask']  # [B, 3, D, H, W]
        
        for i in range(3):
            class_counts[i] += masks[:, i].sum()
        
        total_voxels += masks.numel() // 3  # Divide by 3 channels
    
    # Calculate positive class frequencies
    pos_frequencies = class_counts / total_voxels
    
    # Calculate positive class weights (inverse frequency)
    pos_weights = 1.0 / (pos_frequencies + 1e-8)
    
    print(f"Class frequencies: {pos_frequencies}")
    print(f"Positive class weights: {pos_weights}")
    
    return pos_weights

# Example usage for testing
if __name__ == "__main__":
    # Test the model
    model = UNet3D(in_channels=1, out_channels=3)
    
    # Create dummy input
    x = torch.randn(1, 1, 64, 64, 64)  # [B, C, D, H, W]
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test loss
    target = torch.randint(0, 2, (1, 3, 64, 64, 64)).float()
    criterion = MultiLabelDiceBCELoss()
    loss = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")
    import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import KITS23Dataset
from model import UNet3D, MultiLabelDiceBCELoss, calculate_dice_scores, calculate_class_weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=2):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    all_dice_scores = {
        'kidney_tumor_cyst': [],
        'tumor_cyst': [],
        'tumor_only': [],
        'overall': []
    }
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Calculate dice scores for monitoring
        with torch.no_grad():
            dice_scores = calculate_dice_scores(outputs, masks)
            for key, value in dice_scores.items():
                all_dice_scores[key].append(value)
    
    # Final optimizer step if there are remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    avg_dice_scores = {key: np.mean(values) for key, values in all_dice_scores.items()}
    
    return avg_loss, avg_dice_scores

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_dice_scores = {
        'kidney_tumor_cyst': [],
        'tumor_cyst': [],
        'tumor_only': [],
        'overall': []
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate dice scores
            dice_scores = calculate_dice_scores(outputs, masks)
            for key, value in dice_scores.items():
                all_dice_scores[key].append(value)
    
    avg_loss = total_loss / len(val_loader)
    avg_dice_scores = {key: np.mean(values) for key, values in all_dice_scores.items()}
    
    return avg_loss, avg_dice_scores

def plot_training_history(history, save_dir):
    """Plot and save training history."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot overall dice scores
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_dice']['overall'], 'b-', label='Train Dice')
    plt.plot(epochs, history['val_dice']['overall'], 'r-', label='Val Dice')
    plt.title('Overall Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    # Plot individual channel dice scores (validation only for clarity)
    plt.subplot(1, 3, 3)
    for channel in ['kidney_tumor_cyst', 'tumor_cyst', 'tumor_only']:
        plt.plot(epochs, history['val_dice'][channel], label=f'Val {channel}')
    plt.title('Validation Dice Scores by Channel')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    accumulation_steps: int = 2
) -> dict:
    """Train the model with comprehensive monitoring."""
    
    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': {'kidney_tumor_cyst': [], 'tumor_cyst': [], 'tumor_only': [], 'overall': []},
        'val_dice': {'kidney_tumor_cyst': [], 'tumor_cyst': [], 'tumor_only': [], 'overall': []},
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )
        
        # Validation
        val_loss, val_
    # Test dice scores
    dice_scores = calculate_dice_scores(output, target)
    print(f"Dice scores: {dice_scores}")