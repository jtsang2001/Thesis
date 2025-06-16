import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """Double 3D Convolution with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet3D(nn.Module):
    """3D U-Net model for medical image segmentation."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 64)
        self.enc2 = DoubleConv3D(64, 128)
        self.enc3 = DoubleConv3D(128, 256)
        self.enc4 = DoubleConv3D(256, 512)
        self.enc5 = DoubleConv3D(512, 1024)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(1024, 1024)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv3D(1536, 512)  # 1536 = 512 (upconv) + 1024 (skip)
        
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(768, 256)  # 768 = 256 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(384, 128)  # 384 = 128 (upconv) + 256 (skip)
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(192, 64)  # 192 = 64 (upconv) + 128 (skip)
        
        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(128, 64)  # 128 = 64 (upconv) + 64 (skip)
        
        # Final convolution
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        x = F.max_pool3d(enc1, kernel_size=2, stride=2)
        
        enc2 = self.enc2(x)
        x = F.max_pool3d(enc2, kernel_size=2, stride=2)
        
        enc3 = self.enc3(x)
        x = F.max_pool3d(enc3, kernel_size=2, stride=2)
        
        enc4 = self.enc4(x)
        x = F.max_pool3d(enc4, kernel_size=2, stride=2)
        
        enc5 = self.enc5(x)
        x = F.max_pool3d(enc5, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv5(x)
        x = torch.cat([x, enc5], dim=1)
        x = self.dec5(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return torch.sigmoid(x)

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Calculate Dice loss."""
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / 
                 (pred.sum(dim=2).sum(dim=2).sum(dim=2) + 
                  target.sum(dim=2).sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class DiceLoss(nn.Module):
    """Combined Dice and BCE loss."""
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss_val = dice_loss(pred, target)
        return self.weight * dice_loss_val