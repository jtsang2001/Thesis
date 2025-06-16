import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nibabel import load
from typing import Tuple, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import zoom

class KITS23Dataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 transform: A.Compose = None,
                 mode: str = 'train',
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 target_shape: Tuple[int, int, int] = (128, 128, 128)):
        """
        KITS23 Dataset for 3D tumor segmentation.
        
        Args:
            root_dir: Root directory containing case folders
            transform: Albumentations transforms
            mode: 'train' or 'val'
            patch_size: Size of 3D patches to extract
            target_shape: Target shape for resizing the full volume
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.patch_size = patch_size
        self.target_shape = target_shape
        
        # Get list of case directories
        self.case_dirs = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('case_')]
        
        # Sort case directories
        self.case_dirs.sort()
        
        # Split into train/val
        split_idx = int(len(self.case_dirs) * 0.8)
        if mode == 'train':
            self.case_dirs = self.case_dirs[:split_idx]
        else:
            self.case_dirs = self.case_dirs[split_idx:]
            
        # Default transforms
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.case_dirs)
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocess a 3D volume to have consistent shape and intensity range.
        
        Args:
            volume: Input 3D volume
            
        Returns:
            Preprocessed volume
        """
        # Calculate zoom factors
        zoom_factors = [t/s for t, s in zip(self.target_shape, volume.shape)]
        
        # Resize volume to target shape
        resized = zoom(volume, zoom_factors, order=1)
        
        # Normalize to 0-1 range
        normalized = (resized - resized.min()) / (resized.max() - resized.min())
        
        return normalized
    
    def load_case(self, case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a single case's image and segmentation."""
        case_path = os.path.join(self.root_dir, case_dir)
        
        # Load image and segmentation
        image = load(os.path.join(case_path, 'imaging.nii.gz')).get_fdata()
        segmentation = load(os.path.join(case_path, 'segmentation.nii.gz')).get_fdata()
        
        # Preprocess volumes
        image = self.preprocess_volume(image)
        segmentation = self.preprocess_volume(segmentation)
        
        # Ensure segmentation is binary
        segmentation = (segmentation > 0.5).astype(np.float32)
        
        return image, segmentation
    
    def extract_patch(self, 
                     image: np.ndarray, 
                     mask: np.ndarray,
                     center: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a 3D patch from the image and mask."""
        d, h, w = self.patch_size
        z, y, x = center
        
        # Calculate patch boundaries
        z_start = max(0, z - d//2)
        y_start = max(0, y - h//2)
        x_start = max(0, x - w//2)
        
        z_end = min(image.shape[0], z_start + d)
        y_end = min(image.shape[1], y_start + h)
        x_end = min(image.shape[2], x_start + w)
        
        # Extract patches
        image_patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if image_patch.shape != self.patch_size:
            pad_d = max(0, d - image_patch.shape[0])
            pad_h = max(0, h - image_patch.shape[1])
            pad_w = max(0, w - image_patch.shape[2])
            
            image_patch = np.pad(image_patch, 
                               ((0, pad_d), (0, pad_h), (0, pad_w)),
                               mode='constant')
            mask_patch = np.pad(mask_patch,
                              ((0, pad_d), (0, pad_h), (0, pad_w)),
                              mode='constant')
        
        return image_patch, mask_patch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_dir = self.case_dirs[idx]
        image, mask = self.load_case(case_dir)
        
        # For training, randomly sample a patch
        if self.mode == 'train':
            # Find a random point in the image
            z = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            x = np.random.randint(0, image.shape[2])
            center = (z, y, x)
        else:
            # For validation, use the center of the image
            center = (image.shape[0]//2, image.shape[1]//2, image.shape[2]//2)
        
        # Extract patch
        image_patch, mask_patch = self.extract_patch(image, mask, center)
        
        # Apply transforms
        if self.transform:
            # Process each slice with transforms
            transformed_slices = []
            transformed_masks = []
            
            for i in range(image_patch.shape[0]):
                # Get 2D slice
                slice_img = image_patch[i]
                slice_mask = mask_patch[i]
                
                # Apply transforms
                transformed = self.transform(image=slice_img, mask=slice_mask)
                transformed_slices.append(transformed['image'])
                transformed_masks.append(transformed['mask'].unsqueeze(0))  # Add channel dimension
            
            # Stack slices back into 3D tensors
            image_patch = torch.stack(transformed_slices, dim=0)  # [D, C, H, W]
            mask_patch = torch.stack(transformed_masks, dim=0)    # [D, C, H, W]
            
            # Rearrange dimensions to [C, D, H, W]
            image_patch = image_patch.permute(1, 0, 2, 3)
            mask_patch = mask_patch.permute(1, 0, 2, 3)
        else:
            # Convert to tensors and add channel dimension
            image_patch = torch.from_numpy(image_patch).float()  # [D, H, W]
            mask_patch = torch.from_numpy(mask_patch).float()    # [D, H, W]
            
            # Add channel dimension
            image_patch = image_patch.unsqueeze(0)  # [1, D, H, W]
            mask_patch = mask_patch.unsqueeze(0)    # [1, D, H, W]
        
        return {
            'image': image_patch,
            'mask': mask_patch,
            'case_id': case_dir
        } 