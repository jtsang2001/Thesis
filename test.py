import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import KITS23Dataset
from model import UNet3D
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def visualize_prediction(image, mask, prediction, save_path, slice_idx=None):
    """Visualize the original image, ground truth mask, and prediction."""
    if slice_idx is None:
        # If no specific slice is provided, use the middle slice
        slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(mask[slice_idx], cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(prediction[slice_idx], cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 5
) -> None:
    """Test the model and visualize results."""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Convert to numpy for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            # Visualize multiple slices for each sample
            for j in range(images_np.shape[0]):  # For each image in batch
                # Visualize middle slice
                middle_slice = images_np.shape[2] // 2
                save_path = os.path.join(save_dir, f'sample_{i}_image_{j}_middle.png')
                visualize_prediction(
                    images_np[j, 0],  # Remove channel dimension
                    masks_np[j, 0],
                    predictions_np[j, 0],
                    save_path,
                    middle_slice
                )
                
                # Visualize quarter slices
                quarter_slice = images_np.shape[2] // 4
                save_path = os.path.join(save_dir, f'sample_{i}_image_{j}_quarter.png')
                visualize_prediction(
                    images_np[j, 0],
                    masks_np[j, 0],
                    predictions_np[j, 0],
                    save_path,
                    quarter_slice
                )
                
                # Visualize three-quarter slice
                three_quarter_slice = (images_np.shape[2] * 3) // 4
                save_path = os.path.join(save_dir, f'sample_{i}_image_{j}_three_quarter.png')
                visualize_prediction(
                    images_np[j, 0],
                    masks_np[j, 0],
                    predictions_np[j, 0],
                    save_path,
                    three_quarter_slice
                )

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory for visualizations
    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms (same as validation)
    test_transform = A.Compose([
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])
    
    # Create test dataset
    test_dataset = KITS23Dataset(
        root_dir='kits23/dataset',
        transform=test_transform,
        mode='test',
        patch_size=(64, 64, 64)
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Load trained model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model and visualize results
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
        num_samples=5  # Number of samples to visualize
    )
    
    print(f'Test results have been saved to {save_dir}')

if __name__ == '__main__':
    main() 