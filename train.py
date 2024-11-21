import torch
import torch.nn as nn
import torch.optim as optim
from model import MNISTModel
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
from data_module import get_data_loaders
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def visualize_augmentations(train_loader):
    """Display various augmentations on a sample image from the dataset"""
    plt.style.use('dark_background')
    
    # Get a sample image
    batch = next(iter(train_loader))
    orig_img = batch[0][0].numpy()  # Get first image
    
    # Create augmentation sequences for visualization
    augmenters = {
        'Original': None,
        'Rotation': iaa.Affine(rotate=15),
        'Translation': iaa.Affine(translate_percent={"x": 0.1, "y": 0.1}),
        'Perspective': iaa.PerspectiveTransform(scale=0.15),
        'Blur': iaa.GaussianBlur(sigma=2.0),
        'Crop & Pad': iaa.CropAndPad(percent=0.15)
    }
    
    # Create subplot grid
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Data Augmentation Examples', fontsize=16)
    
    # Prepare image for augmentation (reshape to HWC format)
    img_for_aug = np.transpose(orig_img, (1, 2, 0))
    
    for idx, (name, augmenter) in enumerate(augmenters.items(), 1):
        plt.subplot(2, 3, idx)
        plt.axis('off')
        plt.title(name)
        
        # Apply augmentation if not original
        if augmenter is None:
            img_to_show = img_for_aug
        else:
            img_to_show = augmenter(images=[img_for_aug])[0]
        
        plt.imshow(img_to_show.squeeze(), cmap='gray')
    
    plt.tight_layout()
    
    # Save the visualization
    save_dir = Path("visualizations")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "augmentations.png")
    plt.close()
    print(f"Augmentation visualization saved to {save_dir}/augmentations.png")

def validate(model, test_loader, criterion, device):
    """Validate the model on the test set"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            num_batches += 1

    val_loss = val_loss / num_batches
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def train(epochs=1, batch_size=128, visualize=True):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Get data loaders
        train_loader, test_loader = get_data_loaders(batch_size)
        
        # Visualize augmentations if requested
        if visualize:
            visualize_augmentations(train_loader)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    best_val_acc = 0
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for organization
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            unit='batch',
            leave=True
        )

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss.item()

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Validation phase
        print("\nRunning validation...")
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs} Results:")
        print(f"Training Loss: {current_loss:.4f} | Training Accuracy: {current_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"\nNew best validation accuracy: {val_acc:.2f}%")
            
            # Generate filename with timestamp and accuracy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'model_mnist_acc_{val_acc:.2f}_{timestamp}.pth'
            model_path = models_dir / model_filename
            
            print(f"Saving model as: {model_filename}")
            # Save model weights and training info
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, model_path)
            
            # Save training metrics
            metrics_file = artifacts_dir / "training_metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write(f"Training completed at: {datetime.now()}\n")
                f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
                f.write(f"Final validation loss: {val_loss:.4f}\n")
                f.write(f"Model saved as: {model_filename}\n")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
if __name__ == '__main__':
    train(epochs=1, batch_size=128, visualize=True) 