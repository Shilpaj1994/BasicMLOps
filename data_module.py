import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gzip
import shutil
from pathlib import Path
import urllib.request
import imgaug.augmenters as iaa

def download_and_extract_mnist_data():
    """Download and extract MNIST dataset from a reliable mirror"""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    data_dir = Path("data/MNIST/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for file_name in files.values():
        gz_file_path = data_dir / file_name
        extracted_file_path = data_dir / file_name.replace('.gz', '')

        # If the extracted file exists, skip downloading
        if extracted_file_path.exists():
            # print(f"{extracted_file_path} already exists, skipping download.")
            continue

        # Download the file
        print(f"Downloading {file_name}...")
        url = base_url + file_name
        try:
            urllib.request.urlretrieve(url, gz_file_path)
            print(f"Successfully downloaded {file_name}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
            raise Exception(f"Could not download {file_name}")

        # Extract the files
        try:
            print(f"Extracting {file_name}...")
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(extracted_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully extracted {file_name}")
        except Exception as e:
            print(f"Failed to extract {file_name}: {e}")
            raise Exception(f"Could not extract {file_name}")

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28).astype(np.float32)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

class CustomMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, augment=False):
        # Load images and normalize to [0, 1] range
        self.images = load_mnist_images(images_path) / 255.0
        self.labels = load_mnist_labels(labels_path)
        self.augment = augment
        
        # Define augmentation pipeline
        self.aug_pipeline = iaa.Sequential([
            iaa.Sometimes(0.7, [
                iaa.Affine(
                    rotate=(-15, 15),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    mode='constant',
                    cval=0
                ),
                iaa.GaussianBlur(sigma=(0.0, 2.0)),
                iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                iaa.CropAndPad(
                    percent=(-0.15, 0.15),
                    pad_mode='constant',
                    pad_cval=0
                )
            ])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image and reshape to (28, 28, 1) for imgaug
        image = self.images[idx].squeeze()
        image = np.expand_dims(image, axis=-1)
        
        # Apply augmentation if training
        if self.augment:
            image = self.aug_pipeline(images=[image])[0]
        
        # Convert to tensor and reshape to (1, 28, 28)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        label = int(self.labels[idx])
        
        # Normalize
        image = (image - 0.1307) / 0.3081
            
        return image, label

def get_data_loaders(batch_size=128):
    """Initialize and return train and test dataloaders"""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Ensure data is downloaded and extracted
    print("Preparing dataset...")
    download_and_extract_mnist_data()

    # Paths to the extracted files
    train_images_path = "data/MNIST/raw/train-images-idx3-ubyte"
    train_labels_path = "data/MNIST/raw/train-labels-idx1-ubyte"
    test_images_path = "data/MNIST/raw/t10k-images-idx3-ubyte"
    test_labels_path = "data/MNIST/raw/t10k-labels-idx1-ubyte"
    
    # Create datasets
    train_dataset = CustomMNISTDataset(train_images_path, train_labels_path, augment=True)
    test_dataset = CustomMNISTDataset(test_images_path, test_labels_path, augment=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader 