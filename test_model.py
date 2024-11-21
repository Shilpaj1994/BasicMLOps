import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
from data_module import get_data_loaders, CustomMNISTDataset
import glob
import pytest
import warnings
import numpy as np

# Suppress the torchvision warning
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def get_latest_model():
    model_files = glob.glob('models/model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def test_model_architecture():
    model = MNISTModel()
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25_000, f"Model has {total_params} parameters, should be < 25_000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    """Test if model achieves >95% accuracy on validation set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    try:
        # Load the latest trained model
        model_path = get_latest_model()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"\nTesting model: {model_path}")
    except FileNotFoundError as e:
        pytest.fail("No trained model found. Please train the model first.")
    except Exception as e:
        pytest.fail(f"Error loading model: {str(e)}")
    
    model.eval()
    
    # Get validation data loader
    _, test_loader = get_data_loaders(batch_size=1000)
    
    correct = 0
    total = 0
    
    # Evaluate model
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    
    # Assert accuracy is above 95%
    assert accuracy > 95, f"Model accuracy ({accuracy:.2f}%) is below the required 95% threshold"
    print("✓ Model meets accuracy requirement (>95%)")

def test_model_predictions():
    """Test if model makes sensible predictions for each digit"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    try:
        model_path = get_latest_model()
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        pytest.fail(f"Error loading model: {str(e)}")
    
    model.eval()
    _, test_loader = get_data_loaders(batch_size=100)
    
    # Track predictions for each digit
    digit_correct = {i: 0 for i in range(10)}
    digit_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Update counts for each digit
            for t, p in zip(target, pred):
                digit_total[t.item()] += 1
                if t.item() == p.item():
                    digit_correct[t.item()] += 1
    
    # Check accuracy for each digit
    for digit in range(10):
        accuracy = 100 * digit_correct[digit] / digit_total[digit]
        print(f"Digit {digit} accuracy: {accuracy:.2f}%")
        assert accuracy > 90, f"Poor accuracy ({accuracy:.2f}%) for digit {digit}"

def test_model_robustness():
    """Test model's robustness to slight input perturbations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    try:
        model_path = get_latest_model()
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        pytest.fail(f"Error loading model: {str(e)}")
    
    model.eval()
    _, test_loader = get_data_loaders(batch_size=100)
    
    # Get a batch of test data
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # Get original predictions
    with torch.no_grad():
        original_output = model(data)
        original_pred = original_output.argmax(dim=1)
        
        # Test with small random noise
        noise = torch.randn_like(data) * 0.1
        noisy_output = model(data + noise)
        noisy_pred = noisy_output.argmax(dim=1)
        
        # At least 80% of predictions should remain the same with noise
        stability = (original_pred == noisy_pred).float().mean().item()
        print(f"\nModel stability under noise: {stability*100:.2f}%")
        assert stability > 0.8, f"Model predictions are too unstable under noise: {stability*100:.2f}%"

def test_model_confidence():
    """Test if model's confidence aligns with its accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    try:
        model_path = get_latest_model()
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        pytest.fail(f"Error loading model: {str(e)}")
    
    model.eval()
    _, test_loader = get_data_loaders(batch_size=100)
    
    confidences = []
    accuracies = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get probabilities using softmax
            probs = F.softmax(output, dim=1)
            confidence, pred = probs.max(dim=1)
            
            # Record confidence and whether prediction was correct
            correct = pred.eq(target)
            
            confidences.extend(confidence.cpu().numpy())
            accuracies.extend(correct.cpu().numpy())
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # High confidence predictions (>0.9) should be mostly correct
    high_conf_mask = confidences > 0.9
    high_conf_accuracy = accuracies[high_conf_mask].mean()
    
    print(f"\nAccuracy on high confidence predictions: {high_conf_accuracy*100:.2f}%")
    assert high_conf_accuracy > 0.95, f"Model's high confidence predictions are not reliable enough: {high_conf_accuracy*100:.2f}%"

if __name__ == '__main__':
    pytest.main([__file__]) 