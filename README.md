# Basic MLOps Project

[![ML Pipeline](https://github.com/Shilpaj1994/BasicMLOps/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/Shilpaj1994/BasicMLOps/actions/workflows/ml-pipeline.yml)

This repository contains a machine learning project with a complete training pipeline and GitHub Actions workflow integration.

## Data Augmentation Examples

Before training, the model visualizes various augmentation techniques applied to a sample MNIST digit:

![Data Augmentation Examples](visualizations/augmentations.png)

The following augmentations are applied:
- Center Crop: Crops the center portion of the image
- Random Rotation: Rotates the image between -15 and 15 degrees
- Random Affine: Applies random affine transformations
- Random Perspective: Applies perspective transformations
- Gaussian Blur: Applies Gaussian blur with random sigma

## Project Structure

```
├── model.py # Model architecture definition
├── data_module.py # Data loading and preprocessing
├── train.py # Training script
├── test_model.py # Unit tests for the model
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # CI/CD pipeline configuration
```

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Shilpaj1994/BasicMLOps.git
cd BasicMLOps
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run:
```bash
python train.py
```

You can customize training parameters using command-line arguments:
```bash
python train.py
```

### Testing

Run unit tests with:
```bash
pytest test_model.py
```

The test suite includes the following test cases:

1. Model Architecture Tests:
   - Verifies model has less than 25,000 parameters
   - Checks input/output shape compatibility (28x28 input to 10 classes)
   - Tests if model can process standard MNIST image dimensions

2. Model Accuracy Tests:
   - Validates model achieves >95% accuracy on test set
   - Tests model performance on full validation dataset
   - Ensures model meets minimum accuracy threshold

3. Per-Digit Performance Tests:
   - Tests model accuracy for each digit (0-9)
   - Ensures >90% accuracy for every digit class
   - Validates balanced performance across all classes

4. Model Robustness Tests:
   - Tests model stability under random noise
   - Verifies at least 80% prediction consistency with noisy inputs
   - Validates model's resilience to input perturbations

5. Confidence Calibration Tests:
   - Verifies correlation between confidence and accuracy
   - Tests if high confidence (>0.9) predictions are accurate
   - Ensures >95% accuracy for high confidence predictions

This will generate a detailed coverage report showing which parts of the code are tested.

## CI/CD Pipeline

This project includes a GitHub Actions workflow that:
- Runs tests on every push and pull request
- Validates code formatting
- Executes the training pipeline
- Reports test coverage
- Saves the model to the artifacts folder


## Author

Shilpaj Bhalerao - [@Shilpaj1994](https://github.com/Shilpaj1994)
Project Link: [https://github.com/Shilpaj1994/BasicMLOps](https://github.com/Shilpaj1994/BasicMLOps)