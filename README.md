# Food Freshness Quality Control

A deep learning notebook for automated food freshness detection using TensorFlow and MobileNetV2 transfer learning.

## Overview

This Jupyter notebook implements a CNN-based image classification system to automatically assess food freshness levels. By leveraging transfer learning with MobileNetV2, the model achieves 92% accuracy on the validation set while maintaining computational efficiency.

## Dataset

This project uses the [Food Freshness Dataset](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset) from Kaggle, which contains labeled images of food items at various stages of freshness.

### Dataset Structure
- **Classes**: Fresh, Stale, Spoiled
- **Format**: RGB images in various resolutions
- **Size**: ~5,000 images across all categories

## Technical Implementation

- **Framework**: TensorFlow 2.x / Keras
- **Architecture**: MobileNetV2 (pre-trained on ImageNet)
- **Approach**: Transfer learning with fine-tuning
- **Language**: Python 3.8+

## Notebook Contents

The notebook is structured into the following sections:

1. **Data Loading and Exploration**
   - Import required libraries
   - Load dataset from Kaggle
   - Exploratory data analysis and visualization

2. **Data Preprocessing**
   - Image resizing and normalization
   - Train/validation/test split
   - Data augmentation pipeline

3. **Model Development**
   - MobileNetV2 base model setup
   - Custom classification head
   - Model compilation and configuration

4. **Training**
   - Training loop with callbacks
   - Learning rate scheduling
   - Early stopping implementation

5. **Evaluation**
   - Performance metrics calculation
   - Confusion matrix visualization
   - Sample predictions display

6. **Results Analysis**
   - Model accuracy: 92% on validation set
   - Per-class performance breakdown
   - Error analysis

## Requirements

```python
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
pillow>=8.3.0
```

## Usage

1. **Download the dataset**:
   ```bash
   # Download from Kaggle (requires Kaggle API credentials)
   kaggle datasets download -d ulnnproject/food-freshness-dataset
   unzip food-freshness-dataset.zip
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook food_freshness_classification.ipynb
   ```

## Model Architecture Details

The implementation uses transfer learning with the following configuration:

- **Base Model**: MobileNetV2 (weights='imagenet', include_top=False)
- **Input Shape**: (224, 224, 3)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(3, activation='softmax')

## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20%

## Data Augmentation

To improve model generalization, the following augmentations are applied:
- Random rotation (range: ±20 degrees)
- Width and height shifts (range: 20%)
- Horizontal flipping
- Zoom (range: ±20%)
- Brightness adjustment

## Key Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 94.3% |
| Validation Accuracy | 92.1% |
| Test Accuracy | 91.8% |
| Average F1-Score | 0.916 |

### Per-Class Performance
- **Fresh**: Precision 0.95, Recall 0.93
- **Stale**: Precision 0.89, Recall 0.91
- **Spoiled**: Precision 0.92, Recall 0.94

## Potential Improvements

- Experiment with other pre-trained models (EfficientNet, ResNet)
- Implement ensemble methods
- Add more aggressive data augmentation
- Fine-tune more layers of the base model
- Implement gradient accumulation for larger effective batch sizes

## Citation

If you use this code, please cite the original dataset:
```
@dataset{food_freshness_2023,
  title={Food Freshness Dataset},
  author={ULNN Project},
  year={2023},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset}
}
```
