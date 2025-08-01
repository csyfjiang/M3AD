# M3AD: Multi-Modal Multi-Scale Alzheimer's Disease Classification

A deep learning framework for Alzheimer's Disease classification and transition prediction using sMRI with advanced transformer architectures and multi-gate mixture of experts (MMoE) models.

## Overview

M3AD is a comprehensive framework designed for Alzheimer's Disease classification that leverages multi-modal neuroimaging data. The project implements state-of-the-art transformer architectures with Tok-MLP enhanced Swin Transformer v2 backbone and MMoE for robust and accurate AD classification.

## Repository Structure

```
M3AD/
├── configs/
│   └── swin_admoe/          # Configuration files for Swin-AdMoE model
├── data/                    # Data handling and loading utilities
├── data_preprocess/         # Data preprocessing scripts
├── models/                  # Model architectures and components
├── config.py               # Main configuration for 3-class classification
├── config_nine_label.py    # Configuration for 9-class classification
├── train.py                # Training script for 3-class model
├── train_nine_label.py     # Training script for 9-class model
├── trainer.py              # Main trainer class
├── trainer_nine_label.py   # Trainer for 9-class classification
├── eval.py                 # Evaluation utilities
├── lr_scheduler.py         # Learning rate scheduling
├── utils.py                # General utility functions
├── start_train_C3.sh       # Training script for 3-class classification
├── start_training_C9.sh    # Training script for 9-class classification
└── requirements.txt        # Project dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/csyfjiang/M3AD.git
cd M3AD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

Place your preprocessed neuroimaging data in the `data/` directory. The framework expects data to be organized by class labels (AD, MCI, CN for 3-class classification).

### Training

#### 3-Class Classification (AD/MCI/CN)
```bash
# Using shell script
bash start_train_C3.sh

# Or directly with Python
python train.py --config configs/swin_admoe/config.yaml
```

#### 9-Class Fine-grained Classification
```bash
# Using shell script
bash start_training_C9.sh

# Or directly with Python
python train_nine_label.py --config configs/swin_admoe/config_nine.yaml
```

### Evaluation

```bash
python eval.py --model_path path/to/trained/model --test_data path/to/test/data
```

## Configuration

### 3-Class Configuration (`config.py`)
- Standard AD classification setup
- Three output classes: AD, MCI, CN
- Optimized hyperparameters for balanced accuracy

### 9-Class Configuration (`config_nine_label.py`)
- Fine-grained classification setup
- Nine output classes for detailed cognitive assessment
- Specialized loss functions and evaluation metrics

## Model Components

### Tok-MLP enhanced Swin Transformer v2 with MMoE
- **Swin Transformer**: Hierarchical attention mechanism
- **Adaptive MoE**: Dynamic expert selection based on input characteristics
- **Multi-Scale Features**: Integration of local and global representations

### Training Features
- **Custom Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Mixed Precision Training**: Efficient memory usage and faster training
- **Data Augmentation**: Specialized augmentations for medical images
- **Cross-Validation**: Robust model evaluation

## Performance Metrics

The framework provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## Data Requirements

- **Input Format**: NIfTI (.nii.gz) files
- **Preprocessing**: Skull stripping, normalization, registration
- **Resolution**: Supports various spatial resolutions
- **Modalities**: T1-weighted MRI (extensible to other modalities)

## Acknowledgements

We gratefully acknowledge the foundational contributions of the open-source community, particularly the Microsoft Swin Transformer team (https://github.com/microsoft/Swin-Transformer), OpenMoE project (https://github.com/XueFuzhao/OpenMoE), M4 framework (https://github.com/Bigyehahaha/M4), and SimMIM implementation (https://github.com/microsoft/SimMIM), whose innovative architectures and methodologies enabled the development of this 3D medical image analysis framework.

