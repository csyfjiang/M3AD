Based on the GitHub repository structure for M3AD, here's a comprehensive README.md:

# M3AD: Multi-Modal Multi-Scale Alzheimer's Disease Classification

A deep learning framework for Alzheimer's Disease classification using multi-modal neuroimaging data with advanced transformer architectures and mixture of experts (MoE) models.

## Overview

M3AD is a comprehensive framework designed for Alzheimer's Disease classification that leverages multi-modal neuroimaging data. The project implements state-of-the-art transformer architectures with Swin Transformer backbone and Adaptive Mixture of Experts (AdMoE) for robust and accurate AD classification.

## Key Features

- **Multi-Modal Learning**: Supports multiple neuroimaging modalities for comprehensive analysis
- **Multi-Scale Processing**: Hierarchical feature extraction at different spatial scales
- **Swin Transformer Architecture**: Efficient attention mechanism for medical image analysis
- **Adaptive Mixture of Experts (AdMoE)**: Dynamic expert selection for improved performance
- **Flexible Classification**: Supports both 3-class (AD/MCI/CN) and 9-class fine-grained classification
- **Comprehensive Evaluation**: Built-in evaluation metrics and visualization tools

## Architecture

The M3AD framework consists of:

1. **Swin Transformer Backbone**: Hierarchical vision transformer for feature extraction
2. **Adaptive MoE Module**: Dynamic expert routing for specialized feature processing
3. **Multi-Scale Fusion**: Integration of features from different spatial resolutions
4. **Classification Head**: Final prediction layer with configurable number of classes

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

### Swin Transformer with AdMoE
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
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification results

## Data Requirements

- **Input Format**: NIfTI (.nii.gz) files
- **Preprocessing**: Skull stripping, normalization, registration
- **Resolution**: Supports various spatial resolutions
- **Modalities**: T1-weighted MRI (extensible to other modalities)

## Advanced Features

### Mixture of Experts (MoE)
- Dynamic expert routing based on input characteristics
- Improved model capacity without proportional parameter increase
- Specialized experts for different brain regions or pathology patterns

### Multi-Scale Processing
- Hierarchical feature extraction at multiple spatial scales
- Integration of local detail and global context
- Adaptive pooling for variable input sizes

## Hyperparameter Tuning

Key hyperparameters that can be adjusted:
- **Learning Rate**: Initial learning rate and scheduling strategy
- **Batch Size**: Training batch size (memory dependent)
- **Expert Number**: Number of experts in MoE modules
- **Attention Heads**: Number of attention heads in transformer layers
- **Dropout Rate**: Regularization strength

## Citation

If you use M3AD in your research, please cite:

```bibtex
@article{m3ad2025,
  title={M3AD: Multi-Modal Multi-Scale Alzheimer's Disease Classification},
  author={Yufeng Jiang},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International License).

### You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

### Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes

For commercial licensing inquiries, please contact: [your-email@example.com]

See the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Yufeng Jiang
- **Email**: csyfjiang@gmail.com
- **GitHub**: [@csyfjiang](https://github.com/csyfjiang)

## Acknowledgments

- Built on top of Swin Transformer architecture
- Inspired by mixture of experts methodologies
- Developed for Alzheimer's Disease research community
- Thanks to all contributors and the open-source community
