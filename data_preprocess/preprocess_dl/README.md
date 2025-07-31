# MRI Preprocessing Pipeline for Alzheimer's Disease Research

A comprehensive MRI preprocessing pipeline built with MONAI for Alzheimer's disease research, specifically designed to handle ADNI (Alzheimer's Disease Neuroimaging Initiative) datasets.

## Overview

This pipeline provides automated preprocessing of T1-weighted MRI images with features including background cropping, intensity normalization, slice filtering, and batch processing capabilities. It's optimized for Alzheimer's disease classification tasks involving AD (Alzheimer's Disease), CN (Cognitively Normal), and MCI (Mild Cognitive Impairment) groups.

## Features

- **Automated Background Cropping**: Removes unnecessary background regions while preserving brain tissue
- **Multiple Normalization Methods**: 
  - Min-Max normalization (0-1 range)
  - Z-score standardization
  - Percentile-based normalization (1%-99%)
- **Intelligent Slice Filtering**: Removes slices with insufficient brain content based on configurable thresholds
- **Batch Processing**: Process entire datasets efficiently
- **MONAI Integration**: Leverages MONAI's robust medical imaging transformations
- **Flexible Configuration**: Easy customization for different research needs

## Quick Start

### Single File Processing

```python
from mri_preprocessing import preprocess_mri_sample

# Process a single MRI file
output_path = preprocess_mri_sample(
    input_path="path/to/your/mri_file.nii.gz",
    output_dir="path/to/output/directory",
    normalization_method="minmax",
    slice_threshold=0.15
)
```

### Batch Processing

```python
from mri_preprocessing import batch_preprocess

# Process all files in a directory
processed_files = batch_preprocess(
    input_dir="path/to/input/directory",
    output_dir="path/to/output/directory",
    file_pattern="*_reg.nii.gz",
    normalization_method="minmax",
    slice_threshold=0.15
)
```

### Multiple Dataset Processing

```python
from mri_preprocessing import process_multiple_folders

# Configure multiple datasets
folder_configs = [
    {
        'input_dir': "path/to/AD/data",
        'output_dir': "path/to/AD/preprocessed",
        'label': 'AD (Alzheimer\'s Disease)'
    },
    {
        'input_dir': "path/to/CN/data",
        'output_dir': "path/to/CN/preprocessed",
        'label': 'CN (Cognitively Normal)'
    },
    {
        'input_dir': "path/to/MCI/data",
        'output_dir': "path/to/MCI/preprocessed",
        'label': 'MCI (Mild Cognitive Impairment)'
    }
]

# Process all datasets
process_multiple_folders(
    folder_configs=folder_configs,
    normalization_method="zscore",
    slice_threshold=0.45
)
```

## Configuration Options

### Normalization Methods

1. **Min-Max Normalization** (`"minmax"`):
   - Scales intensities to [0, 1] range
   - Preserves background as 0
   - Best for maintaining relative intensity differences

2. **Z-Score Standardization** (`"zscore"`):
   - Standardizes to zero mean and unit variance
   - Maps to [0, 1] range using 3-sigma clipping
   - Good for handling intensity variations across subjects

3. **Percentile Normalization** (`"percentile"`):
   - Uses 1st and 99th percentiles as bounds
   - Robust against outliers
   - Clips extreme values

### Slice Filtering Thresholds

- **0.05 (5%)**: Very permissive, keeps most slices
- **0.15 (15%)**: Balanced approach, good for most cases
- **0.45 (45%)**: Conservative, keeps only slices with substantial brain content

## Pipeline Steps

1. **Image Loading**: Load NIfTI files with metadata preservation
2. **Channel Formatting**: Ensure proper channel-first format
3. **Orientation Standardization**: Convert to RAS+ orientation
4. **Background Cropping**: Automatic foreground detection and cropping
5. **Slice Filtering**: Remove slices with insufficient content
6. **Intensity Normalization**: Apply selected normalization method
7. **Tensor Conversion**: Prepare for deep learning frameworks
8. **Output Saving**: Save with preserved metadata

## Output Format

- **File Format**: NIfTI (.nii.gz)
- **Naming Convention**: `{original_name}_preprocessed.nii.gz`
- **Metadata**: Original affine matrix and header information preserved
- **Intensity Range**: Depends on normalization method (typically [0, 1])

## Usage Tips

1. **Choose appropriate slice threshold**: 
   - Lower thresholds (0.05-0.15) for detailed analysis
   - Higher thresholds (0.3-0.5) for focusing on central brain regions

2. **Select normalization method based on downstream task**:
   - Use `minmax` for visualization and general preprocessing
   - Use `zscore` for machine learning models expecting standardized input
   - Use `percentile` when dealing with varying acquisition parameters

3. **Monitor processing logs**: Check for warnings about slice filtering and intensity ranges

## Performance Considerations

- **Memory Usage**: Large volumes may require significant RAM
- **Processing Time**: Depends on image size and number of files
- **Storage**: Preprocessed files maintain original resolution unless cropped

## Common Issues and Solutions

### Issue: "No slices meet threshold" warning
**Solution**: Lower the slice threshold or check input data quality

### Issue: Unexpected intensity ranges
**Solution**: Verify normalization method and check for extreme outliers in input data

### Issue: Affine matrix warnings
**Solution**: Ensure input files have proper spatial metadata

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex

```

## Contact

- **Author**: Yufeng Jiang
- **Email**: yufeng.jiang@connect.polyu.hk
- **Institution**: The Hong Kong Polytechnic University

## Acknowledgments

- Built with [MONAI](https://monai.io/) - Medical Open Network for AI
- Designed for [ADNI](http://adni.loni.usc.edu/) dataset processing
- Inspired by best practices in medical image preprocessing