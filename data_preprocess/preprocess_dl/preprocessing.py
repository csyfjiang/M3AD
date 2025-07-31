"""
Description: 
Author: YufengJiang
Email: yufeng.jiang@connect.polyu.hk
Date: 2025/6/12
LastEditTime: 2025/6/12 17:27
Version: 1.0
"""
import os
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    CropForegroundd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityd,  # Add this for simple Min-Max normalization
    NormalizeIntensityd,
    ToTensord,
    Lambda
)
from monai.data import DataLoader, Dataset
import torch


def filter_slices_by_content(data, threshold=0.15):
    """
    Filter slices along Z-axis, removing slices with valid pixel ratio below threshold

    Args:
        data: Dictionary containing image data
        threshold: Valid pixel ratio threshold, default 15% (more suitable for Alzheimer's research)
    Returns:
        Processed data dictionary
    """
    image = data["image"]

    # Assume image shape is (C, H, W, D) or (H, W, D)
    # Ensure we handle channel dimension correctly
    if image.ndim == 4:  # Has channel dimension
        # Process each channel separately
        c, h, w, d = image.shape
        valid_slices = []

        # Calculate valid pixel ratio for each slice
        for z in range(d):
            slice_data = image[0, :, :, z]  # Assume single channel or use first channel for judgment
            total_pixels = h * w
            non_zero_pixels = np.sum(slice_data != 0)
            ratio = non_zero_pixels / total_pixels

            if ratio >= threshold:
                valid_slices.append(z)

        # Keep valid slices
        if len(valid_slices) > 0:
            image = image[:, :, :, valid_slices]
            print(f"Kept {len(valid_slices)}/{d} slices (threshold: {threshold * 100:.1f}%)")
        else:
            # If no valid slices, keep at least one
            print(f"Warning: No slices meet {threshold * 100:.1f}% threshold, keeping all slices")

    else:  # No channel dimension (H, W, D)
        h, w, d = image.shape
        valid_slices = []

        # Calculate valid pixel ratio for each slice
        for z in range(d):
            slice_data = image[:, :, z]
            total_pixels = h * w
            non_zero_pixels = np.sum(slice_data != 0)
            ratio = non_zero_pixels / total_pixels

            if ratio >= threshold:
                valid_slices.append(z)

        # Keep valid slices
        if len(valid_slices) > 0:
            image = image[:, :, valid_slices]
            print(f"Kept {len(valid_slices)}/{d} slices (threshold: {threshold * 100:.1f}%)")
        else:
            # If no valid slices, keep at least one
            print(f"Warning: No slices meet {threshold * 100:.1f}% threshold, keeping all slices")

    data["image"] = image
    return data


def preprocess_mri_sample(input_path, output_dir, normalization_method="minmax", slice_threshold=0.15):
    """
    Use MONAI to perform background cropping, intensity normalization and slice filtering on a single MRI sample

    Args:
        input_path: Input .nii.gz file path
        output_dir: Output directory
        normalization_method: Normalization method ("minmax", "zscore", "percentile")
        slice_threshold: Slice filtering threshold (valid pixel ratio)
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename (without extension)
    filename = os.path.basename(input_path)
    base_name = filename.replace('.nii.gz', '')
    output_filename = f"{base_name}_preprocessed.nii.gz"
    output_path = os.path.join(output_dir, output_filename)

    # Prepare data dictionary
    data_dict = [{"image": input_path, "output_path": output_path}]

    # Build preprocessing pipeline
    transforms_list = [
        # 1. Load image
        LoadImaged(keys=["image"], image_only=True),

        # 2. Ensure channel dimension is first
        EnsureChannelFirstd(keys=["image"]),

        # 3. Standardize orientation (RAS+)
        Orientationd(keys=["image"], axcodes="RAS"),

        # 4. Background cropping - automatically detect foreground area and crop
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=5,  # Keep 5 pixels margin around boundary
            allow_smaller=True
        ),

        # 5. Filter slices along Z-axis (remove slices with valid pixels below threshold)
        Lambda(func=lambda x: filter_slices_by_content(x, threshold=slice_threshold)),
    ]

    # 6. Add normalization based on selected method
    if normalization_method == "minmax":
        # Min-Max normalization to [0,1] - keep background as 0
        transforms_list.append(
            ScaleIntensityd(
                keys=["image"],
                minv=0.0,  # Output minimum value
                maxv=1.0,  # Output maximum value
            )
        )
        print("Using Min-Max normalization (background remains 0)")

    elif normalization_method == "percentile":
        # Percentile-based normalization - more robust, ignores extreme values
        transforms_list.append(
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1,  # Use 1st percentile as minimum
                upper=99,  # Use 99th percentile as maximum
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False
            )
        )
        print("Using percentile normalization (1%-99%)")

    elif normalization_method == "zscore":
        # Z-score standardization - Note: produces negative values, background may not be 0
        transforms_list.extend([
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,  # Only normalize non-zero pixels
                channel_wise=True
            ),
            # Map Z-score results to [0,1]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-3,  # 3 standard deviation range for Z-score
                a_max=3,
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        ])
        print("Using Z-score standardization (Note: background may become gray)")

    # 7. Convert to tensor
    transforms_list.append(ToTensord(keys=["image"]))

    # Create complete transformation pipeline
    transforms = Compose(transforms_list)

    # Create dataset and dataloader
    dataset = Dataset(data=data_dict, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process data
    for batch_data in dataloader:
        processed_image = batch_data["image"]

        # Remove batch dimension and channel dimension for saving
        processed_image = processed_image.squeeze(0).squeeze(0)

        # Convert back to numpy array
        if isinstance(processed_image, torch.Tensor):
            processed_image = processed_image.numpy()

        # Save processed image
        save_preprocessed_image(processed_image, input_path, output_path)

        print(f"\nPreprocessing completed!")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Original shape: {nib.load(input_path).get_fdata().shape}")
        print(f"Processed shape: {processed_image.shape}")
        print(f"Intensity range: [{processed_image.min():.4f}, {processed_image.max():.4f}]")

        # Check background values
        background_pixels = processed_image[processed_image < 0.01]
        if len(background_pixels) > 0:
            print(f"Background pixel value: {background_pixels.mean():.4f} (should be close to 0)")

        return output_path


def save_preprocessed_image(processed_array, original_path, output_path):
    """
    Save preprocessed image while preserving original affine and header information

    Note: Since slice filtering may change image dimensions, affine matrix may need updating
    """
    # Load original image to get affine and header
    original_nii = nib.load(original_path)
    original_affine = original_nii.affine.copy()

    # If Z-axis dimension changed, may need to adjust affine matrix
    # Simplified handling here, assuming slices are continuous

    # Create new NIfTI image
    processed_nii = nib.Nifti1Image(
        processed_array,
        affine=original_affine,
        header=original_nii.header
    )

    # Save
    nib.save(processed_nii, output_path)


import os
from pathlib import Path


def batch_preprocess(input_dir, output_dir, file_pattern="*_reg.nii.gz",
                     normalization_method="minmax", slice_threshold=0.15):
    """
    Batch process multiple files

    Args:
        input_dir: Input directory
        output_dir: Output directory
        file_pattern: File matching pattern
        normalization_method: Normalization method
        slice_threshold: Slice filtering threshold
    """
    import glob

    # Find all matching files
    search_pattern = os.path.join(input_dir, file_pattern)
    files = glob.glob(search_pattern)

    print(f"Found {len(files)} files to process")
    print(f"Normalization method: {normalization_method}")
    print(f"Slice filtering threshold: {slice_threshold * 100:.1f}%")

    processed_files = []
    for file_path in files:
        try:
            output_path = preprocess_mri_sample(
                file_path,
                output_dir,
                normalization_method=normalization_method,
                slice_threshold=slice_threshold
            )
            processed_files.append(output_path)
            print(f"✓ Processing completed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Processing failed: {os.path.basename(file_path)}, Error: {str(e)}")

    return processed_files

def process_multiple_folders(folder_configs, file_pattern="*_reg.nii.gz",
                             normalization_method="minmax", slice_threshold=0.05):
    """
    Batch process MRI data from multiple folders

    Args:
        folder_configs: Configuration list containing input/output paths
        file_pattern: File matching pattern
        normalization_method: Normalization method
        slice_threshold: Slice filtering threshold
    """
    total_processed = 0
    total_failed = 0

    for config in folder_configs:
        input_dir = config['input_dir']
        output_dir = config['output_dir']
        label = config.get('label', os.path.basename(input_dir))

        print(f"\n{'=' * 60}")
        print(f"Processing {label} dataset")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}")

        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"❌ Error: Input directory does not exist - {input_dir}")
            continue

        # Create output directory (if it doesn't exist)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory ready")

        # Process current folder
        try:
            processed_files = batch_preprocess(
                input_dir=input_dir,
                output_dir=output_dir,
                file_pattern=file_pattern,
                normalization_method=normalization_method,
                slice_threshold=slice_threshold
            )

            success_count = len(processed_files)
            total_processed += success_count

            print(f"\n✅ {label} processing completed! Successfully processed {success_count} files")

        except Exception as e:
            print(f"\n❌ {label} processing failed! Error: {str(e)}")
            total_failed += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Batch processing completed!")
    print(f"Total files processed successfully: {total_processed}")
    print(f"Failed folders: {total_failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Define folder configurations to process
    folder_configs = [
        {
            'input_dir': r"F:\data\ADNI\AD_preprocessed\reg",
            'output_dir': r"F:\data\ADNI\AD_preprocessed\preprocessed_zscore_st45",
            'label': 'AD (Alzheimer\'s Disease)'
        },
        {
            'input_dir': r"F:\data\ADNI\CN_preprocessed\reg",
            'output_dir': r"F:\data\ADNI\CN_preprocessed\preprocessed_zscore_st45",
            'label': 'CN (Cognitively Normal)'
        },
        {
            'input_dir': r"F:\data\ADNI\MCI_preprocessed\reg",
            'output_dir': r"F:\data\ADNI\MCI_preprocessed\preprocessed_zscore_st45",
            'label': 'MCI (Mild Cognitive Impairment)'
        }
    ]

    # Process all folders
    process_multiple_folders(
        folder_configs=folder_configs,
        file_pattern="*_reg.nii.gz",
        normalization_method="zscore",
        slice_threshold=0.45  # Use 45% threshold
    )

    # If you want a simpler approach, you can also use a direct loop:
    """
    # Simple loop approach
    folders = [
        (r"F:\data\ADNI\AD_preprocessed\reg", r"F:\data\ADNI\AD_preprocessed\preprocessed"),
        (r"F:\data\ADNI\CN_preprocessed\reg", r"F:\data\ADNI\CN_preprocessed\preprocessed"),
        (r"F:\data\ADNI\MCI_preprocessed\reg", r"F:\data\ADNI\MCI_preprocessed\preprocessed")
    ]

    for input_dir, output_dir in folders:
        print(f"\nProcessing: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        batch_preprocess(
            input_dir=input_dir,
            output_dir=output_dir,
            normalization_method="minmax",
            slice_threshold=0.05
        )
    """