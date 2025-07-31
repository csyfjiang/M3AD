# `group.py` will process all CSV files. Detailed explanation about checkpoint recovery:

## 🔍 How to Check Checkpoint Status

### 1. **Check Progress Log Files**
The program creates progress files in the E.g.`F:\data\ADNI\CN_preprocessed\progress_logs\` directory:
```
progress_logs/
├── ADNI_CN_1_progress.json
├── ADNI_CN_2_progress.json
├── ADNI_CN_3_progress.json
└── ...
```

### 2. **Check Progress File Contents**
Each JSON file contains detailed progress information:
```json
{
  "batch_info": {
    "csv_basename": "ADNI_CN_3",
    "total_files": 500,
    "completed": false  // true means completed, false means interrupted
  },
  "processed_files": [...],  // List of processed files
  "failed_files": [...],     // List of failed files
  "total_processed": 245,    // Number of processed files
  "total_failed": 12,        // Number of failed files
  "timestamp": "2025-06-04T15:30:45"
}
```
The checkpoint checking tool script is in `./utils/check_preprocess_progress.py`, you can run it to view the processing status of each CSV.

## 🔧 How Checkpoint Recovery Works

### **Automatic Checkpoint Recovery**
When you set `resume=True`, the program will:

1. **Check progress file for each CSV**
2. **Skip completed CSVs** (`completed: true`)
3. **Continue from breakpoint** for incomplete CSVs

### **Checkpoint Checking Methods**

#### **Method 1: Using Progress Check Tool**
Run the progress check script above, it will show:
- Which CSVs are completed ✅
- Which CSVs are in progress 🔄
- Which CSVs haven't started ⏳
- Detailed progress for each batch

#### **Method 2: Manually Check Progress Files**
```bash
# View progress log directory
ls F:\data\ADNI\CN_preprocessed\progress_logs\

# View specific batch progress
notepad F:\data\ADNI\CN_preprocessed\progress_logs\ADNI_CN_3_progress.json
```

#### **Method 3: Check Output File Count**
```python
import glob

# Check number of processed files
brain_mask_files = len(glob.glob(r"F:\data\ADNI\CN_preprocessed\brain_mask\*.nii.gz"))
reg_files = len(glob.glob(r"F:\data\ADNI\CN_preprocessed\reg\*.nii.gz"))

print(f"Brain mask files: {brain_mask_files}")
print(f"Registration files: {reg_files}")
```

## 🚀 Steps to Resume Processing

1. **Run progress check tool** to confirm checkpoint location
2. **Set `resume=True`**
3. **Rerun main program**

The program will automatically:
- Skip completed CSVs
- Continue from breakpoint of interrupted CSV
- Process remaining CSVs

This way you can safely resume from any checkpoint without reprocessing completed files!

## CSV File Structure Requirements

Each batch CSV file must include the following columns:

| Column Name | Description | Example |
|------------|-------------|---------|
| `index` | Sequential number | `0, 1, 2...` |
| `filename` | Name of the file | `ADNI_002.nii.gz` |
| `full_path` | Complete file path | `F:/data/ADNI/raw/ADNI_002.nii.gz` |
| `file_extension` | File extension | `.nii.gz` |
| `file_size_mb` | File size in megabytes | `15.6` |
| `directory` | Directory path | `F:/data/ADNI/raw` |
| `created_date` | File creation date | `2025-06-04 15:30:45` |

Example CSV format:
```csv
index,filename,full_path,file_extension,file_size_mb,directory,created_date
0,ADNI_002.nii.gz,F:/data/ADNI/raw/ADNI_002.nii.gz,.nii.gz,15.6,F:/data/ADNI/raw,2025-06-04 15:30:45
1,ADNI_003.nii.gz,F:/data/ADNI/raw/ADNI_003.nii.gz,.nii.gz,14.8,F:/data/ADNI/raw,2025-06-04 15:31:12
```