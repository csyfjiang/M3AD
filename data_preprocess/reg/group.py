CUSTOM_TEMP_DIR = 'F:/temp'
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)


os.environ['TMPDIR'] = CUSTOM_TEMP_DIR      # Linux/Unix
os.environ['TMP'] = CUSTOM_TEMP_DIR         # Windows
os.environ['TEMP'] = CUSTOM_TEMP_DIR        # Windows
os.environ['TEMPDIR'] = CUSTOM_TEMP_DIR     # 备用


tempfile.tempdir = CUSTOM_TEMP_DIR

# print(f"  Custom temporary directory set to: {CUSTOM_TEMP_DIR}")
# print(f"  Python tempfile directory: {tempfile.gettempdir()}")

import sys
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import time
import gc
import ants
import warnings
import glob
from datetime import datetime, timedelta
import traceback
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# 导入用于方向校正的库
from monai.transforms import Orientation
from monai.data import MetaTensor

# 如果torch中没有GradScaler，从正确位置导入
if not hasattr(torch, 'GradScaler'):
    try:
        from torch.cuda.amp import GradScaler
        torch.GradScaler = GradScaler
        # print("GradScaler patched successfully")
    except ImportError:
        print("Unable to patch GradScaler, please update PyTorch")

# 现在继续HD-BET初始化
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
from HD_BET.checkpoint_download import maybe_download_parameters


def setup_directories(base_output_dir):
    """
    设置输出目录结构
    """
    brain_mask_dir = os.path.join(base_output_dir, 'brain_mask')
    reg_dir = os.path.join(base_output_dir, 'reg')
    progress_dir = os.path.join(base_output_dir, 'progress_logs')
    temp_dir = os.path.join(base_output_dir, 'temp_processing')  # 专用临时目录

    os.makedirs(brain_mask_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"✅ Output directories created:")
    print(f"   Brain mask: {brain_mask_dir}")
    print(f"   Registration: {reg_dir}")
    print(f"   Progress logs: {progress_dir}")
    print(f"   Processing temp: {temp_dir}")

    return brain_mask_dir, reg_dir, progress_dir, temp_dir


def load_csv_file_list(csv_file):
    """
    从CSV文件中加载文件列表

    Parameters:
    -----------
    csv_file : str
        CSV文件路径

    Returns:
    --------
    pd.DataFrame : 包含文件信息的DataFrame
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV: {csv_file}")
        print(f"   Records: {len(df)}")

        # 检查必要的列
        required_columns = ['full_path', 'filename']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None

        # 过滤存在的文件
        existing_files = df[df['full_path'].apply(os.path.exists)]
        missing_count = len(df) - len(existing_files)

        if missing_count > 0:
            print(f"⚠️  Warning: {missing_count} files not found on disk")

        print(f"📁 Valid files to process: {len(existing_files)}")
        return existing_files

    except Exception as e:
        print(f"❌ Error loading CSV {csv_file}: {e}")
        return None


def save_batch_progress(progress_file, batch_info, processed_files, failed_files):
    """
    保存批次处理进度
    """
    progress_data = {
        'batch_info': batch_info,
        'processed_files': processed_files,
        'failed_files': failed_files,
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(processed_files),
        'total_failed': len(failed_files)
    }

    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_batch_progress(progress_file):
    """
    加载批次处理进度
    """
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def process_single_nii_file(input_file, brain_mask_dir, reg_dir, temp_dir, predictor=None, pbar=None):
    """
    处理单个NII文件的完整流程

    Parameters:
    -----------
    input_file : str
        输入文件路径
    brain_mask_dir : str
        脑区提取结果保存目录
    reg_dir : str
        配准结果保存目录
    temp_dir : str
        专用临时文件目录
    predictor : object
        HD-BET预测器（可选，用于复用）
    pbar : tqdm
        进度条对象
    """
    # 获取文件名（不含扩展名）
    filename = os.path.basename(input_file)
    base_name = os.path.splitext(filename)[0]  # 去掉.nii

    # 为每个文件创建唯一的临时文件名（避免并发冲突）
    unique_id = f"{base_name}_{int(time.time() * 1000000) % 1000000}"

    # 设置输出文件路径
    brain_mask_output = os.path.join(brain_mask_dir, f"{base_name}_brain_mask.nii.gz")
    reg_output = os.path.join(reg_dir, f"{base_name}_reg.nii.gz")

    # 设置临时文件路径（使用专用临时目录）
    temp_ras_file = os.path.join(temp_dir, f"{unique_id}_temp_ras.nii")
    temp_n4_file = os.path.join(temp_dir, f"{unique_id}_temp_n4.nii.gz")
    temp_brain_file = os.path.join(temp_dir, f"{unique_id}_temp_brain.nii.gz")

    try:
        # 步骤1: 方向校正
        if pbar:
            pbar.set_postfix_str(f"Orientation correction: {filename}")

        # 读取原始图像
        img = nib.load(input_file)
        data = img.get_fdata()

        # 获取原始方向
        original_orientation = nib.aff2axcodes(img.affine)

        # 方向校正（如果需要）
        if original_orientation != ('R', 'A', 'S'):
            data_tensor = torch.from_numpy(data).unsqueeze(0)
            meta_data = MetaTensor(data_tensor, affine=img.affine)
            orient = Orientation(axcodes="RAS")
            transformed = orient(meta_data)
            data_ras = transformed.numpy()[0]
            affine_ras = transformed.affine

            # 创建临时文件
            img_ras = nib.Nifti1Image(data_ras, affine_ras)
            nib.save(img_ras, temp_ras_file)
            working_file = temp_ras_file
        else:
            working_file = input_file

        # 步骤2: 读取ANTs图像
        if pbar:
            pbar.set_postfix_str(f"Reading with ANTs: {filename}")

        original_img = ants.image_read(working_file)

        # 步骤3: N4偏置场校正
        if pbar:
            pbar.set_postfix_str(f"N4 bias correction: {filename}")

        corrected_img = ants.n4_bias_field_correction(
            original_img,
            mask=None,
            shrink_factor=4,
            convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},
            spline_param=200
        )

        # 步骤4: HD-BET脑区提取
        if pbar:
            pbar.set_postfix_str(f"Brain extraction: {filename}")

        # 保存N4校正后的临时文件
        ants.image_write(corrected_img, temp_n4_file)

        try:
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 如果没有提供预测器，创建新的
            if predictor is None:
                predictor = get_hdbet_predictor(
                    use_tta=False,
                    device=device,
                    verbose=False
                )

            # 执行brain extraction
            hdbet_predict(
                input_file_or_folder=temp_n4_file,
                output_file_or_folder=temp_brain_file,
                predictor=predictor,
                keep_brain_mask=True,
                compute_brain_extracted_image=True
            )

            # 读取结果
            brain_img = ants.image_read(temp_brain_file)

            # 保存脑区提取结果
            ants.image_write(brain_img, brain_mask_output)
            brain_extraction_success = True

        except Exception as e:
            # 使用ANTsPy备用方法
            try:
                brain_extraction_result = ants.brain_extraction(corrected_img, modality="t1")
                brain_img = brain_extraction_result['brain_image']
                ants.image_write(brain_img, brain_mask_output)
                brain_extraction_success = True
            except Exception as e2:
                brain_img = corrected_img
                ants.image_write(brain_img, brain_mask_output)
                brain_extraction_success = False

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 步骤5: 配准到MNI空间
        if pbar:
            pbar.set_postfix_str(f"Registration: {filename}")

        try:
            # 获取MNI模板
            template = ants.image_read(ants.get_ants_data('mni'))

            registration_result = ants.registration(
                fixed=template,
                moving=brain_img,
                type_of_transform='SyN',
                verbose=False
            )

            registered_img = registration_result['warpedmovout']
            ants.image_write(registered_img, reg_output)
            registration_success = True

        except Exception as e:
            ants.image_write(brain_img, reg_output)
            registration_success = False

        # 清理临时文件（使用新的路径）
        temp_files = [temp_ras_file, temp_n4_file, temp_brain_file]

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if pbar:
                        pbar.set_postfix_str(f"Cleaned temp: {os.path.basename(temp_file)}")
                except:
                    pass

        # 清理内存
        gc.collect()

        return {
            'status': 'success',
            'brain_extraction_success': brain_extraction_success,
            'registration_success': registration_success,
            'brain_mask_file': brain_mask_output,
            'reg_file': reg_output
        }

    except Exception as e:
        # 即使出错也要清理临时文件
        temp_files = [temp_ras_file, temp_n4_file, temp_brain_file]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

        return {
            'status': 'failed',
            'error': str(e),
            'brain_extraction_success': False,
            'registration_success': False,
            'brain_mask_file': None,
            'reg_file': None
        }


def process_csv_batch(csv_file, brain_mask_dir, reg_dir, progress_dir, temp_dir, predictor=None, resume=False):
    """
    处理单个CSV批次

    Parameters:
    -----------
    csv_file : str
        CSV文件路径
    brain_mask_dir : str
        脑区提取结果保存目录
    reg_dir : str
        配准结果保存目录
    progress_dir : str
        进度记录目录
    predictor : object
        HD-BET预测器
    resume : bool
        是否从上次中断处继续

    Returns:
    --------
    dict : 批次处理结果
    """

    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    progress_file = os.path.join(progress_dir, f"{csv_basename}_progress.json")

    print(f"\n🔄 Processing CSV batch: {csv_basename}")
    print(f"   CSV file: {csv_file}")
    print(f"   Progress file: {progress_file}")

    # 加载CSV文件
    df = load_csv_file_list(csv_file)
    if df is None or df.empty:
        return {
            'csv_file': csv_file,
            'status': 'failed',
            'error': 'Failed to load CSV or no valid files',
            'total_files': 0,
            'processed': 0,
            'failed': 0
        }

    # 检查是否需要恢复进度
    processed_files = []
    failed_files = []
    start_index = 0

    if resume:
        progress_data = load_batch_progress(progress_file)
        if progress_data:
            processed_files = progress_data.get('processed_files', [])
            failed_files = progress_data.get('failed_files', [])
            start_index = len(processed_files) + len(failed_files)
            print(f"📋 Resuming from index {start_index} (processed: {len(processed_files)}, failed: {len(failed_files)})")

    # 获取待处理的文件列表
    files_to_process = df.iloc[start_index:].copy()

    if files_to_process.empty:
        print(f"✅ All files in {csv_basename} already processed")
        return {
            'csv_file': csv_file,
            'status': 'completed',
            'total_files': len(df),
            'processed': len(processed_files),
            'failed': len(failed_files),
            'skipped': 0
        }

    print(f"📊 Batch statistics:")
    print(f"   Total files in CSV: {len(df)}")
    print(f"   Already processed: {len(processed_files)}")
    print(f"   Already failed: {len(failed_files)}")
    print(f"   Remaining to process: {len(files_to_process)}")

    # 处理文件
    batch_start_time = time.time()

    with tqdm(total=len(files_to_process),
              desc=f"🧠 Processing {csv_basename}",
              unit="file",
              ncols=120) as pbar:

        for idx, (_, row) in enumerate(files_to_process.iterrows()):
            file_path = row['full_path']
            filename = row['filename']

            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    failed_files.append({
                        'file': file_path,
                        'error': 'File not found',
                        'timestamp': datetime.now().isoformat()
                    })
                    pbar.set_postfix_str(f"Skipped: {filename} (not found)")
                    pbar.update(1)
                    continue

                # 处理文件
                result = process_single_nii_file(file_path, brain_mask_dir, reg_dir, temp_dir, predictor, pbar)

                if result['status'] == 'success':
                    processed_files.append({
                        'file': file_path,
                        'filename': filename,
                        'brain_extraction_success': result['brain_extraction_success'],
                        'registration_success': result['registration_success'],
                        'brain_mask_file': result['brain_mask_file'],
                        'reg_file': result['reg_file'],
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    failed_files.append({
                        'file': file_path,
                        'filename': filename,
                        'error': result['error'],
                        'timestamp': datetime.now().isoformat()
                    })

                # 每10个文件保存一次进度
                if (idx + 1) % 10 == 0:
                    save_batch_progress(progress_file, {
                        'csv_file': csv_file,
                        'csv_basename': csv_basename,
                        'total_files': len(df),
                        'start_time': batch_start_time
                    }, processed_files, failed_files)

                    # 清理内存
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 更新进度条
                success_rate = len(processed_files) / (len(processed_files) + len(failed_files)) * 100 if (len(processed_files) + len(failed_files)) > 0 else 0
                pbar.set_postfix_str(f"Success: {success_rate:.1f}%")
                pbar.update(1)

            except Exception as e:
                failed_files.append({
                    'file': file_path,
                    'filename': filename,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                pbar.set_postfix_str(f"Error: {filename}")
                pbar.update(1)

    # 保存最终进度
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time

    final_progress = {
        'csv_file': csv_file,
        'csv_basename': csv_basename,
        'total_files': len(df),
        'start_time': batch_start_time,
        'end_time': batch_end_time,
        'duration_seconds': batch_duration,
        'completed': True
    }

    save_batch_progress(progress_file, final_progress, processed_files, failed_files)

    return {
        'csv_file': csv_file,
        'csv_basename': csv_basename,
        'status': 'completed',
        'total_files': len(df),
        'processed': len(processed_files),
        'failed': len(failed_files),
        'duration_minutes': batch_duration / 60
    }


def batch_process_from_csvs(csv_dir, output_dir, resume=False):
    """
    从CSV文件批量处理ADNI数据

    Parameters:
    -----------
    csv_dir : str
        包含CSV文件的目录
    output_dir : str
        输出目录路径
    resume : bool
        是否从上次中断处继续
    """

    print("🚀 ADNI CSV Batch Preprocessing Pipeline")
    print("=" * 80)
    print(f"CSV directory: {csv_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resume mode: {resume}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 检查CSV目录
    if not os.path.exists(csv_dir):
        print(f"❌ Error: CSV directory does not exist: {csv_dir}")
        return

    # 设置输出目录
    brain_mask_dir, reg_dir, progress_dir, temp_dir = setup_directories(output_dir)

    # 获取所有CSV文件
    csv_pattern = os.path.join(csv_dir, "ADNI_CN_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"❌ No ADNI_CN_*.csv files found in {csv_dir}")
        return

    print(f"📁 Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"   📄 {os.path.basename(csv_file)}")

    # 初始化HD-BET预测器
    print("\n🔧 Initializing HD-BET predictor...")
    try:
        maybe_download_parameters()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        predictor = get_hdbet_predictor(
            use_tta=False,
            device=device,
            verbose=False
        )
        print("✅ HD-BET predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize HD-BET predictor: {e}")
        print("   Will use ANTsPy as backup for all files")
        predictor = None

    # 处理每个CSV文件
    total_start_time = time.time()
    all_results = []

    for csv_idx, csv_file in enumerate(csv_files):
        print(f"\n{'='*60}")
        print(f"📋 Processing CSV {csv_idx + 1}/{len(csv_files)}")
        print(f"{'='*60}")

        try:
            result = process_csv_batch(csv_file, brain_mask_dir, reg_dir, progress_dir, temp_dir, predictor, resume)
            all_results.append(result)

            print(f"\n✅ Completed: {result['csv_basename']}")
            print(f"   Files processed: {result['processed']}")
            print(f"   Files failed: {result['failed']}")
            if 'duration_minutes' in result:
                print(f"   Duration: {result['duration_minutes']:.1f} minutes")

        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            all_results.append({
                'csv_file': csv_file,
                'status': 'error',
                'error': str(e)
            })

    # 最终统计
    total_duration = time.time() - total_start_time

    print(f"\n{'='*80}")
    print("🎊 ALL CSV BATCHES COMPLETED!")
    print(f"{'='*80}")
    print(f"📊 Final Statistics:")

    total_processed = sum(r.get('processed', 0) for r in all_results)
    total_failed = sum(r.get('failed', 0) for r in all_results)
    total_files = sum(r.get('total_files', 0) for r in all_results)
    successful_batches = sum(1 for r in all_results if r.get('status') == 'completed')

    print(f"   CSV files processed: {successful_batches}/{len(csv_files)}")
    print(f"   Total files: {total_files}")
    print(f"   Successfully processed: {total_processed}")
    print(f"   Failed: {total_failed}")
    print(f"   Success rate: {total_processed/total_files*100:.1f}%" if total_files > 0 else "   Success rate: 0%")
    print(f"")
    print(f"⏱️  Total processing time: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"📁 Output directories:")
    print(f"   Brain mask results: {brain_mask_dir}")
    print(f"   Registration results: {reg_dir}")
    print(f"   Progress logs: {progress_dir}")
    print("=" * 80)


def main():
    """
    主函数
    """
    print(f"🗂️  验证临时目录设置:")
    print(f"   系统TEMP目录: {os.environ.get('TEMP', 'Not Set')}")
    print(f"   Python tempfile目录: {tempfile.gettempdir()}")
    print(f"   可用空间检查...")

    # 检查D盘可用空间
    if os.path.exists('D:'):
        import shutil
        total, used, free = shutil.disk_usage('D:')
        print(f"   D盘可用空间: {free // (1024**3)} GB")
        if free < 10 * (1024**3):  # 少于10GB
            print(f"   ⚠️  警告: D盘可用空间不足10GB，建议清理空间")

    # 设置路径
    csv_dir = r"F:\data\ADNI\CN_csv_batches"  # CSV文件目录
    output_dir = r"F:\data\ADNI\CN_preprocessed"  # 输出目录

    # 是否从上次中断处继续 (True/False)
    resume = True

    print(f"\n📁 处理目录:")
    print(f"   CSV输入目录: {csv_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   临时文件将保存到: {CUSTOM_TEMP_DIR}")
    print(f"   专用处理临时目录: {output_dir}/temp_processing")

    # 执行批量处理
    batch_process_from_csvs(csv_dir, output_dir, resume)

    # 最终清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 清理残留临时文件
    try:
        temp_files = glob.glob(os.path.join(CUSTOM_TEMP_DIR, "*"))
        if temp_files:
            print(f"\n🧹 清理残留临时文件: {len(temp_files)} 个")
            for temp_file in temp_files:
                try:
                    if os.path.isfile(temp_file):
                        os.remove(temp_file)
                except:
                    pass
    except:
        pass

    print("🎉 All processing completed successfully!")


if __name__ == "__main__":
    main()