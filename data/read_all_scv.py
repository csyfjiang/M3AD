"""
Description: 
Author: JeffreyJ
Date: 2025/7/15
LastEditTime: 2025/7/15 13:57
Version: 1.0
"""
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def analyze_dataset_labels(data_path):
    """
    分析数据集中所有npz文件的标签分布

    Args:
        data_path: 数据集路径
    """
    print(f"正在分析数据集: {data_path}")
    print("=" * 60)

    # 存储所有标签
    all_labels = []
    all_change_labels = []

    # 获取所有npz文件
    npz_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))

    print(f"找到 {len(npz_files)} 个npz文件")

    # 统计错误文件
    error_files = []

    # 遍历所有npz文件
    for file_path in tqdm(npz_files, desc="读取npz文件"):
        try:
            # 加载npz文件
            data = np.load(file_path, allow_pickle=True)

            # 检查是否包含required keys
            if 'label' not in data or 'change_label' not in data:
                print(f"⚠️  文件 {file_path} 缺少必要的标签字段")
                error_files.append(file_path)
                continue

            # 提取标签
            label = data['label']
            change_label = data['change_label']

            # 确保是标量值
            if np.isscalar(label):
                all_labels.append(int(label))
            elif hasattr(label, 'item'):
                all_labels.append(int(label.item()))
            else:
                all_labels.append(int(label))

            if np.isscalar(change_label):
                all_change_labels.append(int(change_label))
            elif hasattr(change_label, 'item'):
                all_change_labels.append(int(change_label.item()))
            else:
                all_change_labels.append(int(change_label))

        except Exception as e:
            print(f"❌ 读取文件 {file_path} 时出错: {e}")
            error_files.append(file_path)

    # 统计标签分布
    label_counts = Counter(all_labels)
    change_label_counts = Counter(all_change_labels)

    print(f"\n成功读取 {len(all_labels)} 个样本")
    if error_files:
        print(f"❌ {len(error_files)} 个文件读取失败")

    # 打印统计结果
    print("\n" + "=" * 60)
    print("DIAGNOSIS LABEL 分布 (1=CN, 2=MCI, 3=AD):")
    print("=" * 60)

    diagnosis_names = {1: 'CN', 2: 'MCI', 3: 'AD'}
    total_samples = len(all_labels)

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total_samples) * 100
        name = diagnosis_names.get(label, f'Unknown-{label}')
        print(f"  {label} ({name:3s}): {count:6d} 样本 ({percentage:5.1f}%)")

    print(f"\n  总计: {total_samples} 样本")

    print("\n" + "=" * 60)
    print("CHANGE LABEL 分布 (1-7个类别):")
    print("=" * 60)

    change_names = {
        1: 'Stable CN→CN',
        2: 'Stable MCI→MCI',
        3: 'Stable AD→AD',
        4: 'Conv CN→MCI',
        5: 'Conv MCI→AD',
        6: 'Conv CN→AD',
        7: 'Rev MCI→CN'
    }

    total_change_samples = len(all_change_labels)

    for label in range(1, 8):  # 确保显示所有期望的标签
        count = change_label_counts.get(label, 0)
        percentage = (count / total_change_samples) * 100 if total_change_samples > 0 else 0
        name = change_names.get(label, f'Unknown-{label}')
        status = "✓" if count > 0 else "❌"
        print(f"  {label} ({name:15s}): {count:6d} 样本 ({percentage:5.1f}%) {status}")

    print(f"\n  总计: {total_change_samples} 样本")

    # 检查缺失的标签
    expected_diagnosis_labels = {1, 2, 3}
    expected_change_labels = {1, 2, 3, 4, 5, 6, 7}

    actual_diagnosis_labels = set(label_counts.keys())
    actual_change_labels = set(change_label_counts.keys())

    missing_diagnosis = expected_diagnosis_labels - actual_diagnosis_labels
    missing_change = expected_change_labels - actual_change_labels

    print("\n" + "=" * 60)
    print("缺失标签检查:")
    print("=" * 60)

    if missing_diagnosis:
        print(f"❌ 缺失的 diagnosis 标签: {missing_diagnosis}")
    else:
        print("✓ 所有 diagnosis 标签都存在")

    if missing_change:
        print(f"❌ 缺失的 change 标签: {missing_change}")
        for label in missing_change:
            print(f"   - 标签 {label}: {change_names.get(label, 'Unknown')}")
    else:
        print("✓ 所有 change 标签都存在")

    # 异常标签检查
    unexpected_diagnosis = actual_diagnosis_labels - expected_diagnosis_labels
    unexpected_change = actual_change_labels - expected_change_labels

    if unexpected_diagnosis:
        print(f"⚠️  意外的 diagnosis 标签: {unexpected_diagnosis}")
    if unexpected_change:
        print(f"⚠️  意外的 change 标签: {unexpected_change}")

    return label_counts, change_label_counts, error_files


def create_visualization(label_counts, change_label_counts, save_path=None):
    """
    创建标签分布的可视化图表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Diagnosis labels 柱状图
    diagnosis_names = {1: 'CN', 2: 'MCI', 3: 'AD'}
    diag_labels = []
    diag_counts = []
    diag_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿

    for label in sorted(label_counts.keys()):
        diag_labels.append(f"{label}\n({diagnosis_names.get(label, 'Unknown')})")
        diag_counts.append(label_counts[label])

    bars1 = ax1.bar(diag_labels, diag_counts, color=diag_colors[:len(diag_labels)])
    ax1.set_title('Diagnosis Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_xlabel('Diagnosis Labels', fontsize=12)

    # 在柱子上添加数值
    for bar, count in zip(bars1, diag_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # Change labels 柱状图
    change_names = {
        1: 'Stable CN→CN',
        2: 'Stable MCI→MCI',
        3: 'Stable AD→AD',
        4: 'Conv CN→MCI',
        5: 'Conv MCI→AD',
        6: 'Conv CN→AD',
        7: 'Rev MCI→CN'
    }

    change_labels = []
    change_counts = []
    change_colors = []

    # 使用不同颜色区分不同类型
    color_map = {
        1: '#2E8B57', 2: '#2E8B57', 3: '#2E8B57',  # Stable: 深绿
        4: '#DC143C', 5: '#DC143C', 6: '#DC143C',  # Conversion: 深红
        7: '#4169E1'  # Reversion: 蓝色
    }

    for label in range(1, 8):
        count = change_label_counts.get(label, 0)
        change_labels.append(f"{label}")
        change_counts.append(count)
        change_colors.append(color_map.get(label, '#808080'))

    bars2 = ax2.bar(change_labels, change_counts, color=change_colors)
    ax2.set_title('Change Label Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_xlabel('Change Labels', fontsize=12)

    # 在柱子上添加数值
    for i, (bar, count) in enumerate(zip(bars2, change_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + max(change_counts) * 0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

        # 在x轴下方添加标签说明
        label_num = i + 1
        name = change_names.get(label_num, '')
        ax2.text(bar.get_x() + bar.get_width() / 2., -max(change_counts) * 0.15,
                 name, ha='center', va='top', fontsize=8, rotation=45)

    # 调整子图间距
    plt.tight_layout()

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Stable'),
        Patch(facecolor='#DC143C', label='Conversion'),
        Patch(facecolor='#4169E1', label='Reversion')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


def save_statistics_to_file(label_counts, change_label_counts, error_files, save_path):
    """
    将统计结果保存到文件
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("数据集标签分布统计报告\n")
        f.write("=" * 60 + "\n\n")

        # Diagnosis labels
        f.write("DIAGNOSIS LABEL 分布:\n")
        f.write("-" * 30 + "\n")
        diagnosis_names = {1: 'CN', 2: 'MCI', 3: 'AD'}
        total_samples = sum(label_counts.values())

        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            name = diagnosis_names.get(label, f'Unknown-{label}')
            f.write(f"{label} ({name}): {count} 样本 ({percentage:.1f}%)\n")

        f.write(f"总计: {total_samples} 样本\n\n")

        # Change labels
        f.write("CHANGE LABEL 分布:\n")
        f.write("-" * 30 + "\n")
        change_names = {
            1: 'Stable CN→CN', 2: 'Stable MCI→MCI', 3: 'Stable AD→AD',
            4: 'Conv CN→MCI', 5: 'Conv MCI→AD', 6: 'Conv CN→AD', 7: 'Rev MCI→CN'
        }

        total_change_samples = sum(change_label_counts.values())

        for label in range(1, 8):
            count = change_label_counts.get(label, 0)
            percentage = (count / total_change_samples) * 100 if total_change_samples > 0 else 0
            name = change_names.get(label, f'Unknown-{label}')
            f.write(f"{label} ({name}): {count} 样本 ({percentage:.1f}%)\n")

        f.write(f"总计: {total_change_samples} 样本\n\n")

        # Error files
        if error_files:
            f.write("读取失败的文件:\n")
            f.write("-" * 30 + "\n")
            for file_path in error_files:
                f.write(f"{file_path}\n")


def main():
    """
    主函数
    """
    data_path = r"Z:\yufengjiang\data\slice_nine"

    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"❌ 路径不存在: {data_path}")
        return

    print("开始分析数据集...")

    # 分析标签分布
    label_counts, change_label_counts, error_files = analyze_dataset_labels(data_path)

    # 创建可视化
    print("\n正在生成可视化图表...")
    create_visualization(label_counts, change_label_counts,
                         save_path="label_distribution.png")

    # 保存统计结果到文件
    print("正在保存统计结果...")
    save_statistics_to_file(label_counts, change_label_counts, error_files,
                            "label_statistics.txt")

    print("\n✓ 分析完成!")
    print(f"  - 图表保存为: label_distribution.png")
    print(f"  - 统计结果保存为: label_statistics.txt")


if __name__ == "__main__":
    main()