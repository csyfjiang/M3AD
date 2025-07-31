"""
简化版阿尔兹海默症NPZ数据加载器 - 支持prior向量
- 只保留裁剪和旋转增强
- 去除所有归一化（因为数据已经z-score标准化）
- 添加prior向量支持
- 打印数据范围
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from timm.data import Mixup
# 注意：如果没有安装timm，可以注释掉下面的导入
try:
    from timm.data.random_erasing import RandomErasing
    from timm.data.auto_augment import rand_augment_transform
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm not installed, some augmentations will be disabled")
    TIMM_AVAILABLE = False


def random_rot_flip(image):
    """随机旋转90度的倍数（只保留90度旋转，去除翻转）"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    # 注释掉翻转
    # axis = np.random.randint(0, 2)
    # image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    """随机小角度旋转"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    return image


class NPZToTensor(object):
    """将NPZ数据转换为Tensor - 不做任何归一化"""
    def __init__(self, print_stats=False):
        self.print_stats = print_stats
        self.first_print = True

    def __call__(self, image):
        # 如果是PIL图像，先转为numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 确保是float32
        image = image.astype(np.float32)

        # ===== 打印数据范围 =====
        if self.print_stats and self.first_print:
            print(f"\n[数据统计] Z-score标准化后的数据:")
            print(f"  - 范围: [{image.min():.4f}, {image.max():.4f}]")
            print(f"  - 均值: {image.mean():.4f}")
            print(f"  - 标准差: {image.std():.4f}")
            self.first_print = False

        # ===== 删除归一化代码 =====
        # 不再归一化到[0, 1]，保持z-score数据不变
        # if image.max() > 1.0:
        #     image = image / 255.0

        # 如果是2D图像，扩展为3通道
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)

        # 转换为CHW格式
        image = np.transpose(image, (2, 0, 1))

        # 转换为tensor
        image = torch.from_numpy(image).float()

        # ===== 删除标准化代码 =====
        # 不再做(x-mean)/std，保持z-score数据不变
        # if self.normalize:
        #     image = (image - self.mean) / self.std

        return image


class AlzheimerNPZDataset(Dataset):
    """阿尔兹海默症NPZ数据集 - 支持prior向量"""
    def __init__(self, data_dir, split='train', transform=None, file_list=None, debug=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.debug = debug

        # 获取文件列表
        if file_list:
            self.files = file_list
        else:
            self.files = glob.glob(os.path.join(data_dir, '*.npz'))
            self.files = [os.path.basename(f) for f in self.files]

        if len(self.files) == 0:
            raise ValueError(f"No NPZ files found in {data_dir}")

        print(f"Found {len(self.files)} NPZ files for {split} set")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载NPZ文件
        file_name = self.files[idx]
        if not file_name.startswith(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
        else:
            file_path = file_name

        try:
            data = np.load(file_path)

            # 获取2D切片数据
            slice_data = data['slice'].astype(np.float32)

            # 调试模式：打印原始数据信息
            if self.debug and idx == 0:
                print(f"\n[调试] 原始NPZ数据:")
                print(f"  - 文件: {file_name}")
                print(f"  - Slice shape: {slice_data.shape}")
                print(f"  - Slice范围: [{slice_data.min():.4f}, {slice_data.max():.4f}]")
                print(f"  - Slice均值: {slice_data.mean():.4f}, 标准差: {slice_data.std():.4f}")

            # 获取标签
            label = int(data['label'][()]) if data['label'].shape == () else int(data['label'])
            change_label = int(data['change_label'][()]) if data['change_label'].shape == () else int(data['change_label'])

            # ===== 新增：获取prior向量 =====
            prior = data['prior'].astype(np.float32)  # shape: (3,)

            # 调试模式：打印prior信息
            if self.debug and idx == 0:
                print(f"  - Prior shape: {prior.shape}")
                print(f"  - Prior values: {prior}")
                print(f"  - Prior sum: {prior.sum():.6f}")  # 检查是否为概率分布

            # 创建样本字典
            sample = {
                'slice': slice_data,
                'label': label,
                'change_label': change_label,
                'prior': prior,  # 新增prior字段
                'case_name': os.path.splitext(os.path.basename(file_name))[0]
            }

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


class AlzheimerTransform(object):
    """简化的transform类 - 只保留裁剪和旋转，支持prior向量"""
    def __init__(self, output_size, is_train=True, use_crop=True, use_rotation=True, print_stats=False):
        self.output_size = output_size
        self.is_train = is_train
        self.use_crop = use_crop
        self.use_rotation = use_rotation
        self.to_tensor = NPZToTensor(print_stats=print_stats)

    def __call__(self, sample):
        image = sample['slice']

        # ===== 1. 旋转增强（仅训练时）=====
        if self.is_train and self.use_rotation:
            image = random_rotate(image)  # 100%概率
            # if random.random() > 0.5:
            #     # 90度旋转
            #     image = random_rot_flip(image)
            # elif random.random() > 0.5:
            #     # 小角度旋转
            #     image = random_rotate(image)

        # ===== 2. 裁剪或调整大小 =====
        h, w = image.shape

        if self.is_train and self.use_crop:
            # 训练时：随机裁剪
            # 先放大一点（1.1-1.2倍），然后随机裁剪
            scale_factor = random.uniform(1.1, 1.2)
            new_h = int(self.output_size[0] * scale_factor)
            new_w = int(self.output_size[1] * scale_factor)

            # 缩放到稍大的尺寸
            if h != new_h or w != new_w:
                zoom_h = new_h / h
                zoom_w = new_w / w
                image = zoom(image, (zoom_h, zoom_w), order=3)

            # 随机裁剪到目标大小
            h, w = image.shape
            if h > self.output_size[0] and w > self.output_size[1]:
                top = random.randint(0, h - self.output_size[0])
                left = random.randint(0, w - self.output_size[1])
                image = image[top:top+self.output_size[0], left:left+self.output_size[1]]
            else:
                # 如果图像小于目标大小，直接缩放
                zoom_h = self.output_size[0] / h
                zoom_w = self.output_size[1] / w
                image = zoom(image, (zoom_h, zoom_w), order=3)
        else:
            # 验证时：直接缩放到目标大小
            if h != self.output_size[0] or w != self.output_size[1]:
                zoom_h = self.output_size[0] / h
                zoom_w = self.output_size[1] / w
                image = zoom(image, (zoom_h, zoom_w), order=3)

        # ===== 3. 删除所有其他增强 =====
        # 不使用：颜色增强、timm增强、随机擦除等

        # ===== 4. 转换为tensor（不做归一化）=====
        image = self.to_tensor(image)

        # ===== 5. 处理prior向量 =====
        prior = torch.from_numpy(sample['prior']).float()  # 转换为tensor

        # 返回最终格式
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'change_label': torch.tensor(sample['change_label'], dtype=torch.long),
            'prior': prior,  # 新增prior字段
            'case_name': sample['case_name']
        }


def build_dataset(is_train, config):
    """构建数据集"""

    # 配置数据增强
    transform = AlzheimerTransform(
        output_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        is_train=is_train,
        use_rotation=True,
        use_crop=True,
        print_stats=False
    )

    if config.DATA.DATASET == 'alzheimer':
        # 设置数据目录
        data_path = config.DATA.DATA_PATH

        # 检查是否有train/test子目录
        train_dir = os.path.join(data_path, 'train')
        test_dir = os.path.join(data_path, 'test')

        if os.path.exists(train_dir) and os.path.exists(test_dir):
            # 有子目录结构
            if is_train:
                data_dir = train_dir
            else:
                data_dir = test_dir

            dataset = AlzheimerNPZDataset(
                data_dir=data_dir,
                split='train' if is_train else 'val',
                transform=transform
            )
        else:
            # 没有子目录，需要手动分割
            all_files = glob.glob(os.path.join(data_path, '*.npz'))
            if not all_files:
                raise ValueError(f"No NPZ files found in {data_path}")

            # 打印找到的文件数量
            print(f"Found {len(all_files)} NPZ files in {data_path}")

            # 随机打乱并分割
            random.shuffle(all_files)
            split_idx = int(0.8 * len(all_files))

            if is_train:
                file_list = all_files[:split_idx]
            else:
                file_list = all_files[split_idx:]

            print(f"{'Train' if is_train else 'Val'} set: {len(file_list)} files")

            # 创建数据集并设置文件列表
            dataset = AlzheimerNPZDataset(
                data_dir=data_path,
                split='train' if is_train else 'val',
                transform=transform,
                file_list=file_list  # 传入文件列表
            )

        # 设置类别数
        nb_classes = {
            'diagnosis': 3,  # CN(1), MCI(2), Dementia(3)
            'change': 3  # Stable(1), Conversion(2), Reversion(3)
        }

    else:
        raise NotImplementedError(f"Dataset {config.DATA.DATASET} not supported")

    return dataset, nb_classes


def build_loader_finetune(config):
    """构建微调阶段的数据加载器"""
    config.defrost()
    dataset_train, _ = build_dataset(is_train=True, config=config)
    config.MODEL.NUM_CLASSES = 3  # 保持兼容性
    config.MODEL.NUM_CLASSES_DIAGNOSIS = config.MODEL.SWIN_ADMOE.NUM_CLASSES_DIAGNOSIS
    config.MODEL.NUM_CLASSES_CHANGE = config.MODEL.SWIN_ADMOE.NUM_CLASSES_CHANGE

    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config)

    if dist.is_available() and dist.is_initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        # 单GPU训练，使用普通采样器
        sampler_train = None
        sampler_val = None

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle=(sampler_train is None),  # 只有在没有sampler时才shuffle
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        shuffle=False,
    )

    # Mixup设置（对于医学图像可能需要谨慎使用）
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

# ===== 主测试代码 =====
if __name__ == "__main__":
    """测试数据加载器的完整功能"""

    # 测试配置
    class TestConfig:
        class DATA:
            IMG_SIZE = 256
            DATA_PATH = r"D:\codebase\Swin-Transformer\examples"
            BATCH_SIZE = 4
            NUM_WORKERS = 0
            PIN_MEMORY = True

    def visualize_augmentations(dataset, num_samples=4):
        """可视化数据增强效果"""
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))

        # 选择一个样本
        idx = 0

        # 原始图像（无增强）
        dataset.transform.is_train = False
        original_sample = dataset[idx]

        # 增强图像
        dataset.transform.is_train = True

        for i in range(num_samples):
            # 显示原始图像
            img_orig = original_sample['image'][0].numpy()
            axes[0, i].imshow(img_orig, cmap='gray')
            axes[0, i].set_title(f"Original\nRange:[{img_orig.min():.2f}, {img_orig.max():.2f}]\nPrior:{original_sample['prior'].numpy()}")
            axes[0, i].axis('off')

            # 显示增强后的图像
            aug_sample = dataset[idx]
            img_aug = aug_sample['image'][0].numpy()
            axes[1, i].imshow(img_aug, cmap='gray')
            axes[1, i].set_title(f"Augmented {i+1}\nRange:[{img_aug.min():.2f}, {img_aug.max():.2f}]\nPrior:{aug_sample['prior'].numpy()}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def test_dataloader(data_path=None):
        """测试数据加载器的主函数"""
        config = TestConfig()

        # 如果提供了路径，使用提供的路径
        if data_path:
            config.DATA.DATA_PATH = data_path

        print("="*60)
        print("测试阿尔兹海默症NPZ数据加载器（简化版 + Prior支持）")
        print("="*60)

        # 1. 测试基本数据加载
        print("\n1. 测试基本数据加载...")
        try:
            # 创建transform（不做增强）
            transform = AlzheimerTransform(
                output_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                is_train=False,
                use_crop=False,
                use_rotation=False,
                print_stats=True  # 打印数据统计
            )

            # 创建数据集
            dataset = AlzheimerNPZDataset(
                data_dir=config.DATA.DATA_PATH,
                split='test',
                transform=transform,
                debug=True  # 调试模式
            )

            # 加载第一个样本
            sample = dataset[0]
            print(f"\n✓ 成功加载数据")
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Image dtype: {sample['image'].dtype}")
            print(f"  - Label: {sample['label']}")
            print(f"  - Change label: {sample['change_label']}")
            print(f"  - Prior shape: {sample['prior'].shape}")
            print(f"  - Prior values: {sample['prior']}")
            print(f"  - Prior sum: {sample['prior'].sum():.6f}")
            print(f"  - Case name: {sample['case_name']}")

        except Exception as e:
            print(f"✗ 加载数据失败: {e}")
            return

        # 2. 测试数据增强
        print("\n2. 测试数据增强（只有裁剪和旋转）...")
        transform_aug = AlzheimerTransform(
            output_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            is_train=True,
            use_crop=True,      # 启用裁剪
            use_rotation=True,  # 启用旋转
            print_stats=False
        )

        dataset_aug = AlzheimerNPZDataset(
            data_dir=config.DATA.DATA_PATH,
            split='train',
            transform=transform_aug,
            debug=False
        )

        # 可视化增强效果
        print("  显示增强效果...")
        visualize_augmentations(dataset_aug, num_samples=4)

        # 3. 测试DataLoader
        print("\n3. 测试DataLoader...")
        dataloader = DataLoader(
            dataset_aug,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY
        )

        # 加载一个批次
        for i, batch in enumerate(dataloader):
            print(f"✓ 成功加载批次")
            print(f"  - Batch images shape: {batch['image'].shape}")
            print(f"  - Batch images dtype: {batch['image'].dtype}")
            print(f"  - Batch priors shape: {batch['prior'].shape}")
            print(f"  - Batch priors dtype: {batch['prior'].dtype}")

            # 打印批次数据范围
            batch_data = batch['image'].numpy()
            batch_priors = batch['prior'].numpy()
            print(f"  - Batch数据范围: [{batch_data.min():.4f}, {batch_data.max():.4f}]")
            print(f"  - Batch均值: {batch_data.mean():.4f}, 标准差: {batch_data.std():.4f}")
            print(f"  - Prior values sample:")
            for j in range(min(2, batch_priors.shape[0])):
                print(f"    Sample {j}: {batch_priors[j]} (sum: {batch_priors[j].sum():.6f})")

            # 显示批次中的图像
            actual_batch_size = batch['image'].shape[0]
            num_show = min(4, actual_batch_size)

            if num_show == 1:
                # 只有一个图像
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                img = batch['image'][0, 0].numpy()
                prior_vals = batch['prior'][0].numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Label: {batch['label'][0]}, Change: {batch['change_label'][0]}\n"
                           f"Range: [{img.min():.2f}, {img.max():.2f}]\n"
                           f"Prior: [{prior_vals[0]:.3f}, {prior_vals[1]:.3f}, {prior_vals[2]:.3f}]")
                ax.axis('off')
            else:
                # 多个图像
                fig, axes = plt.subplots(1, num_show, figsize=(num_show*3, 3))
                for j in range(num_show):
                    img = batch['image'][j, 0].numpy()
                    prior_vals = batch['prior'][j].numpy()
                    axes[j].imshow(img, cmap='gray')
                    axes[j].set_title(f"L:{batch['label'][j]}, C:{batch['change_label'][j]}\n"
                                    f"[{img.min():.1f}, {img.max():.1f}]\n"
                                    f"P:[{prior_vals[0]:.2f},{prior_vals[1]:.2f},{prior_vals[2]:.2f}]")
                    axes[j].axis('off')

            plt.suptitle("Batch Sample (Z-score Data + Prior)")
            plt.tight_layout()
            plt.show()

            break  # 只测试一个批次

        # 4. 统计信息
        print("\n4. 数据集统计信息...")
        all_labels = []
        all_change_labels = []
        all_priors = []
        all_mins = []
        all_maxs = []

        # 收集所有标签和数据范围
        total_samples = len(dataset)
        print(f"  - 总样本数: {total_samples}")

        for i in range(min(total_samples, 100)):  # 最多检查100个样本
            sample = dataset[i]
            all_labels.append(sample['label'].item())
            all_change_labels.append(sample['change_label'].item())
            all_priors.append(sample['prior'].numpy())

            # 记录数据范围
            img_data = sample['image'][0].numpy()
            all_mins.append(img_data.min())
            all_maxs.append(img_data.max())

        if all_labels:
            print(f"\n  - 标签统计:")
            print(f"    唯一标签值: {np.unique(all_labels)}")
            print(f"    标签分布: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
            print(f"    唯一变化标签值: {np.unique(all_change_labels)}")
            print(f"    变化标签分布: {dict(zip(*np.unique(all_change_labels, return_counts=True)))}")

            print(f"\n  - 数据范围统计:")
            print(f"    最小值范围: [{np.min(all_mins):.4f}, {np.max(all_mins):.4f}]")
            print(f"    最大值范围: [{np.min(all_maxs):.4f}, {np.max(all_maxs):.4f}]")

            print(f"\n  - Prior统计:")
            all_priors = np.array(all_priors)
            print(f"    Prior shape: {all_priors.shape}")
            print(f"    Prior均值: {all_priors.mean(axis=0)}")
            print(f"    Prior标准差: {all_priors.std(axis=0)}")
            print(f"    Prior范围: [{all_priors.min(axis=0)}, {all_priors.max(axis=0)}]")

        print("\n测试完成！")

    # 运行测试
    data_path = r"D:\codebase\Swin-Transformer\examples"

    # 如果路径不存在，尝试当前目录
    if not os.path.exists(data_path):
        print(f"路径 {data_path} 不存在，尝试使用当前目录...")
        data_path = "."

    test_dataloader(data_path)