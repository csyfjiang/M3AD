import logging
import os
import random
import sys
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import math

import logging
from datetime import datetime  # 添加这行

# 假设您的数据加载器在这里
from data.data_ad_loader import build_loader_finetune


def load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=None):
    """
    加载预训练权重，可选择排除decoder

    Args:
        model: 要加载权重的模型
        pretrained_path: 预训练权重路径
        exclude_decoder: 是否排除decoder权重
        logger: 日志记录器

    Returns:
        dict: 包含加载信息的字典
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"Loading pretrained weights from: {pretrained_path}")

    # 加载checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # 获取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch_info = checkpoint.get('epoch', 'unknown')
        phase_info = checkpoint.get('phase', 'unknown')
        logger.info(f"Loading from epoch {epoch_info}, phase: {phase_info}")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        logger.info("Loading from 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        logger.info("Loading state_dict directly")

    # 统计原始权重
    total_keys = len(state_dict)
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]

    logger.info(f"Original checkpoint contains {total_keys} parameters")
    logger.info(f"Found {len(decoder_keys)} decoder parameters")

    if exclude_decoder and decoder_keys:
        # 过滤掉decoder相关权重
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        logger.info(f"Excluded decoder parameters:")
        for key in decoder_keys:
            logger.info(f"  - {key}: {state_dict[key].shape}")

        state_dict = filtered_state_dict
        logger.info(f"Filtered state_dict contains {len(state_dict)} parameters")

    # 获取模型当前的参数
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # 分析匹配情况
    matched_keys = model_keys & checkpoint_keys
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    logger.info(f"Parameter matching analysis:")
    logger.info(f"  - Matched parameters: {len(matched_keys)}")
    logger.info(f"  - Missing parameters: {len(missing_keys)}")
    logger.info(f"  - Unexpected parameters: {len(unexpected_keys)}")

    if missing_keys:
        logger.info("Missing parameters (will be randomly initialized):")
        for key in sorted(missing_keys):
            logger.info(f"  - {key}")

    if unexpected_keys:
        logger.info("Unexpected parameters (will be ignored):")
        for key in sorted(unexpected_keys):
            logger.info(f"  - {key}")

    # 加载权重，允许部分匹配
    load_result = model.load_state_dict(state_dict, strict=False)

    logger.info(f"Weight loading completed!")
    logger.info(f"  - Successfully loaded: {len(matched_keys)} parameters")
    logger.info(f"  - Randomly initialized: {len(missing_keys)} parameters")

    return {
        'total_keys': total_keys,
        'loaded_keys': len(matched_keys),
        'missing_keys': len(missing_keys),
        'unexpected_keys': len(unexpected_keys),
        'excluded_decoder': exclude_decoder and len(decoder_keys) > 0,
        'decoder_keys_count': len(decoder_keys)
    }


def remove_decoder_from_model(model, logger=None):
    """
    从模型中移除decoder组件并记录详细信息

    Args:
        model: 要修改的模型
        logger: 日志记录器

    Returns:
        dict: 包含移除信息的字典
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info("REMOVING DECODER FROM MODEL")
    logger.info("=" * 60)

    # 获取移除前的参数统计
    total_params_before = sum(p.numel() for p in model.parameters())
    decoder_params = 0
    decoder_components = []

    # 检查是否有decoder
    model_to_check = model.module if hasattr(model, 'module') else model

    if hasattr(model_to_check, 'decoder') and model_to_check.decoder is not None:
        # 统计decoder参数
        for name, param in model_to_check.decoder.named_parameters():
            decoder_params += param.numel()
            decoder_components.append((name, param.shape, param.numel()))

        logger.info(f"Decoder components to be removed:")
        for name, shape, numel in decoder_components:
            logger.info(f"  - {name}: {shape} ({numel:,} parameters)")

        logger.info(f"Total decoder parameters: {decoder_params:,}")

        # 移除decoder
        del model_to_check.decoder
        model_to_check.decoder = None

        logger.info("✓ Decoder successfully removed from model")

        # 如果是DataParallel，需要重新包装
        if hasattr(model, 'module'):
            logger.info("Re-wrapping model with DataParallel...")
            device_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model_to_check, device_ids=device_ids)
            logger.info(f"✓ Model re-wrapped with DataParallel on devices: {device_ids}")
    else:
        logger.info("No decoder found in model or decoder already None")

    # 获取移除后的参数统计
    total_params_after = sum(p.numel() for p in model.parameters())
    memory_saved = decoder_params * 4 / (1024**2)  # 假设float32，转换为MB

    logger.info(f"Parameter statistics:")
    logger.info(f"  - Before: {total_params_before:,} parameters")
    logger.info(f"  - After: {total_params_after:,} parameters")
    logger.info(f"  - Removed: {decoder_params:,} parameters")
    logger.info(f"  - Memory saved: ~{memory_saved:.2f} MB")

    logger.info("=" * 60)

    return {
        'removed': decoder_params > 0,
        'decoder_params': decoder_params,
        'params_before': total_params_before,
        'params_after': total_params_after,
        'memory_saved_mb': memory_saved,
        'components': decoder_components
    }


def log_model_components(model, phase="unknown", logger=None):
    """
    记录模型组件的详细信息

    Args:
        model: 要分析的模型
        phase: 当前阶段名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'='*50}")
    logger.info(f"MODEL COMPONENTS ANALYSIS - {phase.upper()}")
    logger.info(f"{'='*50}")

    model_to_check = model.module if hasattr(model, 'module') else model

    # 统计各个组件的参数量
    components = {}

    for name, module in model_to_check.named_children():
        if module is not None:
            param_count = sum(p.numel() for p in module.parameters())
            components[name] = param_count

            # 特别标注重要组件
            if name in ['decoder', 'head_diagnosis', 'head_change', 'clinical_encoder', 'clinical_fusion']:
                status = "✓ Active" if param_count > 0 else "✗ None/Empty"
                logger.info(f"  {name:20}: {param_count:>10,} params {status}")
            else:
                logger.info(f"  {name:20}: {param_count:>10,} params")

    total_params = sum(components.values())
    logger.info(f"  {'='*40}")
    logger.info(f"  {'Total':20}: {total_params:>10,} params")

    # 检查特定组件状态
    special_components = ['decoder', 'head_diagnosis', 'head_change']
    logger.info(f"\nSpecial component status:")
    for comp in special_components:
        if hasattr(model_to_check, comp):
            attr = getattr(model_to_check, comp)
            if attr is None:
                logger.info(f"  - {comp}: None (removed/disabled)")
            else:
                param_count = sum(p.numel() for p in attr.parameters())
                logger.info(f"  - {comp}: Active ({param_count:,} params)")
        else:
            logger.info(f"  - {comp}: Not found")

    logger.info(f"{'='*50}")

    return components


def generate_mask(input_size: Tuple[int, int], patch_size: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """生成SimMIM的随机mask - patch级别的mask"""
    H, W = input_size
    # 计算patch的数量
    h, w = H // patch_size, W // patch_size
    num_patches = h * w
    num_mask = int(num_patches * mask_ratio)

    # 随机选择要mask的patches
    mask = torch.zeros(num_patches, dtype=torch.float32, device=device)
    mask_indices = torch.randperm(num_patches, device=device)[:num_mask]
    mask[mask_indices] = 1

    return mask


def norm_targets(targets, patch_size):
    """标准化目标 - from SimMIM"""
    assert patch_size % 2 == 1

    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.

    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)

    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

    return targets_


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''保存模型当验证损失下降时'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 保存时去除decoder（如果存在）
        model_to_save = model.module if hasattr(model, 'module') else model
        state_dict = model_to_save.state_dict()

        # 过滤decoder权重
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        torch.save(filtered_state_dict, path)
        self.val_loss_min = val_loss


class SimMIMLoss(nn.Module):
    """SimMIM重建损失"""

    def __init__(self, patch_size=4, norm_target=True, norm_target_patch_size=47):
        super().__init__()
        self.patch_size = patch_size
        self.norm_target = norm_target
        self.norm_target_patch_size = norm_target_patch_size

    def forward(self, input_images, reconstructed, mask):
        """
        Args:
            input_images: 原始输入图像 [B, C, H, W]
            reconstructed: 重建图像 [B, C, H, W]
            mask: patch级别的mask [B, num_patches]
        """
        B, C, H, W = input_images.shape

        # 将patch级别的mask转换为像素级别
        h, w = H // self.patch_size, W // self.patch_size
        mask_reshaped = mask.reshape(B, h, w)

        # 扩展mask到像素级别
        mask_upsampled = mask_reshaped.unsqueeze(-1).unsqueeze(-1)
        mask_upsampled = mask_upsampled.repeat(1, 1, 1, self.patch_size, self.patch_size)
        mask_upsampled = mask_upsampled.permute(0, 1, 3, 2, 4).contiguous()
        mask_upsampled = mask_upsampled.view(B, H, W)
        mask_upsampled = mask_upsampled.unsqueeze(1).repeat(1, C, 1, 1)

        # 标准化目标（如果启用）
        targets = input_images
        if self.norm_target:
            targets = norm_targets(targets, self.norm_target_patch_size)

        # 计算重建损失（仅在masked区域）
        loss_recon = F.l1_loss(targets, reconstructed, reduction='none')
        loss = (loss_recon * mask_upsampled).sum() / (mask_upsampled.sum() + 1e-5) / C

        return loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数 - MMoE版本"""

    def __init__(self, weight_diagnosis=1.0, weight_change=1.0, label_smoothing=0.0):
        super().__init__()
        self.criterion_diagnosis = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_change = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weight_diagnosis = weight_diagnosis
        self.weight_change = weight_change

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                diagnosis_labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算两个任务的损失
        注意：标签是1,2,3，需要转换为0,1,2
        Returns:
            包含total_loss, diagnosis_loss, change_loss的字典
        """
        output_diagnosis, output_change = outputs

        # 将标签从1,2,3转换为0,1,2
        diagnosis_labels_zero_indexed = diagnosis_labels - 1
        change_labels_zero_indexed = change_labels - 1

        loss_diagnosis = self.criterion_diagnosis(output_diagnosis, diagnosis_labels_zero_indexed)
        loss_change = self.criterion_change(output_change, change_labels_zero_indexed)

        total_loss = self.weight_diagnosis * loss_diagnosis + self.weight_change * loss_change

        return {
            'total': total_loss,
            'diagnosis': loss_diagnosis,
            'change': loss_change
        }


def compute_metrics(outputs: Tuple[torch.Tensor, torch.Tensor],
                    diagnosis_labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, float]:
    """计算评估指标 - MMoE版本"""
    output_diagnosis, output_change = outputs

    # 获取预测结果（预测的是0,1,2，需要转回1,2,3）
    pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1
    pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

    diagnosis_labels_np = diagnosis_labels.cpu().numpy()
    change_labels_np = change_labels.cpu().numpy()

    # 计算准确率
    acc_diagnosis = accuracy_score(diagnosis_labels_np, pred_diagnosis)
    acc_change = accuracy_score(change_labels_np, pred_change)

    # 计算F1分数（使用1,2,3作为标签）
    f1_diagnosis = f1_score(diagnosis_labels_np, pred_diagnosis, labels=[1, 2, 3], average='weighted', zero_division=0)
    f1_change = f1_score(change_labels_np, pred_change, labels=[1, 2, 3], average='weighted', zero_division=0)

    return {
        'acc_diagnosis': acc_diagnosis,
        'acc_change': acc_change,
        'f1_diagnosis': f1_diagnosis,
        'f1_change': f1_change,
        'acc_avg': (acc_diagnosis + acc_change) / 2,
        'f1_avg': (f1_diagnosis + f1_change) / 2
    }


def log_expert_utilization(model, val_loader, device, epoch):
    """记录专家利用率 - 用于分析MMoE的工作情况（支持clinical prior）"""
    model.eval()
    expert_weights_list = []

    with torch.no_grad():
        # 只取一个batch进行分析
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)  # 新增：获取clinical prior

        # 获取专家利用率
        try:
            expert_utilization = model.get_expert_utilization(
                images,
                clinical_prior=clinical_priors,  # 新增：传入clinical prior
                lbls_diagnosis=diagnosis_labels,
                lbls_change=change_labels
            )

            if expert_utilization:
                # 计算平均专家权重
                for layer_idx, gate_weights in enumerate(expert_utilization):
                    if 'diagnosis_weights' in gate_weights:
                        diagnosis_weights = gate_weights['diagnosis_weights'].mean(dim=[0, 1])  # [num_experts]
                        change_weights = gate_weights['change_weights'].mean(dim=[0, 1])  # [num_experts]

                        # 记录到wandb
                        expert_names = ['Shared', 'AD-focused', 'MCI-focused', 'CN-focused']
                        for i, name in enumerate(expert_names):
                            wandb.log({
                                f'expert_utilization/layer_{layer_idx}_diagnosis_{name}': diagnosis_weights[i].item(),
                                f'expert_utilization/layer_{layer_idx}_change_{name}': change_weights[i].item(),
                                'epoch': epoch
                            })
        except Exception as e:
            logging.warning(f"Failed to log expert utilization: {e}")


def train_one_epoch_pretrain(model, train_loader, criterion_simmim, optimizer, scheduler, device, epoch,
                            mask_ratio=0.6, patch_size=4, img_size=256, position=0):
    """预训练一个epoch - SimMIM重建任务"""
    model.train()

    # 确保模型处于预训练模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = True
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(True)

    total_loss = 0
    num_patches = (img_size // patch_size) ** 2

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Pretrain Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        batch_size = images.shape[0]

        # 生成mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # 前向传播 - SimMIM重建
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,  # 转换为0,1,2
            lbls_change=change_labels - 1,
            mask=masks
        )

        # 计算重建损失
        loss = criterion_simmim(images, reconstructed, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录损失
        total_loss += loss.item()

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}",
            'lr': f"{current_lr:.2e}"
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_one_epoch_finetune(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                            use_timm_scheduler=False, position=0):
    """微调一个epoch - 分类任务"""
    model.train()

    # 确保模型处于微调模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = False
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(False)

    total_loss = 0
    total_diagnosis_loss = 0
    total_change_loss = 0
    all_metrics = []

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Finetune Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # 前向传播 - 分类任务
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,
            lbls_change=change_labels - 1
        )

        # 计算损失
        losses = criterion(outputs, diagnosis_labels, change_labels)
        loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not use_timm_scheduler:
            scheduler.step()

        # 记录损失
        total_loss += loss.item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # 计算指标
        with torch.no_grad():
            metrics = compute_metrics(outputs, diagnosis_labels, change_labels)
            all_metrics.append(metrics)

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc_diag': f"{metrics['acc_diagnosis']:.4f}",
            'acc_chg': f"{metrics['acc_change']:.4f}",
            'acc_avg': f"{metrics['acc_avg']:.4f}",
            'lr': f"{current_lr:.2e}"
        })

    if use_timm_scheduler:
        scheduler.step(epoch)

    # 计算平均值
    avg_loss = total_loss / len(train_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(train_loader)
    avg_change_loss = total_change_loss / len(train_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics


@torch.no_grad()
def validate_pretrain(model, val_loader, criterion_simmim, device, epoch,
                     mask_ratio=0.6, patch_size=4, img_size=256, position=2):
    """预训练验证 - SimMIM重建任务"""
    model.eval()

    # 确保模型处于预训练模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = True
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(True)

    total_loss = 0

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Pretrain Val Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        batch_size = images.shape[0]

        # 生成mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # 前向传播
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,
            lbls_change=change_labels - 1,
            mask=masks
        )

        # 计算损失
        loss = criterion_simmim(images, reconstructed, masks)
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}"
        })

    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def validate_finetune(model, val_loader, criterion, device, epoch, position=2):
    """微调验证 - 分类任务"""
    model.eval()

    # 确保模型处于微调模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = False
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(False)

    total_loss = 0
    total_diagnosis_loss = 0
    total_change_loss = 0
    all_metrics = []

    # For confusion matrix and classification report
    all_pred_diagnosis = []
    all_true_diagnosis = []
    all_pred_change = []
    all_true_change = []

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Finetune Val Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # 前向传播
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )

        # 计算损失
        losses = criterion(outputs, diagnosis_labels, change_labels)

        # 记录损失
        total_loss += losses['total'].item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # 计算指标
        metrics = compute_metrics(outputs, diagnosis_labels, change_labels)
        all_metrics.append(metrics)

        # 收集预测结果
        output_diagnosis, output_change = outputs
        pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1
        pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

        all_pred_diagnosis.extend(pred_diagnosis)
        all_true_diagnosis.extend(diagnosis_labels.cpu().numpy())
        all_pred_change.extend(pred_change)
        all_true_change.extend(change_labels.cpu().numpy())

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'acc': f"{metrics['acc_avg']:.4f}"
        })

    # 计算平均值
    avg_loss = total_loss / len(val_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(val_loader)
    avg_change_loss = total_change_loss / len(val_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # 计算混淆矩阵
    cm_diagnosis = confusion_matrix(all_true_diagnosis, all_pred_diagnosis, labels=[1, 2, 3])
    cm_change = confusion_matrix(all_true_change, all_pred_change, labels=[1, 2, 3])

    return (avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics, cm_diagnosis, cm_change,
            all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change)


def trainer_alzheimer_mmoe(args, model, snapshot_path):
    """Alzheimer's disease dual-task trainer with MMoE main function - 支持SimMIM预训练 + 智能权重管理"""
    model_name = getattr(args, 'model_name', 'swin_admoe')
    if hasattr(args, 'MODEL') and hasattr(args.MODEL, 'NAME'):
        model_name = args.MODEL.NAME

    # 添加时间戳和模型名称到快照路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = os.path.join(
        os.path.dirname(snapshot_path),
        f"{model_name}_{timestamp}"
    )
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "training.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Initialize wandb
    wandb_name = getattr(args, 'wandb_run_name', getattr(args, 'exp_name', 'alzheimer_mmoe_run'))
    wandb_dir = os.path.join(os.path.dirname(snapshot_path), 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=getattr(args, 'wandb_project', 'alzheimer-mmoe-classification'),
        name=wandb_name,
        config=vars(args) if hasattr(args, '__dict__') else args,
        dir=wandb_dir,
        mode='offline' if getattr(args, 'wandb_offline', False) else 'online'
    )

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ============== 权重加载逻辑 ==============
    start_epoch = getattr(args, 'start_epoch', 0)
    resume_path = getattr(args, 'resume', None)
    pretrained_path = getattr(args, 'pretrained', None)

    if resume_path:
        # 从中断点恢复训练
        logging.info("Resuming training from checkpoint...")
        load_info = load_pretrained_weights(model, resume_path, exclude_decoder=False, logger=logging.getLogger())

        # 从checkpoint中获取epoch信息
        checkpoint = torch.load(resume_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0) + 1
        logging.info(f"Resumed from epoch {start_epoch}")

    elif pretrained_path:
        # 加载预训练权重（排除decoder）
        logging.info("Loading pretrained weights (excluding decoder)...")
        load_info = load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=logging.getLogger())

        # 记录权重加载信息到wandb
        wandb.log({
            'weight_loading/total_keys': load_info['total_keys'],
            'weight_loading/loaded_keys': load_info['loaded_keys'],
            'weight_loading/missing_keys': load_info['missing_keys'],
            'weight_loading/excluded_decoder': load_info['excluded_decoder']
        })

    # 记录初始模型组件
    log_model_components(model, "Initial", logging.getLogger())

    # Data loader
    logging.info("Loading data...")
    if hasattr(args, 'config'):
        config = args.config
    elif hasattr(args, 'DATA'):
        config = args
    else:
        config = type('Config', (), {})()
        config.DATA = args.DATA if hasattr(args, 'DATA') else args

    # Build data loaders
    dataset_train, dataset_val, train_loader, val_loader, mixup_fn = build_loader_finetune(config)
    logging.info(f"Train set size: {len(dataset_train)}, Val set size: {len(dataset_val)}")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs")

    # 损失函数
    # SimMIM损失（预训练）
    criterion_simmim = SimMIMLoss(
        patch_size=getattr(args, 'patch_size', 4),
        norm_target=getattr(args, 'norm_target', True),
        norm_target_patch_size=getattr(args, 'norm_target_patch_size', 47)
    )

    # 分类损失（微调）
    criterion_classification = MultiTaskLoss(
        weight_diagnosis=getattr(args, 'weight_diagnosis', 1.0),
        weight_change=getattr(args, 'weight_change', 1.0),
        label_smoothing=getattr(args, 'label_smoothing', 0.0)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    # 如果是resume，加载optimizer状态
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("Optimizer state loaded from checkpoint")

    # Scheduler
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    num_steps_per_epoch = len(train_loader)
    total_steps = num_steps_per_epoch * args.max_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 如果是resume，加载scheduler状态
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Scheduler state loaded from checkpoint")

    logging.info(f"Using cosine scheduler with warmup")
    logging.info(f"Warmup epochs: {warmup_epochs}, Base LR: {args.base_lr}")

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training parameters
    best_val_acc = 0
    pretrain_epochs = getattr(args, 'pretrain_epochs', args.max_epochs // 2)

    # SimMIM parameters
    mask_ratio = getattr(args, 'mask_ratio', 0.6)
    patch_size = getattr(args, 'patch_size', 4)
    img_size = getattr(args, 'img_size', 256)

    logging.info(f"Training plan:")
    logging.info(f"- Pretraining epochs: {pretrain_epochs} (SimMIM reconstruction)")
    logging.info(f"- Finetuning epochs: {args.max_epochs - pretrain_epochs} (Classification)")
    logging.info(f"- SimMIM mask ratio: {mask_ratio}")
    logging.info(f"- Starting from epoch: {start_epoch}")

    overall_pbar = tqdm(
        total=args.max_epochs - start_epoch,
        desc="Overall Training Progress",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}, {rate_fmt}]'
    )

    # 标记是否已经进行了decoder移除
    decoder_removed = False

    for epoch in range(start_epoch, args.max_epochs):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Epoch {epoch}/{args.max_epochs - 1}")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")

        is_pretrain = epoch < pretrain_epochs
        phase = "Pretrain (SimMIM)" if is_pretrain else "Finetune (Classification)"
        logging.info(f"Phase: {phase}")

        # ============== 阶段切换逻辑 ==============
        if not is_pretrain and not decoder_removed:
            # 从预训练切换到微调：移除decoder
            logging.info("\n" + "="*80)
            logging.info("SWITCHING FROM PRETRAINING TO FINETUNING")
            logging.info("="*80)

            # 移除decoder
            remove_info = remove_decoder_from_model(model, logging.getLogger())
            decoder_removed = True

            # 记录移除信息到wandb
            wandb.log({
                'model_modification/decoder_removed': remove_info['removed'],
                'model_modification/decoder_params_removed': remove_info['decoder_params'],
                'model_modification/memory_saved_mb': remove_info['memory_saved_mb'],
                'model_modification/params_after_removal': remove_info['params_after'],
                'epoch': epoch
            })

            # 重新创建optimizer（因为参数可能发生变化）
            logging.info("Recreating optimizer for remaining parameters...")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )

            # 重新创建scheduler
            remaining_steps = num_steps_per_epoch * (args.max_epochs - epoch)
            remaining_warmup = max(0, warmup_steps - epoch * num_steps_per_epoch)

            def new_lr_lambda(current_step):
                if current_step < remaining_warmup:
                    return float(current_step) / float(max(1, remaining_warmup))
                progress = float(current_step - remaining_warmup) / float(max(1, remaining_steps - remaining_warmup))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = LambdaLR(optimizer, new_lr_lambda)

            logging.info("✓ Optimizer and scheduler recreated for finetuning phase")

            # 记录切换后的模型组件
            log_model_components(model, "After Decoder Removal", logging.getLogger())

        if is_pretrain:
            # ===== 预训练阶段：SimMIM重建 =====
            train_loss = train_one_epoch_pretrain(
                model, train_loader, criterion_simmim, optimizer, scheduler, device, epoch,
                mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size, position=1
            )

            # Log training metrics
            wandb.log({
                'train/loss_simmim': train_loss,
                'train/lr': current_lr,
                'train/phase': 1,  # 1 for pretrain
                'train/mask_ratio': mask_ratio,
                'epoch': epoch
            })

            logging.info(f"Pretrain - SimMIM Loss: {train_loss:.4f}")

            # Validation (every eval_interval epochs)
            if (epoch + 1) % args.eval_interval == 0:
                val_loss = validate_pretrain(
                    model, val_loader, criterion_simmim, device, epoch,
                    mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size
                )

                wandb.log({
                    'val/loss_simmim': val_loss,
                    'val/phase': 1,  # 1 for pretrain
                    'epoch': epoch
                })

                logging.info(f"Pretrain Val - SimMIM Loss: {val_loss:.4f}")

                # Early stopping based on reconstruction loss
                early_stopping(val_loss, model, os.path.join(snapshot_path, 'pretrain_checkpoint.pth'))
                if early_stopping.early_stop:
                    logging.info("Early stopping triggered during pretraining!")
                    break

        else:
            # ===== 微调阶段：分类任务 =====
            train_loss, train_diagnosis_loss, train_change_loss, train_metrics = train_one_epoch_finetune(
                model, train_loader, criterion_classification, optimizer, scheduler, device, epoch,
                use_timm_scheduler=False, position=1
            )

            # Log training metrics
            wandb.log({
                'train/loss': train_loss,
                'train/loss_diagnosis': train_diagnosis_loss,
                'train/loss_change': train_change_loss,
                'train/acc_diagnosis': train_metrics['acc_diagnosis'],
                'train/acc_change': train_metrics['acc_change'],
                'train/acc_avg': train_metrics['acc_avg'],
                'train/f1_diagnosis': train_metrics['f1_diagnosis'],
                'train/f1_change': train_metrics['f1_change'],
                'train/f1_avg': train_metrics['f1_avg'],
                'train/lr': current_lr,
                'train/phase': 0,  # 0 for finetune
                'epoch': epoch
            })

            logging.info(f"Finetune - Loss: {train_loss:.4f}, Acc: {train_metrics['acc_avg']:.4f}")

            # Validation (every eval_interval epochs)
            if (epoch + 1) % args.eval_interval == 0:
                val_results = validate_finetune(model, val_loader, criterion_classification, device, epoch)

                if len(val_results) == 6:
                    val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change = val_results
                    all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = None, None, None, None
                else:
                    val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change, \
                        all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = val_results

                # Log validation metrics
                wandb.log({
                    'val/loss': val_loss,
                    'val/loss_diagnosis': val_diagnosis_loss,
                    'val/loss_change': val_change_loss,
                    'val/acc_diagnosis': val_metrics['acc_diagnosis'],
                    'val/acc_change': val_metrics['acc_change'],
                    'val/acc_avg': val_metrics['acc_avg'],
                    'val/f1_diagnosis': val_metrics['f1_diagnosis'],
                    'val/f1_change': val_metrics['f1_change'],
                    'val/f1_avg': val_metrics['f1_avg'],
                    'val/phase': 0,  # 0 for finetune
                    'epoch': epoch
                })

                # Log expert utilization (only during finetuning)
                log_expert_utilization(model, val_loader, device, epoch)

                # Log confusion matrices
                diagnosis_names = ['CN', 'MCI', 'Dementia']
                change_names = ['Stable', 'Conversion', 'Reversion']

                fig_cm_diagnosis = plt.figure(figsize=(8, 6))
                plt.imshow(cm_diagnosis, interpolation='nearest', cmap='Blues')
                plt.title('Confusion Matrix - Diagnosis')
                plt.colorbar()
                tick_marks = np.arange(3)
                plt.xticks(tick_marks, diagnosis_names, rotation=45)
                plt.yticks(tick_marks, diagnosis_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')

                for i in range(3):
                    for j in range(3):
                        plt.text(j, i, str(cm_diagnosis[i, j]),
                                ha="center", va="center", color="black")
                plt.tight_layout()

                fig_cm_change = plt.figure(figsize=(8, 6))
                plt.imshow(cm_change, interpolation='nearest', cmap='Blues')
                plt.title('Confusion Matrix - Change Label')
                plt.colorbar()
                plt.xticks(tick_marks, change_names, rotation=45)
                plt.yticks(tick_marks, change_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')

                for i in range(3):
                    for j in range(3):
                        plt.text(j, i, str(cm_change[i, j]),
                                ha="center", va="center", color="black")
                plt.tight_layout()

                wandb.log({
                    'val/confusion_matrix_diagnosis': wandb.Image(fig_cm_diagnosis),
                    'val/confusion_matrix_change': wandb.Image(fig_cm_change),
                    'epoch': epoch
                })

                plt.close(fig_cm_diagnosis)
                plt.close(fig_cm_change)

                logging.info(f"Finetune Val - Loss: {val_loss:.4f}, Acc: {val_metrics['acc_avg']:.4f}")

                # Save best model (only during finetuning)
                if val_metrics['acc_avg'] > best_val_acc:
                    best_val_acc = val_metrics['acc_avg']
                    best_model_path = os.path.join(snapshot_path, 'best_model.pth')

                    model_to_save = model.module if hasattr(model, 'module') else model

                    # 保存时过滤decoder权重
                    state_dict = model_to_save.state_dict()
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('decoder.')
                    }

                    torch.save(filtered_state_dict, best_model_path)
                    logging.info(f"Best model saved with acc: {best_val_acc:.4f}")

                    wandb.run.summary['best_val_acc'] = best_val_acc
                    wandb.run.summary['best_epoch'] = epoch

                # Early stopping based on classification accuracy
                early_stopping(-val_metrics['acc_avg'], model, os.path.join(snapshot_path, 'finetune_checkpoint.pth'))
                if early_stopping.early_stop:
                    logging.info("Early stopping triggered during finetuning!")
                    break

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(snapshot_path, f'checkpoint_epoch_{epoch}.pth')
            model_to_save = model.module if hasattr(model, 'module') else model

            # 保存时过滤decoder权重
            state_dict = model_to_save.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('decoder.')
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': filtered_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'is_pretrain': is_pretrain,
                'phase': phase,
                'decoder_removed': decoder_removed
            }, checkpoint_path)

            logging.info(f"Checkpoint saved at epoch {epoch}")

        overall_pbar.update(1)

    # Save final model
    final_model_path = os.path.join(snapshot_path, 'final_model.pth')
    model_to_save = model.module if hasattr(model, 'module') else model

    # 保存时过滤decoder权重
    state_dict = model_to_save.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('decoder.')
    }

    torch.save(filtered_state_dict, final_model_path)

    # 记录最终模型组件
    log_model_components(model, "Final", logging.getLogger())

    logging.info(f"\nTraining completed!")
    if best_val_acc > 0:
        logging.info(f"Best validation accuracy: {best_val_acc:.4f}")

    wandb.finish()
    overall_pbar.close()

    return "Training Finished!"


# 使用示例的参数类 - 添加SimMIM参数
class Args:
    def __init__(self):
        # 基础参数
        self.seed = 42
        self.max_epochs = 100
        self.eval_interval = 10
        self.save_interval = 20
        self.patience = 10  # 增加耐心值，因为预训练可能需要更长时间

        # 优化器参数
        self.base_lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4

        # 损失函数参数
        self.weight_diagnosis = 1.0
        self.weight_change = 1.0
        self.label_smoothing = 0.1

        # 训练阶段参数
        self.pretrain_epochs = 50  # 预训练轮数

        # SimMIM参数
        self.mask_ratio = 0.6  # mask比例
        self.patch_size = 4    # patch大小
        self.img_size = 256    # 图像大小
        self.norm_target = True  # 是否标准化目标
        self.norm_target_patch_size = 47  # 标准化patch大小

        # wandb参数
        self.wandb_project = "alzheimer-mmoe-simmim"
        self.exp_name = "dual-task-mmoe-simmim"

        # 权重加载参数
        self.resume = None  # 恢复训练的checkpoint路径
        self.pretrained = None  # 预训练权重路径

        # 数据参数
        self.DATA = type('obj', (object,), {
            'DATASET': 'alzheimer',
            'DATA_PATH': 'path/to/data',
            'IMG_SIZE': 256,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'PIN_MEMORY': True
        })

        self.num_classes = 3


if __name__ == "__main__":
    # 示例用法
    args = Args()
    # 设置预训练权重路径（如果有的话）
    # args.pretrained = "path/to/pretrained_weights.pth"
    # args.resume = "path/to/checkpoint.pth"  # 如果要恢复训练

    # model = SwinTransformerV2_AlzheimerMMoE(...)
    # snapshot_path = "./checkpoints/mmoe_simmim_exp1"
    # trainer_alzheimer_mmoe(args, model, snapshot_path)