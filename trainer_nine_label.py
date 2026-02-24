"""
Alzheimer's Disease Dual-Task Classification Trainer - MMoE + SimMIM Pretraining - 7-Class Change Label
With 5-Fold Cross-Validation and Majority Voting for Validation Predictions.

- Pretrain Phase: SimMIM self-supervised reconstruction
- Finetune Phase: Dual classification tasks
  - Diagnosis (1=CN, 2=MCI, 3=Dementia)
  - Change Label (1-7: detailed transition types)
    1: Stable CN to CN
    2: Stable MCI to MCI  
    3: Stable AD to AD
    4: Conversion CN to MCI
    5: Conversion MCI to AD
    6: Conversion CN to AD
    7: Reversion MCI to CN
- 8-expert MMoE architecture with separate gating networks
- wandb logging
- Early stopping
- Smart weight management: auto-remove decoder after pretraining
- 5-Fold Cross-Validation with majority voting on accumulated predictions
"""
import logging
import os
import random
import sys
import copy
from typing import Tuple, Dict
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import math

import logging
from datetime import datetime

# Data loader
from data.data_ad_loader import build_loader_finetune


def validate_trainer_args(args):
    """
    Validate trainer arguments to catch missing or invalid parameters early
    """
    required_attrs = [
        'seed', 'max_epochs', 'eval_interval', 'save_interval', 'patience',
        'base_lr', 'weight_decay'
    ]

    # Check required attributes
    for attr in required_attrs:
        if not hasattr(args, attr):
            raise ValueError(f"Missing required argument: {attr}")

    # Validate parameter ranges
    if args.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    if args.base_lr <= 0:
        raise ValueError("base_lr must be positive")

    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    # Set default values for optional parameters
    if not hasattr(args, 'pretrain_epochs'):
        args.pretrain_epochs = args.max_epochs // 2
        logging.info(f"pretrain_epochs not specified, using default: {args.pretrain_epochs}")

    if not hasattr(args, 'mask_ratio'):
        args.mask_ratio = 0.6
        logging.info(f"mask_ratio not specified, using default: {args.mask_ratio}")

    if not hasattr(args, 'patch_size'):
        args.patch_size = 4
        logging.info(f"patch_size not specified, using default: {args.patch_size}")

    if not hasattr(args, 'img_size'):
        args.img_size = 256
        logging.info(f"img_size not specified, using default: {args.img_size}")

    # Validate pretrain_epochs
    if args.pretrain_epochs < 0:
        raise ValueError("pretrain_epochs must be non-negative")

    if args.pretrain_epochs >= args.max_epochs:
        logging.warning(f"pretrain_epochs ({args.pretrain_epochs}) >= max_epochs ({args.max_epochs}), "
                        f"no finetuning will occur")

    # Validate class numbers
    num_diagnosis_classes = getattr(args, 'num_classes_diagnosis', 3)
    num_change_classes = getattr(args, 'num_classes_change', 7)

    if num_diagnosis_classes not in [3]:  # Only support 3 diagnosis classes for now
        raise ValueError(f"num_classes_diagnosis must be 3, got {num_diagnosis_classes}")

    if num_change_classes not in [7]:  # Only support 7 change classes for nine label version
        raise ValueError(f"num_classes_change must be 7, got {num_change_classes}")

    # Set default for n_folds if not specified
    if not hasattr(args, 'n_folds'):
        args.n_folds = 5
        logging.info(f"n_folds not specified, using default: {args.n_folds}")

    logging.info("✓ Trainer arguments validation passed")
    return True


def load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=None):
    """
    Load pretrained weights with optional decoder exclusion.

    Args:
        model: Target model
        pretrained_path: Path to pretrained weights
        exclude_decoder: Whether to exclude decoder weights
        logger: Logger instance

    Returns:
        dict: Loading information
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"Loading pretrained weights from: {pretrained_path}")

    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Get state_dict
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

    # Count original weights
    total_keys = len(state_dict)
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]

    logger.info(f"Original checkpoint contains {total_keys} parameters")
    logger.info(f"Found {len(decoder_keys)} decoder parameters")

    if exclude_decoder and decoder_keys:
        # Filter out decoder weights
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        logger.info(f"Excluded decoder parameters:")
        for key in decoder_keys:
            logger.info(f"  - {key}: {state_dict[key].shape}")

        state_dict = filtered_state_dict
        logger.info(f"Filtered state_dict contains {len(state_dict)} parameters")

    # Get current model parameters
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Analyze matching
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

    # Load weights with partial matching
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
    Remove decoder component from model with detailed logging.

    Args:
        model: Model to modify
        logger: Logger instance

    Returns:
        dict: Removal information
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info("REMOVING DECODER FROM MODEL")
    logger.info("=" * 60)

    # Get pre-removal parameter count
    total_params_before = sum(p.numel() for p in model.parameters())
    decoder_params = 0
    decoder_components = []

    # Check for decoder
    model_to_check = model.module if hasattr(model, 'module') else model

    if hasattr(model_to_check, 'decoder') and model_to_check.decoder is not None:
        # Count decoder parameters
        for name, param in model_to_check.decoder.named_parameters():
            decoder_params += param.numel()
            decoder_components.append((name, param.shape, param.numel()))

        logger.info(f"Decoder components to be removed:")
        for name, shape, numel in decoder_components:
            logger.info(f"  - {name}: {shape} ({numel:,} parameters)")

        logger.info(f"Total decoder parameters: {decoder_params:,}")

        # Remove decoder
        del model_to_check.decoder
        model_to_check.decoder = None

        logger.info("✓ Decoder successfully removed from model")

        # Re-wrap with DataParallel if needed
        if hasattr(model, 'module'):
            logger.info("Re-wrapping model with DataParallel...")
            device_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model_to_check, device_ids=device_ids)
            logger.info(f"✓ Model re-wrapped with DataParallel on devices: {device_ids}")
    else:
        logger.info("No decoder found in model or decoder already None")

    # Get post-removal parameter count
    total_params_after = sum(p.numel() for p in model.parameters())
    memory_saved = decoder_params * 4 / (1024 ** 2)  # Assuming float32, convert to MB

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
    Log detailed information about model components.

    Args:
        model: Model to analyze
        phase: Current phase name
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'=' * 50}")
    logger.info(f"MODEL COMPONENTS ANALYSIS - {phase.upper()}")
    logger.info(f"{'=' * 50}")

    model_to_check = model.module if hasattr(model, 'module') else model

    # Count parameters per component
    components = {}

    for name, module in model_to_check.named_children():
        if module is not None:
            param_count = sum(p.numel() for p in module.parameters())
            components[name] = param_count

            # Highlight important components
            if name in ['decoder', 'head_diagnosis', 'head_change', 'clinical_encoder', 'clinical_fusion']:
                status = "✓ Active" if param_count > 0 else "✗ None/Empty"
                logger.info(f"  {name:20}: {param_count:>10,} params {status}")
            else:
                logger.info(f"  {name:20}: {param_count:>10,} params")

    total_params = sum(components.values())
    logger.info(f"  {'=' * 40}")
    logger.info(f"  {'Total':20}: {total_params:>10,} params")

    # Check special component status
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

    logger.info(f"{'=' * 50}")

    return components


def generate_mask(input_size: Tuple[int, int], patch_size: int, mask_ratio: float,
                  device: torch.device) -> torch.Tensor:
    """Generate random mask for SimMIM at patch level"""
    H, W = input_size
    # Calculate number of patches
    h, w = H // patch_size, W // patch_size
    num_patches = h * w
    num_mask = int(num_patches * mask_ratio)

    # Randomly select patches to mask
    mask = torch.zeros(num_patches, dtype=torch.float32, device=device)
    mask_indices = torch.randperm(num_patches, device=device)[:num_mask]
    mask[mask_indices] = 1

    return mask


def norm_targets(targets, patch_size):
    """Normalize targets - from SimMIM"""
    assert patch_size % 2 == 1

    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.

    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                       count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                 count_include_pad=True) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)

    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

    return targets_


def create_data_loaders(config, phase='pretrain'):
    """
    Create data loaders with phase-specific batch sizes.

    Args:
        config: Configuration object
        phase: 'pretrain' or 'finetune'
    """
    # Temporarily adjust batch size
    original_batch_size = config.DATA.BATCH_SIZE

    if phase == 'pretrain':
        batch_size = getattr(config.DATA, 'BATCH_SIZE_PRETRAIN', config.DATA.BATCH_SIZE)
    else:  # finetune
        batch_size = getattr(config.DATA, 'BATCH_SIZE_FINETUNE', config.DATA.BATCH_SIZE)

    # Temporarily modify config
    config.defrost()
    config.DATA.BATCH_SIZE = batch_size
    config.freeze()

    # Create data loaders
    dataset_train, dataset_val, train_loader, val_loader, mixup_fn = build_loader_finetune(config)

    # Restore original config
    config.defrost()
    config.DATA.BATCH_SIZE = original_batch_size
    config.freeze()

    return dataset_train, dataset_val, train_loader, val_loader, mixup_fn


def create_fold_data_loaders(dataset_train, train_indices, val_indices, config, phase='finetune'):
    """
    Create data loaders for a specific cross-validation fold using Subset.

    Args:
        dataset_train: Full training dataset
        train_indices: Indices for training in this fold
        val_indices: Indices for validation in this fold
        config: Configuration object
        phase: 'pretrain' or 'finetune'

    Returns:
        train_loader, val_loader for this fold
    """
    if phase == 'pretrain':
        batch_size = getattr(config.DATA, 'BATCH_SIZE_PRETRAIN', config.DATA.BATCH_SIZE)
    else:
        batch_size = getattr(config.DATA, 'BATCH_SIZE_FINETUNE', config.DATA.BATCH_SIZE)

    num_workers = getattr(config.DATA, 'NUM_WORKERS', 4)
    pin_memory = getattr(config.DATA, 'PIN_MEMORY', True)

    train_subset = Subset(dataset_train, train_indices)
    val_subset = Subset(dataset_train, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


class EarlyStopping:
    """Early stopping mechanism"""

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
        """Save model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Filter decoder weights when saving
        model_to_save = model.module if hasattr(model, 'module') else model
        state_dict = model_to_save.state_dict()

        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        torch.save(filtered_state_dict, path)
        self.val_loss_min = val_loss

    def reset(self):
        """Reset early stopping state for a new fold"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


class SimMIMLoss(nn.Module):
    """SimMIM reconstruction loss"""

    def __init__(self, patch_size=4, norm_target=True, norm_target_patch_size=47):
        super().__init__()
        self.patch_size = patch_size
        self.norm_target = norm_target
        self.norm_target_patch_size = norm_target_patch_size

    def forward(self, input_images, reconstructed, mask):
        """
        Args:
            input_images: Original input images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
            mask: Patch-level mask [B, num_patches]
        """
        B, C, H, W = input_images.shape

        # Convert patch-level mask to pixel-level
        h, w = H // self.patch_size, W // self.patch_size
        mask_reshaped = mask.reshape(B, h, w)

        # Expand mask to pixel level
        mask_upsampled = mask_reshaped.unsqueeze(-1).unsqueeze(-1)
        mask_upsampled = mask_upsampled.repeat(1, 1, 1, self.patch_size, self.patch_size)
        mask_upsampled = mask_upsampled.permute(0, 1, 3, 2, 4).contiguous()
        mask_upsampled = mask_upsampled.view(B, H, W)
        mask_upsampled = mask_upsampled.unsqueeze(1).repeat(1, C, 1, 1)

        # Normalize target (if enabled)
        targets = input_images
        if self.norm_target:
            targets = norm_targets(targets, self.norm_target_patch_size)

        # Compute reconstruction loss (only on masked regions)
        loss_recon = F.l1_loss(targets, reconstructed, reduction='none')
        loss = (loss_recon * mask_upsampled).sum() / (mask_upsampled.sum() + 1e-5) / C

        return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss function - MMoE version - supports 7-class change label"""

    def __init__(self, weight_diagnosis=1.0, weight_change=1.0, label_smoothing=0.0,
                 num_diagnosis_classes=3, num_change_classes=7):
        super().__init__()
        self.criterion_diagnosis = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_change = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weight_diagnosis = weight_diagnosis
        self.weight_change = weight_change
        self.num_diagnosis_classes = num_diagnosis_classes
        self.num_change_classes = num_change_classes

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                diagnosis_labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses for both tasks.
        Note: Labels are 1-based, need to convert to 0-based.
        Returns:
            Dict with total_loss, diagnosis_loss, change_loss
        """
        output_diagnosis, output_change = outputs

        # Convert labels from 1-based to 0-based
        diagnosis_labels_zero_indexed = diagnosis_labels - 1
        change_labels_zero_indexed = change_labels - 1

        # Ensure labels are in valid range
        assert torch.all(diagnosis_labels_zero_indexed >= 0) and torch.all(
            diagnosis_labels_zero_indexed < self.num_diagnosis_classes), \
            f"Invalid diagnosis labels: {diagnosis_labels_zero_indexed}"
        assert torch.all(change_labels_zero_indexed >= 0) and torch.all(
            change_labels_zero_indexed < self.num_change_classes), \
            f"Invalid change labels: {change_labels_zero_indexed}"

        loss_diagnosis = self.criterion_diagnosis(output_diagnosis, diagnosis_labels_zero_indexed)
        loss_change = self.criterion_change(output_change, change_labels_zero_indexed)

        total_loss = self.weight_diagnosis * loss_diagnosis + self.weight_change * loss_change

        return {
            'total': total_loss,
            'diagnosis': loss_diagnosis,
            'change': loss_change
        }


def compute_metrics(outputs: Tuple[torch.Tensor, torch.Tensor],
                    diagnosis_labels: torch.Tensor, change_labels: torch.Tensor,
                    num_diagnosis_classes=3, num_change_classes=7) -> Dict[str, float]:
    """Compute evaluation metrics - MMoE version - supports 7-class"""
    output_diagnosis, output_change = outputs

    # Get predictions (0-based output, convert back to 1-based)
    pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1
    pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

    diagnosis_labels_np = diagnosis_labels.cpu().numpy()
    change_labels_np = change_labels.cpu().numpy()

    # Compute accuracy
    acc_diagnosis = accuracy_score(diagnosis_labels_np, pred_diagnosis)
    acc_change = accuracy_score(change_labels_np, pred_change)

    # Compute F1 scores
    diagnosis_labels_list = list(range(1, num_diagnosis_classes + 1))
    change_labels_list = list(range(1, num_change_classes + 1))

    f1_diagnosis = f1_score(diagnosis_labels_np, pred_diagnosis, labels=diagnosis_labels_list,
                            average='weighted', zero_division=0)
    f1_change = f1_score(change_labels_np, pred_change, labels=change_labels_list,
                         average='weighted', zero_division=0)

    return {
        'acc_diagnosis': acc_diagnosis,
        'acc_change': acc_change,
        'f1_diagnosis': f1_diagnosis,
        'f1_change': f1_change,
        'acc_avg': (acc_diagnosis + acc_change) / 2,
        'f1_avg': (f1_diagnosis + f1_change) / 2
    }


def log_expert_utilization(model, val_loader, device, epoch, num_experts=8):
    """Log expert utilization for analyzing MMoE behavior (supports clinical prior and 8 experts)"""
    model.eval()
    expert_weights_list = []

    with torch.no_grad():
        # Only analyze one batch
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # Get expert utilization
        try:
            expert_utilization = model.get_expert_utilization(
                images,
                clinical_prior=clinical_priors,
                lbls_diagnosis=diagnosis_labels,
                lbls_change=change_labels
            )

            if expert_utilization:
                # Compute average expert weights
                for layer_idx, gate_weights in enumerate(expert_utilization):
                    if 'diagnosis_weights' in gate_weights:
                        diagnosis_weights = gate_weights['diagnosis_weights'].mean(dim=[0, 1])  # [num_experts]
                        change_weights = gate_weights['change_weights'].mean(dim=[0, 1])  # [num_experts]

                        # Log to wandb - 8 experts
                        expert_names = ['Shared1', 'Shared2', 'CN1', 'CN2', 'MCI1', 'MCI2', 'AD1', 'AD2']
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
    """Pretrain one epoch - SimMIM reconstruction task"""
    model.train()

    # Ensure model is in pretrain mode
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
        # Get data
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        batch_size = images.shape[0]

        # Generate mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # Forward pass - SimMIM reconstruction
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,  # Convert to 0-based
            lbls_change=change_labels - 1,
            mask=masks
        )

        # Compute reconstruction loss
        loss = criterion_simmim(images, reconstructed, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        total_loss += loss.item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}",
            'lr': f"{current_lr:.2e}"
        })

    # Compute average loss
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_one_epoch_finetune(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                             use_timm_scheduler=False, position=0, num_diagnosis_classes=3, num_change_classes=7):
    """Finetune one epoch - classification tasks - supports 7-class"""
    model.train()

    # Ensure model is in finetune mode
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
        # Get data
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # Forward pass - classification
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,  # Convert to 0-indexed
            lbls_change=change_labels - 1
        )

        # Compute loss
        losses = criterion(outputs, diagnosis_labels, change_labels)
        loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not use_timm_scheduler:
            scheduler.step()

        # Record losses
        total_loss += loss.item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, diagnosis_labels, change_labels,
                                      num_diagnosis_classes, num_change_classes)
            all_metrics.append(metrics)

        # Update progress bar
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

    # Compute averages
    avg_loss = total_loss / len(train_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(train_loader)
    avg_change_loss = total_change_loss / len(train_loader)

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics


@torch.no_grad()
def validate_pretrain(model, val_loader, criterion_simmim, device, epoch,
                      mask_ratio=0.6, patch_size=4, img_size=256, position=2):
    """Pretrain validation - SimMIM reconstruction task"""
    model.eval()

    # Ensure model is in pretrain mode
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
        # Get data
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        batch_size = images.shape[0]

        # Generate mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # Forward pass
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,
            lbls_change=change_labels - 1,
            mask=masks
        )

        # Compute loss
        loss = criterion_simmim(images, reconstructed, masks)
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}"
        })

    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def validate_finetune(model, val_loader, criterion, device, epoch, position=2,
                      num_diagnosis_classes=3, num_change_classes=7):
    """Finetune validation - classification tasks - supports 7-class"""
    model.eval()

    # Ensure model is in finetune mode
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

    # For storing per-sample indices and predictions (used by majority voting)
    all_sample_indices = []

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Finetune Val Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # Get data
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # Forward pass
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )

        # Compute loss
        losses = criterion(outputs, diagnosis_labels, change_labels)

        # Record losses
        total_loss += losses['total'].item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # Compute metrics
        metrics = compute_metrics(outputs, diagnosis_labels, change_labels,
                                  num_diagnosis_classes, num_change_classes)
        all_metrics.append(metrics)

        # Collect predictions
        output_diagnosis, output_change = outputs
        pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1
        pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

        all_pred_diagnosis.extend(pred_diagnosis)
        all_true_diagnosis.extend(diagnosis_labels.cpu().numpy())
        all_pred_change.extend(pred_change)
        all_true_change.extend(change_labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'acc': f"{metrics['acc_avg']:.4f}"
        })

    # Compute averages
    avg_loss = total_loss / len(val_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(val_loader)
    avg_change_loss = total_change_loss / len(val_loader)

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Compute confusion matrices
    cm_diagnosis = confusion_matrix(all_true_diagnosis, all_pred_diagnosis,
                                    labels=list(range(1, num_diagnosis_classes + 1)))
    cm_change = confusion_matrix(all_true_change, all_pred_change,
                                 labels=list(range(1, num_change_classes + 1)))

    return (avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics, cm_diagnosis, cm_change,
            all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change)


def majority_vote(predictions_list):
    """
    Perform majority voting on a list of predictions for each sample.

    Args:
        predictions_list: list of prediction arrays, each of shape [num_samples_in_fold].
                          This is a dict mapping sample_index -> list of predictions across folds.

    Returns:
        voted_predictions: dict mapping sample_index -> majority-voted label
    """
    voted = {}
    for sample_idx, preds in predictions_list.items():
        # Use Counter to find the most common prediction
        counter = Counter(preds)
        # most_common(1) returns [(element, count)]
        voted[sample_idx] = counter.most_common(1)[0][0]
    return voted


def compute_majority_vote_metrics(sample_true_labels, sample_voted_preds,
                                  num_classes, task_name=""):
    """
    Compute metrics from majority-voted predictions.

    Args:
        sample_true_labels: dict mapping sample_index -> true label
        sample_voted_preds: dict mapping sample_index -> voted prediction
        num_classes: number of classes
        task_name: name of the task for logging

    Returns:
        dict with accuracy, f1, confusion_matrix
    """
    # Align indices
    indices = sorted(sample_true_labels.keys())
    true_labels = [sample_true_labels[i] for i in indices]
    pred_labels = [sample_voted_preds[i] for i in indices]

    labels_list = list(range(1, num_classes + 1))

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, labels=labels_list, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)

    logging.info(f"\n{'=' * 50}")
    logging.info(f"MAJORITY VOTE RESULTS - {task_name}")
    logging.info(f"{'=' * 50}")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Weighted F1: {f1:.4f}")
    logging.info(f"Total samples with votes: {len(indices)}")
    logging.info(f"\nClassification Report:")
    logging.info(classification_report(true_labels, pred_labels, labels=labels_list, zero_division=0))
    logging.info(f"{'=' * 50}")

    return {
        'accuracy': acc,
        'f1': f1,
        'confusion_matrix': cm
    }


def create_detailed_confusion_matrix_plot(cm, labels, title, figsize=(10, 8)):
    """Create detailed confusion matrix plot - supports 7x7 matrix"""
    fig, ax = plt.subplots(figsize=figsize)

    # Use heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                square=True, cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

    # Rotate labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


def get_dataset_labels(dataset):
    """
    Extract change_labels from dataset for stratified splitting.
    Handles both raw datasets and Subset wrappers.

    Args:
        dataset: PyTorch Dataset

    Returns:
        numpy array of change labels
    """
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample['change_label'] if isinstance(sample['change_label'], int)
                      else sample['change_label'].item())
    return np.array(labels)


def trainer_alzheimer_mmoe_nine_label(args, model, snapshot_path):
    """
    Alzheimer's disease dual-task trainer with MMoE - SimMIM pretraining + 7-class change label.
    Now with 5-Fold Cross-Validation and majority voting on accumulated validation predictions.
    """

    # Validate arguments early
    validate_trainer_args(args)

    model_name = getattr(args, 'model_name', 'swin_admoe_nine_label')
    if hasattr(args, 'MODEL') and hasattr(args.MODEL, 'NAME'):
        model_name = args.MODEL.NAME

    # Add timestamp and model name to snapshot path
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

    # Get class counts
    num_diagnosis_classes = getattr(args, 'num_classes_diagnosis', 3)
    num_change_classes = getattr(args, 'num_classes_change', 7)
    n_folds = getattr(args, 'n_folds', 5)

    logging.info(f"Number of diagnosis classes: {num_diagnosis_classes}")
    logging.info(f"Number of change classes: {num_change_classes}")
    logging.info(f"Number of cross-validation folds: {n_folds}")

    # Initialize wandb
    wandb_name = getattr(args, 'wandb_run_name', getattr(args, 'exp_name', 'alzheimer_mmoe_nine_label_run'))
    wandb_dir = os.path.join(os.path.dirname(snapshot_path), 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=getattr(args, 'wandb_project', 'alzheimer-mmoe-nine-label'),
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

    # ============== Training parameters ==============
    start_epoch = getattr(args, 'start_epoch', 0)
    resume_path = getattr(args, 'resume', None)
    pretrained_path = getattr(args, 'pretrained', None)
    pretrain_epochs = getattr(args, 'pretrain_epochs', args.max_epochs // 2)

    # Data loader config
    logging.info("Loading data...")
    if hasattr(args, 'config'):
        config = args.config
    elif hasattr(args, 'DATA'):
        config = args
    else:
        config = type('Config', (), {})()
        config.DATA = args.DATA if hasattr(args, 'DATA') else args

    # Get phase-specific batch sizes
    batch_size_pretrain = getattr(config.DATA, 'BATCH_SIZE_PRETRAIN', config.DATA.BATCH_SIZE)
    batch_size_finetune = getattr(config.DATA, 'BATCH_SIZE_FINETUNE', config.DATA.BATCH_SIZE)

    logging.info(f"Batch sizes - Pretrain: {batch_size_pretrain}, Finetune: {batch_size_finetune}")

    # ============== Load the FULL training dataset (used for CV splitting) ==============
    # We load the full dataset once; cross-validation splits will be done on indices.
    dataset_train, dataset_val_unused, full_train_loader, full_val_loader, mixup_fn = create_data_loaders(
        config, 'finetune'
    )
    logging.info(f"Full training set size: {len(dataset_train)}")

    # Save initial model state for resetting at each fold
    initial_model_state = copy.deepcopy(model.state_dict())

    # ============== Extract labels for stratified splitting ==============
    logging.info("Extracting labels for stratified cross-validation split...")
    try:
        all_change_labels = get_dataset_labels(dataset_train)
        logging.info(f"Successfully extracted {len(all_change_labels)} labels for stratification")
        use_stratified = True
    except Exception as e:
        logging.warning(f"Failed to extract labels for stratification: {e}. Using standard KFold instead.")
        use_stratified = False

    # ============== Setup K-Fold Cross-Validation ==============
    if use_stratified:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(np.arange(len(dataset_train)), all_change_labels))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(np.arange(len(dataset_train))))

    # ============== Majority voting accumulators ==============
    # Maps: sample_index -> list of predictions from each fold where it was in validation
    diagnosis_predictions_per_sample = defaultdict(list)
    change_predictions_per_sample = defaultdict(list)
    # Maps: sample_index -> true label (should be consistent)
    diagnosis_true_per_sample = {}
    change_true_per_sample = {}

    # Track per-fold best metrics
    fold_best_accs = []
    fold_best_f1s = []

    # SimMIM parameters
    mask_ratio = getattr(args, 'mask_ratio', 0.6)
    patch_size = getattr(args, 'patch_size', 4)
    img_size = getattr(args, 'img_size', 256)

    # Change label names for 7 classes
    change_label_names = [
        'Stable CN→CN',
        'Stable MCI→MCI',
        'Stable AD→AD',
        'Conv CN→MCI',
        'Conv MCI→AD',
        'Conv CN→AD',
        'Rev MCI→CN'
    ]
    diagnosis_names = ['CN', 'MCI', 'AD']

    logging.info(f"\nTraining plan (5-Fold Cross-Validation):")
    logging.info(f"- Total epochs per fold: {args.max_epochs}")
    logging.info(f"- Pretraining epochs: {pretrain_epochs} (SimMIM reconstruction)")
    logging.info(f"- Finetuning epochs: {args.max_epochs - pretrain_epochs} (Classification)")
    logging.info(f"- Number of folds: {n_folds}")
    logging.info(f"- SimMIM mask ratio: {mask_ratio}")
    logging.info(f"- Change labels: {change_label_names}")

    # ============== CROSS-VALIDATION LOOP ==============
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        logging.info(f"\n{'#' * 80}")
        logging.info(f"# FOLD {fold_idx + 1}/{n_folds}")
        logging.info(f"# Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        logging.info(f"{'#' * 80}")

        # Create fold-specific output directory
        fold_snapshot_path = os.path.join(snapshot_path, f'fold_{fold_idx + 1}')
        os.makedirs(fold_snapshot_path, exist_ok=True)

        # ============== Reset model to initial state for each fold ==============
        model.load_state_dict(copy.deepcopy(initial_model_state))
        model = model.to(device)

        # ============== Weight loading logic (per fold) ==============
        if resume_path:
            logging.info(f"[Fold {fold_idx+1}] Resuming training from checkpoint...")
            load_info = load_pretrained_weights(model, resume_path, exclude_decoder=False, logger=logging.getLogger())
            checkpoint = torch.load(resume_path, map_location='cpu')
            fold_start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"[Fold {fold_idx+1}] Resumed from epoch {fold_start_epoch}")
        elif pretrained_path:
            logging.info(f"[Fold {fold_idx+1}] Loading pretrained weights (excluding decoder)...")
            load_info = load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=logging.getLogger())
            fold_start_epoch = start_epoch
        else:
            fold_start_epoch = start_epoch

        # Log initial model components
        log_model_components(model, f"Fold {fold_idx+1} Initial", logging.getLogger())

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs")

        # ============== Create fold-specific data loaders ==============
        current_phase = 'pretrain' if pretrain_epochs > 0 else 'finetune'
        train_loader, val_loader = create_fold_data_loaders(
            dataset_train, train_indices, val_indices, config, current_phase
        )
        logging.info(f"[Fold {fold_idx+1}] Created data loaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # ============== Loss functions ==============
        # SimMIM loss (pretraining)
        criterion_simmim = SimMIMLoss(
            patch_size=getattr(args, 'patch_size', 4),
            norm_target=getattr(args, 'norm_target', True),
            norm_target_patch_size=getattr(args, 'norm_target_patch_size', 47)
        )

        # Classification loss (finetuning) - 7-class
        criterion_classification = MultiTaskLoss(
            weight_diagnosis=getattr(args, 'weight_diagnosis', 1.0),
            weight_change=getattr(args, 'weight_change', 1.0),
            label_smoothing=getattr(args, 'label_smoothing', 0.0),
            num_diagnosis_classes=num_diagnosis_classes,
            num_change_classes=num_change_classes
        )

        # ============== Optimizer ==============
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )

        # ============== Scheduler ==============
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

        logging.info(f"[Fold {fold_idx+1}] Using cosine scheduler with warmup")
        logging.info(f"[Fold {fold_idx+1}] Warmup epochs: {warmup_epochs}, Base LR: {args.base_lr}")

        # Early stopping (reset per fold)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        # Track best val accuracy for this fold
        best_val_acc = 0
        decoder_removed = False

        overall_pbar = tqdm(
            total=args.max_epochs - fold_start_epoch,
            desc=f"Fold {fold_idx+1}/{n_folds} Progress",
            position=0,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for epoch in range(fold_start_epoch, args.max_epochs):
            logging.info(f"\n{'=' * 50}")
            logging.info(f"[Fold {fold_idx+1}] Epoch {epoch}/{args.max_epochs - 1}")

            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Learning rate: {current_lr:.6f}")

            is_pretrain = epoch < pretrain_epochs
            phase = "Pretrain (SimMIM)" if is_pretrain else "Finetune (Classification)"
            logging.info(f"Phase: {phase}")

            # ============== Phase switch logic ==============
            if not is_pretrain and not decoder_removed:
                logging.info(f"\n[Fold {fold_idx+1}] SWITCHING FROM PRETRAINING TO FINETUNING")

                # Remove decoder
                remove_info = remove_decoder_from_model(model, logging.getLogger())
                decoder_removed = True

                # Log removal info to wandb
                wandb.log({
                    f'fold_{fold_idx+1}/model_modification/decoder_removed': remove_info['removed'],
                    f'fold_{fold_idx+1}/model_modification/decoder_params_removed': remove_info['decoder_params'],
                    f'fold_{fold_idx+1}/model_modification/memory_saved_mb': remove_info['memory_saved_mb'],
                    'epoch': epoch
                })

                # Recreate data loaders for finetuning phase
                logging.info(f"[Fold {fold_idx+1}] Recreating data loaders for finetuning phase...")
                train_loader, val_loader = create_fold_data_loaders(
                    dataset_train, train_indices, val_indices, config, 'finetune'
                )
                num_steps_per_epoch = len(train_loader)
                logging.info(f"[Fold {fold_idx+1}] Data loaders recreated with batch size: {batch_size_finetune}")

                # Recreate optimizer (parameters may have changed)
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.base_lr,
                    weight_decay=args.weight_decay
                )

                # Recreate scheduler
                remaining_steps = num_steps_per_epoch * (args.max_epochs - epoch)
                remaining_warmup = max(0, warmup_steps - epoch * num_steps_per_epoch)

                def new_lr_lambda(current_step):
                    if current_step < remaining_warmup:
                        return float(current_step) / float(max(1, remaining_warmup))
                    progress = float(current_step - remaining_warmup) / float(
                        max(1, remaining_steps - remaining_warmup))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

                scheduler = LambdaLR(optimizer, new_lr_lambda)

                logging.info(f"[Fold {fold_idx+1}] Optimizer and scheduler recreated for finetuning phase")
                log_model_components(model, f"Fold {fold_idx+1} After Decoder Removal", logging.getLogger())

            if is_pretrain:
                # ===== Pretrain phase: SimMIM reconstruction =====
                train_loss = train_one_epoch_pretrain(
                    model, train_loader, criterion_simmim, optimizer, scheduler, device, epoch,
                    mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size, position=1
                )

                # Log training metrics
                wandb.log({
                    f'fold_{fold_idx+1}/train/loss_simmim': train_loss,
                    f'fold_{fold_idx+1}/train/lr': current_lr,
                    f'fold_{fold_idx+1}/train/phase': 1,
                    'epoch': epoch
                })

                logging.info(f"[Fold {fold_idx+1}] Pretrain - SimMIM Loss: {train_loss:.4f}")

                # Validation (every eval_interval epochs)
                if (epoch + 1) % args.eval_interval == 0:
                    val_loss = validate_pretrain(
                        model, val_loader, criterion_simmim, device, epoch,
                        mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size
                    )

                    wandb.log({
                        f'fold_{fold_idx+1}/val/loss_simmim': val_loss,
                        'epoch': epoch
                    })

                    logging.info(f"[Fold {fold_idx+1}] Pretrain Val - SimMIM Loss: {val_loss:.4f}")

                    early_stopping(val_loss, model, os.path.join(fold_snapshot_path, 'pretrain_checkpoint.pth'))
                    if early_stopping.early_stop:
                        logging.info(f"[Fold {fold_idx+1}] Early stopping triggered during pretraining!")
                        # Reset early stopping for finetuning phase
                        early_stopping.reset()
                        break

            else:
                # ===== Finetune phase: Classification tasks =====
                train_loss, train_diagnosis_loss, train_change_loss, train_metrics = train_one_epoch_finetune(
                    model, train_loader, criterion_classification, optimizer, scheduler, device, epoch,
                    use_timm_scheduler=False, position=1,
                    num_diagnosis_classes=num_diagnosis_classes,
                    num_change_classes=num_change_classes
                )

                # Log training metrics
                wandb.log({
                    f'fold_{fold_idx+1}/train/loss': train_loss,
                    f'fold_{fold_idx+1}/train/loss_diagnosis': train_diagnosis_loss,
                    f'fold_{fold_idx+1}/train/loss_change': train_change_loss,
                    f'fold_{fold_idx+1}/train/acc_diagnosis': train_metrics['acc_diagnosis'],
                    f'fold_{fold_idx+1}/train/acc_change': train_metrics['acc_change'],
                    f'fold_{fold_idx+1}/train/acc_avg': train_metrics['acc_avg'],
                    f'fold_{fold_idx+1}/train/f1_diagnosis': train_metrics['f1_diagnosis'],
                    f'fold_{fold_idx+1}/train/f1_change': train_metrics['f1_change'],
                    f'fold_{fold_idx+1}/train/f1_avg': train_metrics['f1_avg'],
                    f'fold_{fold_idx+1}/train/lr': current_lr,
                    'epoch': epoch
                })

                logging.info(f"[Fold {fold_idx+1}] Finetune - Loss: {train_loss:.4f}, Acc: {train_metrics['acc_avg']:.4f}")

                # Validation (every eval_interval epochs)
                if (epoch + 1) % args.eval_interval == 0:
                    val_results = validate_finetune(model, val_loader, criterion_classification, device, epoch,
                                                    num_diagnosis_classes=num_diagnosis_classes,
                                                    num_change_classes=num_change_classes)

                    if len(val_results) == 6:
                        val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change = val_results
                        all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = None, None, None, None
                    else:
                        val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change, \
                            all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = val_results

                    # ============== Accumulate predictions for majority voting ==============
                    # Map validation sample indices to their predictions from this fold
                    if all_pred_diagnosis is not None and all_pred_change is not None:
                        for local_idx, global_idx in enumerate(val_indices):
                            if local_idx < len(all_pred_diagnosis):
                                diagnosis_predictions_per_sample[global_idx].append(all_pred_diagnosis[local_idx])
                                change_predictions_per_sample[global_idx].append(all_pred_change[local_idx])
                                # Store true labels (should be consistent across folds)
                                diagnosis_true_per_sample[global_idx] = all_true_diagnosis[local_idx]
                                change_true_per_sample[global_idx] = all_true_change[local_idx]

                    # Log validation metrics
                    wandb.log({
                        f'fold_{fold_idx+1}/val/loss': val_loss,
                        f'fold_{fold_idx+1}/val/loss_diagnosis': val_diagnosis_loss,
                        f'fold_{fold_idx+1}/val/loss_change': val_change_loss,
                        f'fold_{fold_idx+1}/val/acc_diagnosis': val_metrics['acc_diagnosis'],
                        f'fold_{fold_idx+1}/val/acc_change': val_metrics['acc_change'],
                        f'fold_{fold_idx+1}/val/acc_avg': val_metrics['acc_avg'],
                        f'fold_{fold_idx+1}/val/f1_diagnosis': val_metrics['f1_diagnosis'],
                        f'fold_{fold_idx+1}/val/f1_change': val_metrics['f1_change'],
                        f'fold_{fold_idx+1}/val/f1_avg': val_metrics['f1_avg'],
                        'epoch': epoch
                    })

                    # Log expert utilization (only during finetuning)
                    log_expert_utilization(model, val_loader, device, epoch, num_experts=8)

                    # Log confusion matrices
                    fig_cm_diagnosis = create_detailed_confusion_matrix_plot(
                        cm_diagnosis, diagnosis_names,
                        f'Fold {fold_idx+1} - Confusion Matrix - Diagnosis (3 classes)',
                        figsize=(8, 6)
                    )

                    fig_cm_change = create_detailed_confusion_matrix_plot(
                        cm_change, change_label_names,
                        f'Fold {fold_idx+1} - Confusion Matrix - Change Label (7 classes)',
                        figsize=(12, 10)
                    )

                    wandb.log({
                        f'fold_{fold_idx+1}/val/confusion_matrix_diagnosis': wandb.Image(fig_cm_diagnosis),
                        f'fold_{fold_idx+1}/val/confusion_matrix_change': wandb.Image(fig_cm_change),
                        'epoch': epoch
                    })

                    plt.close(fig_cm_diagnosis)
                    plt.close(fig_cm_change)

                    logging.info(f"[Fold {fold_idx+1}] Finetune Val - Loss: {val_loss:.4f}, Acc: {val_metrics['acc_avg']:.4f}")

                    # Save best model (only during finetuning)
                    if val_metrics['acc_avg'] > best_val_acc:
                        best_val_acc = val_metrics['acc_avg']
                        best_model_path = os.path.join(fold_snapshot_path, 'best_model.pth')

                        model_to_save = model.module if hasattr(model, 'module') else model

                        # Filter decoder weights when saving
                        state_dict = model_to_save.state_dict()
                        filtered_state_dict = {
                            k: v for k, v in state_dict.items()
                            if not k.startswith('decoder.')
                        }

                        torch.save(filtered_state_dict, best_model_path)
                        logging.info(f"[Fold {fold_idx+1}] Best model saved with acc: {best_val_acc:.4f}")

                        wandb.run.summary[f'fold_{fold_idx+1}/best_val_acc'] = best_val_acc
                        wandb.run.summary[f'fold_{fold_idx+1}/best_epoch'] = epoch

                    # Early stopping based on classification accuracy
                    early_stopping(-val_metrics['acc_avg'], model,
                                   os.path.join(fold_snapshot_path, 'finetune_checkpoint.pth'))
                    if early_stopping.early_stop:
                        logging.info(f"[Fold {fold_idx+1}] Early stopping triggered during finetuning!")
                        break

            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_path = os.path.join(fold_snapshot_path, f'checkpoint_epoch_{epoch}.pth')
                model_to_save = model.module if hasattr(model, 'module') else model

                # Filter decoder weights when saving
                state_dict = model_to_save.state_dict()
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('decoder.')
                }

                torch.save({
                    'epoch': epoch,
                    'fold': fold_idx + 1,
                    'model_state_dict': filtered_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'is_pretrain': is_pretrain,
                    'phase': phase,
                    'decoder_removed': decoder_removed
                }, checkpoint_path)

                logging.info(f"[Fold {fold_idx+1}] Checkpoint saved at epoch {epoch}")

            overall_pbar.update(1)

        overall_pbar.close()

        # Save final model for this fold
        final_model_path = os.path.join(fold_snapshot_path, 'final_model.pth')
        model_to_save = model.module if hasattr(model, 'module') else model
        state_dict = model_to_save.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }
        torch.save(filtered_state_dict, final_model_path)

        log_model_components(model, f"Fold {fold_idx+1} Final", logging.getLogger())

        # Record fold best metrics
        fold_best_accs.append(best_val_acc)
        logging.info(f"[Fold {fold_idx+1}] Completed. Best validation accuracy: {best_val_acc:.4f}")

        # Unwrap DataParallel for next fold reset
        if hasattr(model, 'module'):
            model = model.module

    # ============== MAJORITY VOTING ACROSS ALL FOLDS ==============
    logging.info(f"\n{'#' * 80}")
    logging.info(f"# MAJORITY VOTING ACROSS ALL {n_folds} FOLDS")
    logging.info(f"{'#' * 80}")

    # Perform majority voting on accumulated predictions
    voted_diagnosis = majority_vote(diagnosis_predictions_per_sample)
    voted_change = majority_vote(change_predictions_per_sample)

    logging.info(f"Total samples with accumulated votes (diagnosis): {len(voted_diagnosis)}")
    logging.info(f"Total samples with accumulated votes (change): {len(voted_change)}")

    # Log vote count distribution
    vote_counts = [len(v) for v in diagnosis_predictions_per_sample.values()]
    if vote_counts:
        logging.info(f"Vote count distribution - Min: {min(vote_counts)}, Max: {max(vote_counts)}, "
                      f"Mean: {np.mean(vote_counts):.2f}")

    # Compute majority vote metrics for diagnosis task
    if len(voted_diagnosis) > 0 and len(diagnosis_true_per_sample) > 0:
        mv_diagnosis_metrics = compute_majority_vote_metrics(
            diagnosis_true_per_sample, voted_diagnosis,
            num_diagnosis_classes, "Diagnosis (Majority Vote)"
        )

        # Log majority vote confusion matrix for diagnosis
        fig_mv_diag = create_detailed_confusion_matrix_plot(
            mv_diagnosis_metrics['confusion_matrix'], diagnosis_names,
            'Majority Vote - Confusion Matrix - Diagnosis (3 classes)',
            figsize=(8, 6)
        )
        wandb.log({
            'majority_vote/acc_diagnosis': mv_diagnosis_metrics['accuracy'],
            'majority_vote/f1_diagnosis': mv_diagnosis_metrics['f1'],
            'majority_vote/confusion_matrix_diagnosis': wandb.Image(fig_mv_diag),
        })
        plt.close(fig_mv_diag)

    # Compute majority vote metrics for change task
    if len(voted_change) > 0 and len(change_true_per_sample) > 0:
        mv_change_metrics = compute_majority_vote_metrics(
            change_true_per_sample, voted_change,
            num_change_classes, "Change Label (Majority Vote)"
        )

        # Log majority vote confusion matrix for change
        fig_mv_change = create_detailed_confusion_matrix_plot(
            mv_change_metrics['confusion_matrix'], change_label_names,
            'Majority Vote - Confusion Matrix - Change Label (7 classes)',
            figsize=(12, 10)
        )
        wandb.log({
            'majority_vote/acc_change': mv_change_metrics['accuracy'],
            'majority_vote/f1_change': mv_change_metrics['f1'],
            'majority_vote/confusion_matrix_change': wandb.Image(fig_mv_change),
        })
        plt.close(fig_mv_change)

    # ============== SUMMARY ACROSS FOLDS ==============
    logging.info(f"\n{'=' * 80}")
    logging.info("CROSS-VALIDATION SUMMARY")
    logging.info(f"{'=' * 80}")

    for i, acc in enumerate(fold_best_accs):
        logging.info(f"  Fold {i+1}: Best Acc = {acc:.4f}")

    mean_acc = np.mean(fold_best_accs)
    std_acc = np.std(fold_best_accs)
    logging.info(f"\n  Mean Best Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    if len(voted_diagnosis) > 0:
        logging.info(f"  Majority Vote Diagnosis Acc: {mv_diagnosis_metrics['accuracy']:.4f}")
        logging.info(f"  Majority Vote Diagnosis F1: {mv_diagnosis_metrics['f1']:.4f}")
    if len(voted_change) > 0:
        logging.info(f"  Majority Vote Change Acc: {mv_change_metrics['accuracy']:.4f}")
        logging.info(f"  Majority Vote Change F1: {mv_change_metrics['f1']:.4f}")

    logging.info(f"{'=' * 80}")

    # Log summary to wandb
    wandb.run.summary['cv/mean_best_acc'] = mean_acc
    wandb.run.summary['cv/std_best_acc'] = std_acc
    if len(voted_diagnosis) > 0:
        wandb.run.summary['cv/majority_vote_diagnosis_acc'] = mv_diagnosis_metrics['accuracy']
        wandb.run.summary['cv/majority_vote_diagnosis_f1'] = mv_diagnosis_metrics['f1']
    if len(voted_change) > 0:
        wandb.run.summary['cv/majority_vote_change_acc'] = mv_change_metrics['accuracy']
        wandb.run.summary['cv/majority_vote_change_f1'] = mv_change_metrics['f1']

    # Save majority vote results to file
    mv_results_path = os.path.join(snapshot_path, 'majority_vote_results.npz')
    np.savez(
        mv_results_path,
        diagnosis_voted=np.array([(k, v) for k, v in voted_diagnosis.items()]),
        change_voted=np.array([(k, v) for k, v in voted_change.items()]),
        diagnosis_true=np.array([(k, v) for k, v in diagnosis_true_per_sample.items()]),
        change_true=np.array([(k, v) for k, v in change_true_per_sample.items()]),
        fold_best_accs=np.array(fold_best_accs)
    )
    logging.info(f"Majority vote results saved to {mv_results_path}")

    wandb.finish()

    logging.info(f"\nTraining completed!")
    return "Training Finished!"


# Example Args class - with SimMIM and CV parameters
class Args:
    def __init__(self):
        # Basic parameters
        self.seed = 42
        self.max_epochs = 100
        self.eval_interval = 10
        self.save_interval = 20
        self.patience = 10

        # Optimizer parameters
        self.base_lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4

        # Loss function parameters
        self.weight_diagnosis = 1.0
        self.weight_change = 1.0
        self.label_smoothing = 0.1

        # Training phase parameters
        self.pretrain_epochs = 50

        # SimMIM parameters
        self.mask_ratio = 0.6
        self.patch_size = 4
        self.img_size = 256
        self.norm_target = True
        self.norm_target_patch_size = 47

        # wandb parameters
        self.wandb_project = "alzheimer-mmoe-nine-label"
        self.exp_name = "dual-task-mmoe-simmim-7class-5fold-cv"

        # Weight loading parameters
        self.resume = None
        self.pretrained = None

        # Cross-validation parameters
        self.n_folds = 5

        # Data parameters
        self.DATA = type('obj', (object,), {
            'DATASET': 'alzheimer',
            'DATA_PATH': 'path/to/data',
            'IMG_SIZE': 256,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'PIN_MEMORY': True
        })

        # Class counts
        self.num_classes_diagnosis = 3  # CN, MCI, AD
        self.num_classes_change = 7  # 7 detailed change classes


if __name__ == "__main__":
    # Example usage
    args = Args()
    # args.pretrained = "path/to/pretrained_weights.pth"
    # args.resume = "path/to/checkpoint.pth"

    # model = SwinTransformerV2_AlzheimerMMoE_NineLabel(...)
    # snapshot_path = "./checkpoints/mmoe_nine_label_exp1"
    # trainer_alzheimer_mmoe_nine_label(args, model, snapshot_path)