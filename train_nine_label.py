"""
Description: 
Author: JeffreyJ
Date: 2025/7/14
LastEditTime: 2025/7/14 12:56
Version: 2.0 - With 5-Fold Cross-Validation Support
"""
"""
Description: 
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:22
Version: 2.0 - Nine Label Version
"""
# !/usr/bin/env python
"""
Alzheimer's Disease Dual-Task Classification Training Script with SimMIM Pretraining
Nine Label Version - 7 Change Classes - 5-Fold Cross-Validation with Majority Voting

- Pretrain Phase: SimMIM reconstruction task
- Finetune Phase: Dual classification tasks
  - Diagnosis Task: CN(1), MCI(2), Dementia(3)
  - Change Task: 7 detailed transition types
    1: Stable CN to CN
    2: Stable MCI to MCI  
    3: Stable AD to AD
    4: Conversion CN to MCI
    5: Conversion MCI to AD
    6: Conversion CN to AD
    7: Reversion MCI to CN
- 5-Fold Cross-Validation: Each fold trains independently, validation predictions
  are accumulated across folds and a majority vote determines the final label.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime

# Import custom modules
from config_nine_label import get_config, validate_nine_label_config, print_config_summary
from models import build_model
from trainer_nine_label import trainer_alzheimer_mmoe_nine_label
from logger import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
import logging

# Configure warning filters
warnings.filterwarnings("ignore", message=".*Fused window process.*")
warnings.filterwarnings("ignore", message=".*Tutel.*")

# Reduce logging level for certain modules
logging.getLogger("models").setLevel(logging.ERROR)

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Ignore FutureWarnings from specific libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ignore UserWarning (e.g. PyTorch tips)
warnings.filterwarnings("ignore", category=UserWarning)

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('Alzheimer Nine Label (7-Class Change) Training with SimMIM Pretraining + 5-Fold CV')

    # Basic settings configs/swin_admoe/swin_admoe_tiny_nine_label_patch4_window16_256.yaml
    parser.add_argument('--cfg', type=str,
                        default=r'D:\codebase\Swin-Transformer\configs\swin_admoe\swin_admoe_tiny_nine_label_patch4_window16_256.yaml',
                        metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs",
                        default=None, nargs='+')

    # Data settings
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

    # Model settings
    parser.add_argument('--pretrained', help='pretrained weight path')
    parser.add_argument('--resume', help='resume from checkpoint')

    # Training settings
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <o>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')

    # Distributed training - Fix for LOCAL_RANK issue
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')
    else:
        # For PyTorch 2.x, make local_rank optional and default to 0
        parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')

    # Optimization settings
    parser.add_argument('--base-lr', type=float, help='base learning rate')
    parser.add_argument('--weight-decay', type=float, help='weight decay')
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")

    # GPU settings
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable-amp', action='store_true', help='Disable automatic mixed precision training')

    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='alzheimer-nine-label',
                        help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, help='wandb run name')
    parser.add_argument('--wandb-offline', action='store_true', help='disable wandb online sync')

    # Task weights (for finetuning phase)
    parser.add_argument('--weight-diagnosis', type=float, default=1.0,
                        help='weight for diagnosis task loss')
    parser.add_argument('--weight-change', type=float, default=1.0,
                        help='weight for change task loss')

    # SimMIM specific settings
    parser.add_argument('--mask-ratio', type=float, default=0.6,
                        help='mask ratio for SimMIM pretraining')
    parser.add_argument('--norm-target', action='store_true', default=True,
                        help='normalize target for SimMIM')
    parser.add_argument('--norm-target-patch-size', type=int, default=47,
                        help='patch size for target normalization')

    # Training phase control
    parser.add_argument('--pretrain-epochs', type=int, help='number of pretraining epochs')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='skip pretraining phase and go directly to finetuning')

    # Nine label specific settings
    parser.add_argument('--num-change-classes', type=int, default=7,
                        help='number of change classes (default: 7 for nine label version)')
    parser.add_argument('--num-experts', type=int, default=8,
                        help='number of experts (default: 8 for nine label version)')
    parser.add_argument('--detailed-eval', action='store_true',
                        help='enable detailed evaluation reports')
    parser.add_argument('--save-predictions', action='store_true',
                        help='save prediction results for analysis')

    # ShiftedBlock specific settings
    parser.add_argument('--use-shifted-last-layer', action='store_true',
                        help='use ShiftedBlock in the last layer instead of BasicLayerMMoE')
    parser.add_argument('--shift-mlp-ratio', type=float, default=1.0,
                        help='MLP ratio for ShiftedBlock (default: 1.0)')

    # Cross-validation settings
    parser.add_argument('--n-folds', type=int, default=5,
                        help='number of folds for cross-validation (default: 5)')

    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training"""
    # Check if distributed training is actually being used
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        # Only initialize distributed if world_size > 1
        if world_size > 1:
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            dist.barrier()
        else:
            # Single GPU, no need for distributed
            rank = 0
            world_size = 1
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        print("Single GPU training mode")

    return rank, world_size


def set_random_seed(seed, deterministic=True):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def prepare_config(args):
    """Prepare configuration"""
    config = get_config(args)

    # Override config with command line arguments
    if args.batch_size:
        config.defrost()
        config.DATA.BATCH_SIZE = args.batch_size
        config.freeze()

    if args.data_path:
        config.defrost()
        config.DATA.DATA_PATH = args.data_path
        config.freeze()

    if args.base_lr:
        config.defrost()
        config.TRAIN.BASE_LR = args.base_lr
        config.freeze()

    if args.weight_decay:
        config.defrost()
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
        config.freeze()

    if args.accumulation_steps > 1:
        config.defrost()
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
        config.freeze()

    if args.use_checkpoint:
        config.defrost()
        config.TRAIN.USE_CHECKPOINT = True
        config.freeze()

    if args.disable_amp:
        config.defrost()
        config.TRAIN.AMP_ENABLE = False
        config.freeze()

    # SimMIM specific overrides
    if args.mask_ratio:
        config.defrost()
        config.MODEL.SIMMIM.MASK_RATIO = args.mask_ratio
        config.freeze()

    if args.norm_target is not None:
        config.defrost()
        config.MODEL.SIMMIM.NORM_TARGET.ENABLE = args.norm_target
        config.freeze()

    if args.norm_target_patch_size:
        config.defrost()
        config.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = args.norm_target_patch_size
        config.freeze()

    if args.pretrain_epochs:
        config.defrost()
        config.TRAIN.PRETRAIN_EPOCHS = args.pretrain_epochs
        config.freeze()

    if args.skip_pretrain:
        config.defrost()
        config.TRAIN.PRETRAIN_EPOCHS = 0  # Skip pretraining
        config.freeze()

    # Nine label specific overrides
    if args.num_change_classes:
        config.defrost()
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE = args.num_change_classes
        config.freeze()

    if args.num_experts:
        config.defrost()
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS = args.num_experts
        config.freeze()

    if args.detailed_eval:
        config.defrost()
        config.EVAL.DETAILED_REPORT = True
        config.freeze()

    if args.save_predictions:
        config.defrost()
        config.EVAL.SAVE_PREDICTIONS = True
        config.freeze()

    # ShiftedBlock specific overrides
    if args.use_shifted_last_layer:
        config.defrost()
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.USE_SHIFTED_LAST_LAYER = True
        config.freeze()

    if args.shift_mlp_ratio:
        config.defrost()
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.SHIFT_MLP_RATIO = args.shift_mlp_ratio
        config.freeze()

    # Set output directory
    config.defrost()
    config.OUTPUT = args.output  # Use base output directory only
    config.freeze()

    return config


def log_change_label_distribution(train_loader, val_loader, logger):
    """Log change label distribution"""
    logger.info("\n" + "=" * 60)
    logger.info("CHANGE LABEL DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    change_label_names = {
        1: "Stable CN→CN",
        2: "Stable MCI→MCI",
        3: "Stable AD→AD",
        4: "Conv CN→MCI",
        5: "Conv MCI→AD",
        6: "Conv CN→AD",
        7: "Rev MCI→CN"
    }

    def count_labels(loader, split_name):
        label_counts = {i: 0 for i in range(1, 8)}
        total = 0

        for batch in loader:
            change_labels = batch['change_label'].numpy()
            for label in change_labels:
                label_counts[label] += 1
                total += 1

        logger.info(f"\n{split_name} Set Distribution:")
        logger.info(f"Total samples: {total}")
        for label, count in label_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            logger.info(f"  {label}: {change_label_names[label]:<20} - {count:>5} ({percentage:>5.1f}%)")

    count_labels(train_loader, "Training")
    count_labels(val_loader, "Validation")

    logger.info("=" * 60)


def main():
    """Main training function with 5-Fold Cross-Validation"""
    # Parse arguments
    args = parse_args()

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Prepare config
    config = prepare_config(args)

    # Validate nine label configuration
    if config.MODEL.TYPE == 'swin_admoe_nine_label':
        validate_nine_label_config(config)

    # Set random seed
    seed = config.SEED + rank
    set_random_seed(seed, deterministic=args.deterministic)

    # Create output directory
    os.makedirs(config.OUTPUT, exist_ok=True)

    # Create logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank, name=f"{config.MODEL.NAME}")

    # Log config
    if rank == 0:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        # Print configuration summary
        print_config_summary(config)
        logger.info(config.dump())

    # Build model
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    # Calculate model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_parameters:,}")

    # Load pretrained weights if specified
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)

    # Move model to GPU
    model.cuda()

    # Create trainer arguments
    trainer_args = argparse.Namespace(
        # Basic settings
        seed=config.SEED,
        output_dir=config.OUTPUT,
        model_name=config.MODEL.NAME,
        tag=config.TAG,

        # Data settings
        data_path=config.DATA.DATA_PATH,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        img_size=config.DATA.IMG_SIZE,

        # Model settings
        num_classes=config.MODEL.NUM_CLASSES,
        num_classes_diagnosis=config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_DIAGNOSIS,
        num_classes_change=config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE,

        # Training settings
        max_epochs=config.TRAIN.EPOCHS,
        eval_interval=config.EVAL.INTERVAL,
        save_interval=config.SAVE_FREQ,

        # Optimization settings
        base_lr=config.TRAIN.BASE_LR,
        min_lr=config.TRAIN.MIN_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,

        # Task weights (for finetuning phase)
        weight_diagnosis=args.weight_diagnosis,
        weight_change=args.weight_change,

        # Early stopping
        patience=config.EARLY_STOP.PATIENCE,

        # WandB settings
        wandb_project=args.wandb_project or config.WANDB.PROJECT,
        wandb_run_name=args.wandb_run_name or f"{config.MODEL.NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        wandb_offline=args.wandb_offline,

        # Warmup settings
        warmup_epochs=getattr(config.TRAIN, 'WARMUP_EPOCHS', 5),
        warmup_lr=getattr(config.TRAIN, 'WARMUP_LR', 1e-6),

        # SimMIM pretraining settings
        pretrain_epochs=getattr(config.TRAIN, 'PRETRAIN_EPOCHS', config.TRAIN.EPOCHS // 2),
        mask_ratio=getattr(config.MODEL.SIMMIM, 'MASK_RATIO', 0.6),
        patch_size=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PATCH_SIZE', 4),
        norm_target=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'ENABLE', True),
        norm_target_patch_size=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'PATCH_SIZE', 47),

        # Evaluation settings
        detailed_eval=config.EVAL.DETAILED_REPORT,
        save_predictions=config.EVAL.SAVE_PREDICTIONS,

        # Config object
        config=config,

        # Distributed settings
        rank=rank,
        world_size=world_size,
        local_rank=args.local_rank,

        # Cross-validation settings
        n_folds=args.n_folds,
    )

    # Add data-specific attributes
    trainer_args.DATA = config.DATA
    trainer_args.AUG = config.AUG
    trainer_args.MODEL = config.MODEL
    trainer_args.TRAIN = config.TRAIN

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        trainer_args.start_epoch = checkpoint['epoch'] + 1

        # Check if there's training phase info
        if 'phase' in checkpoint:
            logger.info(f"Resumed from {checkpoint['phase']} phase")

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        trainer_args.start_epoch = 0

    # Evaluation only mode
    if args.eval:
        logger.info("Evaluation mode")
        raise NotImplementedError("Evaluation mode not implemented yet for nine label version")

    # Log training plan
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PLAN - NINE LABEL VERSION - 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 60)

    pretrain_epochs = trainer_args.pretrain_epochs
    total_epochs = trainer_args.max_epochs
    finetune_epochs = total_epochs - pretrain_epochs

    logger.info(f"\nModel Configuration:")
    logger.info(f"  Type: {config.MODEL.TYPE}")
    logger.info(f"  Diagnosis Classes: {trainer_args.num_classes_diagnosis} (CN, MCI, AD)")
    logger.info(f"  Change Classes: {trainer_args.num_classes_change} (7 detailed transition types)")
    logger.info(f"  Number of Experts: {config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS}")

    # Log cross-validation settings
    logger.info(f"\nCross-Validation Configuration:")
    logger.info(f"  Number of Folds: {args.n_folds}")
    logger.info(f"  Majority Voting: Enabled (accumulate predictions across validation folds)")

    if pretrain_epochs > 0:
        logger.info(f"\nPhase 1 - SimMIM Pretraining (per fold):")
        logger.info(f"  Epochs: 0 - {pretrain_epochs - 1} ({pretrain_epochs} epochs)")
        logger.info(f"  Task: Self-supervised reconstruction")
        logger.info(f"  Mask ratio: {trainer_args.mask_ratio}")
        logger.info(f"  Expert assignment: Based on diagnosis labels only")

        logger.info(f"\nPhase 2 - Classification Finetuning (per fold):")
        logger.info(f"  Epochs: {pretrain_epochs} - {total_epochs - 1} ({finetune_epochs} epochs)")
        logger.info(f"  Tasks: Diagnosis (3-class) + Change (7-class) classification")
        logger.info(f"  Expert gating: Learned adaptive gating")
    else:
        logger.info(f"\nSingle Phase - Classification Training (per fold):")
        logger.info(f"  Epochs: 0 - {total_epochs - 1} ({total_epochs} epochs)")
        logger.info(f"  Tasks: Diagnosis (3-class) + Change (7-class) classification")
        logger.info(f"  Note: Skipping SimMIM pretraining")

    logger.info("\nChange Label Definitions:")
    change_labels = [
        "1: Stable CN→CN",
        "2: Stable MCI→MCI",
        "3: Stable AD→AD",
        "4: Conversion CN→MCI",
        "5: Conversion MCI→AD",
        "6: Conversion CN→AD",
        "7: Reversion MCI→CN"
    ]
    for label in change_labels:
        logger.info(f"  {label}")

    logger.info("=" * 60)

    # Log data distribution if debug mode
    if config.DEBUG.CHECK_DATA_LOADING:
        logger.info("\nChecking data loading and label distribution...")

    # Start training with 5-Fold Cross-Validation
    logger.info("Start training with 5-Fold Cross-Validation and Majority Voting")
    trainer_alzheimer_mmoe_nine_label(trainer_args, model, config.OUTPUT)


if __name__ == '__main__':
    main()