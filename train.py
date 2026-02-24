"""
Alzheimer's Disease Dual-Task Classification Training Script with SimMIM Pretraining
- Pretrain Phase: SimMIM reconstruction task
- Finetune Phase: Dual classification tasks
  - Diagnosis Task: CN(1), MCI(2), Dementia(3)
  - Change Task: Stable(1), Conversion(2), Reversion(3)
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
from config import get_config
from models import build_model
from trainer import trainer_alzheimer_mmoe
from logger import create_logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
import logging
# 忽略特定的警告信息
# 配置警告过滤
warnings.filterwarnings("ignore", message=".*Fused window process.*")
warnings.filterwarnings("ignore", message=".*Tutel.*")

# 降低某些模块的日志级别
logging.getLogger("models").setLevel(logging.ERROR)

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 或者更具体地忽略某些库的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# 如果想忽略 DeprecationWarning 也可以添加
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 忽略 UserWarning（比如 PyTorch 的一些提示）
warnings.filterwarnings("ignore", category=UserWarning)
# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('Alzheimer Dual-Task Classification Training with SimMIM Pretraining')

    # Basic settings configs/swin_admoe/swin_admoe_tiny_finetune_patch4_window16_256.yaml
    parser.add_argument('--cfg', type=str,
                        default=r'D:\codebase\Swin-Transformer\configs\swin_admoe\swin_admoe_tiny_finetune_patch4_window16_256.yaml',
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
                        help='root of output folder, the full path is <output>/<model_name>/<tag>')
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
    parser.add_argument('--wandb-project', type=str, default='alzheimer-simmim-dual-task',
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

    # Set output directory
    config.defrost()
    config.OUTPUT = args.output  # 只使用基础输出目录
    config.freeze()

    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Prepare config
    config = prepare_config(args)

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

        # Warmup设置
        warmup_epochs=getattr(config.TRAIN, 'WARMUP_EPOCHS', 5),
        warmup_lr=getattr(config.TRAIN, 'WARMUP_LR', 1e-6),

        # SimMIM预训练设置
        pretrain_epochs=getattr(config.TRAIN, 'PRETRAIN_EPOCHS', config.TRAIN.EPOCHS // 2),
        mask_ratio=getattr(config.MODEL.SIMMIM, 'MASK_RATIO', 0.6),
        patch_size=getattr(config.MODEL.SWIN_ADMOE, 'PATCH_SIZE', 4),
        norm_target=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'ENABLE', True),
        norm_target_patch_size=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'PATCH_SIZE', 47),

        # Config object
        config=config,

        # Distributed settings
        rank=rank,
        world_size=world_size,
        local_rank=args.local_rank,
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

        # 检查是否有训练阶段信息
        if 'phase' in checkpoint:
            logger.info(f"Resumed from {checkpoint['phase']} phase")

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        trainer_args.start_epoch = 0

    # Evaluation only mode
    if args.eval:
        logger.info("Evaluation mode")
        # TODO: Implement evaluation function
        raise NotImplementedError("Evaluation mode not implemented yet")

    # Log training plan
    logger.info("\n" + "="*60)
    logger.info("TRAINING PLAN")
    logger.info("="*60)

    pretrain_epochs = trainer_args.pretrain_epochs
    total_epochs = trainer_args.max_epochs
    finetune_epochs = total_epochs - pretrain_epochs

    if pretrain_epochs > 0:
        logger.info(f"Phase 1 - SimMIM Pretraining:")
        logger.info(f"  Epochs: 0 - {pretrain_epochs-1} ({pretrain_epochs} epochs)")
        logger.info(f"  Task: Self-supervised reconstruction")
        logger.info(f"  Mask ratio: {trainer_args.mask_ratio}")
        logger.info(f"  Expert assignment: Based on diagnosis labels")

        logger.info(f"\nPhase 2 - Classification Finetuning:")
        logger.info(f"  Epochs: {pretrain_epochs} - {total_epochs-1} ({finetune_epochs} epochs)")
        logger.info(f"  Tasks: Diagnosis + Change classification")
        logger.info(f"  Expert gating: Learned adaptive gating")
    else:
        logger.info(f"Single Phase - Classification Training:")
        logger.info(f"  Epochs: 0 - {total_epochs-1} ({total_epochs} epochs)")
        logger.info(f"  Tasks: Diagnosis + Change classification")
        logger.info(f"  Note: Skipping SimMIM pretraining")

    logger.info("="*60)

    # Start training
    logger.info("Start training")
    trainer_alzheimer_mmoe(trainer_args, model, config.OUTPUT)


if __name__ == '__main__':
    main()