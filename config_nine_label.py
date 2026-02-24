import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
_C.DATA.BATCH_SIZE_PRETRAIN = 256  # 预训练阶段batch size
_C.DATA.BATCH_SIZE_FINETUNE = 224  # 微调阶段batch size
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = 0
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = 0
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
# Mask ratio for SimMIM pretraining
_C.MODEL.SIMMIM.MASK_RATIO = 0.6
# Normalization target settings
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# Swin Transformer with Alzheimer MoE parameters (Original 3-class version)
_C.MODEL.SWIN_ADMOE = CN()
_C.MODEL.SWIN_ADMOE.PATCH_SIZE = 4
_C.MODEL.SWIN_ADMOE.IN_CHANS = 3
_C.MODEL.SWIN_ADMOE.EMBED_DIM = 96
_C.MODEL.SWIN_ADMOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_ADMOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_ADMOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_ADMOE.MLP_RATIO = 4.
_C.MODEL.SWIN_ADMOE.QKV_BIAS = True
_C.MODEL.SWIN_ADMOE.APE = False
_C.MODEL.SWIN_ADMOE.PATCH_NORM = True
_C.MODEL.SWIN_ADMOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_ADMOE.SHIFT_MLP_RATIO = 1.0
_C.MODEL.SWIN_ADMOE.IS_PRETRAIN = True
_C.MODEL.SWIN_ADMOE.USE_SHIFTED_LAST_LAYER = False
# Dual-task specific parameters
_C.MODEL.SWIN_ADMOE.NUM_CLASSES_DIAGNOSIS = 3
_C.MODEL.SWIN_ADMOE.NUM_CLASSES_CHANGE = 3
# Clinical prior parameters
_C.MODEL.SWIN_ADMOE.USE_CLINICAL_PRIOR = True
_C.MODEL.SWIN_ADMOE.PRIOR_DIM = 3
_C.MODEL.SWIN_ADMOE.PRIOR_HIDDEN_DIM = 128
_C.MODEL.SWIN_ADMOE.FUSION_STAGE = 2
_C.MODEL.SWIN_ADMOE.FUSION_TYPE = 'adaptive'

# Swin Transformer with Alzheimer MoE parameters (Nine Label Version - 7 change classes)
_C.MODEL.SWIN_ADMOE_NINE_LABEL = CN()
_C.MODEL.SWIN_ADMOE_NINE_LABEL.PATCH_SIZE = 4
_C.MODEL.SWIN_ADMOE_NINE_LABEL.IN_CHANS = 3
_C.MODEL.SWIN_ADMOE_NINE_LABEL.EMBED_DIM = 96
_C.MODEL.SWIN_ADMOE_NINE_LABEL.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_ADMOE_NINE_LABEL.WINDOW_SIZE = 7
_C.MODEL.SWIN_ADMOE_NINE_LABEL.MLP_RATIO = 4.
_C.MODEL.SWIN_ADMOE_NINE_LABEL.QKV_BIAS = True
_C.MODEL.SWIN_ADMOE_NINE_LABEL.APE = False
_C.MODEL.SWIN_ADMOE_NINE_LABEL.PATCH_NORM = True
_C.MODEL.SWIN_ADMOE_NINE_LABEL.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_ADMOE_NINE_LABEL.IS_PRETRAIN = True
# Dual-task specific parameters - 7分类版本
_C.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_DIAGNOSIS = 3
_C.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE = 7  # 7个细化的change类别
# Clinical prior parameters
_C.MODEL.SWIN_ADMOE_NINE_LABEL.USE_CLINICAL_PRIOR = True
_C.MODEL.SWIN_ADMOE_NINE_LABEL.PRIOR_DIM = 3
_C.MODEL.SWIN_ADMOE_NINE_LABEL.PRIOR_HIDDEN_DIM = 128
_C.MODEL.SWIN_ADMOE_NINE_LABEL.FUSION_STAGE = 2
_C.MODEL.SWIN_ADMOE_NINE_LABEL.FUSION_TYPE = 'adaptive'
# Expert configuration - 8专家系统
_C.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS = 8
_C.MODEL.SWIN_ADMOE_NINE_LABEL.EXPERT_ASSIGNMENT_PRETRAIN = 'diagnosis_based'
_C.MODEL.SWIN_ADMOE_NINE_LABEL.EXPERT_ASSIGNMENT_FINETUNE = 'learned'
# ShiftedBlock configuration - 新增
_C.MODEL.SWIN_ADMOE_NINE_LABEL.USE_SHIFTED_LAST_LAYER = False
_C.MODEL.SWIN_ADMOE_NINE_LABEL.SHIFT_MLP_RATIO = 1.0
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Pretraining epochs for SimMIM
_C.TRAIN.PRETRAIN_EPOCHS = 100
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = (0.0, 1.0)
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Loss settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.WEIGHT_DIAGNOSIS = 1.0
_C.LOSS.WEIGHT_CHANGE = 1.0

# -----------------------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.INTERVAL = 10
_C.EVAL.SAVE_BEST = True
_C.EVAL.METRICS = ['accuracy', 'f1_score', 'confusion_matrix']
# 评估时的详细报告
_C.EVAL.DETAILED_REPORT = False
_C.EVAL.SAVE_PREDICTIONS = False

# -----------------------------------------------------------------------------
# Early stopping settings
# -----------------------------------------------------------------------------
_C.EARLY_STOP = CN()
_C.EARLY_STOP.ENABLE = True
_C.EARLY_STOP.PATIENCE = 5
_C.EARLY_STOP.MONITOR = 'val_loss'
_C.EARLY_STOP.DELTA = 0.0

# -----------------------------------------------------------------------------
# WandB settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.PROJECT = 'alzheimer-classification'
_C.WANDB.NAME = 'experiment'
_C.WANDB.TAGS = []

# -----------------------------------------------------------------------------
# Phase-specific configurations
# -----------------------------------------------------------------------------
_C.PHASES = CN()
_C.PHASES.PRETRAIN = CN()
_C.PHASES.PRETRAIN.DESCRIPTION = "SimMIM self-supervised pretraining"
_C.PHASES.PRETRAIN.LOSS_TYPE = "reconstruction"
_C.PHASES.PRETRAIN.EXPERT_ASSIGNMENT = "label_based"
_C.PHASES.PRETRAIN.EVALUATION_METRIC = "reconstruction_loss"
_C.PHASES.PRETRAIN.MASK_RATIO_SCHEDULE = []

_C.PHASES.FINETUNE = CN()
_C.PHASES.FINETUNE.DESCRIPTION = "Dual-task classification finetuning"
_C.PHASES.FINETUNE.LOSS_TYPE = "classification"
_C.PHASES.FINETUNE.EXPERT_ASSIGNMENT = "learned_gating"
_C.PHASES.FINETUNE.EVALUATION_METRIC = "classification_accuracy"
_C.PHASES.FINETUNE.CLASS_WEIGHTS_DIAGNOSIS = []
_C.PHASES.FINETUNE.CLASS_WEIGHTS_CHANGE = []

# -----------------------------------------------------------------------------
# Expert Configuration Details (for 8-expert system)
# -----------------------------------------------------------------------------
_C.EXPERTS = CN()
_C.EXPERTS.DESCRIPTION = "Expert system configuration"
_C.EXPERTS.ALLOCATION = CN()
_C.EXPERTS.ALLOCATION.SHARED = []
_C.EXPERTS.ALLOCATION.CN = []
_C.EXPERTS.ALLOCATION.MCI = []
_C.EXPERTS.ALLOCATION.AD = []
_C.EXPERTS.ROUTING_TEMPERATURE = CN()
_C.EXPERTS.ROUTING_TEMPERATURE.PRETRAIN = 1.0
_C.EXPERTS.ROUTING_TEMPERATURE.FINETUNE = 0.5

# -----------------------------------------------------------------------------
# Data Loading Configuration
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.USE_WEIGHTED_SAMPLING = False
_C.DATALOADER.PREFETCH_FACTOR = 2
_C.DATALOADER.PERSISTENT_WORKERS = True

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
_C.LOGGING.LOG_EXPERT_UTILIZATION_FREQ = 5
_C.LOGGING.SAVE_ATTENTION_MAPS = False
_C.LOGGING.LOG_GRADIENT_STATS = False

# -----------------------------------------------------------------------------
# Debugging Options
# -----------------------------------------------------------------------------
_C.DEBUG = CN()
_C.DEBUG.USE_SMALL_DATASET = False
_C.DEBUG.SMALL_DATASET_SIZE = 100
_C.DEBUG.PRINT_MODEL_DETAILS = True
_C.DEBUG.CHECK_DATA_LOADING = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # SimMIM specific argument overrides
    if hasattr(args, 'mask_ratio') and args.mask_ratio is not None:
        config.MODEL.SIMMIM.MASK_RATIO = args.mask_ratio
    if hasattr(args, 'norm_target') and args.norm_target is not None:
        config.MODEL.SIMMIM.NORM_TARGET.ENABLE = args.norm_target
    if hasattr(args, 'norm_target_patch_size') and args.norm_target_patch_size is not None:
        config.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = args.norm_target_patch_size
    if hasattr(args, 'pretrain_epochs') and args.pretrain_epochs is not None:
        config.TRAIN.PRETRAIN_EPOCHS = args.pretrain_epochs

    # Nine label specific overrides
    if hasattr(args, 'num_change_classes') and args.num_change_classes is not None:
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE = args.num_change_classes

    if hasattr(args, 'num_experts') and args.num_experts is not None:
        if config.MODEL.TYPE == 'swin_admoe_nine_label':
            config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS = args.num_experts

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        # For PyTorch 2.x, safely get LOCAL_RANK with fallback
        config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def validate_nine_label_config(config):
    """验证九标签配置的合理性"""
    if config.MODEL.TYPE == 'swin_admoe_nine_label':
        # 检查类别数量
        assert config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_DIAGNOSIS == 3, \
            "Diagnosis classes must be 3 (CN, MCI, AD)"
        assert config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE == 7, \
            "Change classes must be 7 for nine label version"
        assert config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS == 8, \
            "Number of experts must be 8 for nine label version"

        # 检查专家分配
        if hasattr(config.EXPERTS.ALLOCATION, 'SHARED'):
            assert len(config.EXPERTS.ALLOCATION.SHARED) == 2, "Must have 2 shared experts"
            assert len(config.EXPERTS.ALLOCATION.CN) == 2, "Must have 2 CN experts"
            assert len(config.EXPERTS.ALLOCATION.MCI) == 2, "Must have 2 MCI experts"
            assert len(config.EXPERTS.ALLOCATION.AD) == 2, "Must have 2 AD experts"

        print("Nine label configuration validated successfully!")
        return True
    return False


def print_config_summary(config):
    """打印配置摘要"""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)

    print(f"\nModel Type: {config.MODEL.TYPE}")
    print(f"Model Name: {config.MODEL.NAME}")

    if config.MODEL.TYPE == 'swin_admoe_nine_label':
        print("\nNine Label Specific Settings:")
        print(f"  Diagnosis Classes: {config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_DIAGNOSIS}")
        print(f"  Change Classes: {config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_CLASSES_CHANGE}")
        print(f"  Number of Experts: {config.MODEL.SWIN_ADMOE_NINE_LABEL.NUM_EXPERTS}")
        print(f"  Clinical Prior: {config.MODEL.SWIN_ADMOE_NINE_LABEL.USE_CLINICAL_PRIOR}")
        print(f"  Fusion Type: {config.MODEL.SWIN_ADMOE_NINE_LABEL.FUSION_TYPE}")

    print(f"\nTraining Settings:")
    print(f"  Total Epochs: {config.TRAIN.EPOCHS}")
    print(f"  Pretrain Epochs: {config.TRAIN.PRETRAIN_EPOCHS}")
    print(f"  Base LR: {config.TRAIN.BASE_LR}")
    print(f"  Batch Size: {config.DATA.BATCH_SIZE}")

    print(f"\nData Settings:")
    print(f"  Dataset: {config.DATA.DATASET}")
    print(f"  Image Size: {config.DATA.IMG_SIZE}")
    print(f"  Data Path: {config.DATA.DATA_PATH}")

    print("=" * 80 + "\n")