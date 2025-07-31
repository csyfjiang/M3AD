"""
Description: 
Author: JeffreyJ
Date: 2025/7/5
LastEditTime: 2025/7/5 14:38
Version: 1.0
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified for Single Task Support
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .swin_transformer_v2_mtad_ptft import SwinTransformerV2_AlzheimerMMoE
from .swin_transformer_v2_mtad_ptft_single_task import SwinTransformerV2_SingleTask
from .simmim import build_simmim


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                   patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                                   in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                                   num_classes=config.MODEL.NUM_CLASSES,
                                   embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                                   depths=config.MODEL.SWIN_MOE.DEPTHS,
                                   num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                                   window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                                   mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                                   qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                                   qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                                   drop_rate=config.MODEL.DROP_RATE,
                                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                   ape=config.MODEL.SWIN_MOE.APE,
                                   patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                                   mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                                   init_std=config.MODEL.SWIN_MOE.INIT_STD,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                   pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                   moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                                   num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                                   capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                                   normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                                   is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                                   gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                                   cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                   moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                                   aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_single_task':
        # ============================================================================
        # Single Task Swin Transformer with MoE
        # ============================================================================

        # 获取配置参数
        swin_config = getattr(config.MODEL, 'SWIN_SINGLE_TASK', config.MODEL.SWIN_ADMOE)

        # 基础参数
        img_size = config.DATA.IMG_SIZE
        patch_size = getattr(swin_config, 'PATCH_SIZE', 4)
        in_chans = getattr(swin_config, 'IN_CHANS', 3)
        num_classes = config.MODEL.NUM_CLASSES
        embed_dim = getattr(swin_config, 'EMBED_DIM', 96)
        depths = getattr(swin_config, 'DEPTHS', [2, 2, 6, 2])
        num_heads = getattr(swin_config, 'NUM_HEADS', [3, 6, 12, 24])
        window_size = getattr(swin_config, 'WINDOW_SIZE', 8)  # 注意默认值改为 8
        mlp_ratio = getattr(swin_config, 'MLP_RATIO', 4.)
        qkv_bias = getattr(swin_config, 'QKV_BIAS', True)
        drop_rate = config.MODEL.DROP_RATE
        attn_drop_rate = getattr(config.MODEL, 'ATTN_DROP_RATE', 0.)
        drop_path_rate = config.MODEL.DROP_PATH_RATE
        ape = getattr(swin_config, 'APE', False)
        patch_norm = getattr(swin_config, 'PATCH_NORM', True)
        use_checkpoint = config.TRAIN.USE_CHECKPOINT
        pretrained_window_sizes = getattr(swin_config, 'PRETRAINED_WINDOW_SIZES', [0, 0, 0, 0])
        is_pretrain = getattr(swin_config, 'IS_PRETRAIN', True)

        # 临床先验参数
        use_clinical_prior = getattr(swin_config, 'USE_CLINICAL_PRIOR', True)
        prior_dim = getattr(swin_config, 'PRIOR_DIM', 3)
        prior_hidden_dim = getattr(swin_config, 'PRIOR_HIDDEN_DIM', 128)
        fusion_stage = getattr(swin_config, 'FUSION_STAGE', 2)
        fusion_type = getattr(swin_config, 'FUSION_TYPE', 'adaptive')

        # 检查benchmark模式
        if hasattr(config, 'SINGLE_TASK') and hasattr(config.SINGLE_TASK, 'BENCHMARK'):
            if getattr(config.SINGLE_TASK.BENCHMARK, 'ENABLED', False):
                use_clinical_prior = False  # Benchmark模式下禁用临床先验

        # 创建单任务模型
        model = SwinTransformerV2_SingleTask(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=layernorm,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
            is_pretrain=is_pretrain,
            # 临床先验参数
            use_clinical_prior=use_clinical_prior,
            prior_dim=prior_dim,
            prior_hidden_dim=prior_hidden_dim,
            fusion_stage=fusion_stage,
            fusion_type=fusion_type
        )

    elif model_type == 'swin_admoe':
        # ============================================================================
        # Dual Task Swin Transformer with MMoE (Original)
        # ============================================================================
        model = SwinTransformerV2_AlzheimerMMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=getattr(config.MODEL.SWIN_ADMOE, 'PATCH_SIZE', 4),
            in_chans=getattr(config.MODEL.SWIN_ADMOE, 'IN_CHANS', 3),
            # 双任务类别数设置
            num_classes=config.MODEL.NUM_CLASSES,  # 保留兼容性
            num_classes_diagnosis=getattr(config.MODEL.SWIN_ADMOE, 'NUM_CLASSES_DIAGNOSIS', 3),
            num_classes_change=getattr(config.MODEL.SWIN_ADMOE, 'NUM_CLASSES_CHANGE', 3),
            embed_dim=getattr(config.MODEL.SWIN_ADMOE, 'EMBED_DIM', 96),
            depths=getattr(config.MODEL.SWIN_ADMOE, 'DEPTHS', [2, 2, 6, 2]),
            num_heads=getattr(config.MODEL.SWIN_ADMOE, 'NUM_HEADS', [3, 6, 12, 24]),
            window_size=getattr(config.MODEL.SWIN_ADMOE, 'WINDOW_SIZE', 7),
            mlp_ratio=getattr(config.MODEL.SWIN_ADMOE, 'MLP_RATIO', 4.),
            qkv_bias=getattr(config.MODEL.SWIN_ADMOE, 'QKV_BIAS', True),
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=getattr(config.MODEL, 'ATTN_DROP_RATE', 0.),
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            ape=getattr(config.MODEL.SWIN_ADMOE, 'APE', False),
            patch_norm=getattr(config.MODEL.SWIN_ADMOE, 'PATCH_NORM', True),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=getattr(config.MODEL.SWIN_ADMOE, 'PRETRAINED_WINDOW_SIZES', [0, 0, 0, 0]),
            shift_mlp_ratio=getattr(config.MODEL.SWIN_ADMOE, 'SHIFT_MLP_RATIO', 1.0),
            is_pretrain=getattr(config.MODEL.SWIN_ADMOE, 'IS_PRETRAIN', True),
            use_shifted_last_layer=getattr(config.MODEL.SWIN_ADMOE, 'USE_SHIFTED_LAST_LAYER', False),
            # 临床先验参数
            use_clinical_prior=getattr(config.MODEL.SWIN_ADMOE, 'USE_CLINICAL_PRIOR', True),
            prior_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_DIM', 3),
            prior_hidden_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_HIDDEN_DIM', 128),
            fusion_stage=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_STAGE', 2),
            fusion_type=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_TYPE', 'adaptive')
        )

    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model


def print_model_info(model, config):
    """Print detailed model information"""
    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)

    # 基础信息
    print(f"Model Type: {config.MODEL.TYPE}")
    print(f"Model Name: {config.MODEL.NAME}")
    print(f"Number of Classes: {config.MODEL.NUM_CLASSES}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 单任务特定信息
    if hasattr(config, 'SINGLE_TASK') and config.SINGLE_TASK.ENABLED:
        print(f"\nSingle Task Configuration:")
        print(f"  Task Type: {config.SINGLE_TASK.TASK_TYPE}")
        print(f"  Class Names: {config.SINGLE_TASK.CLASS_NAMES}")

        # 专家配置信息
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_block = model.layers[0].blocks[0]
            if hasattr(first_block, 'mmoe'):
                print(f"  Expert Configuration:")
                print(f"    Number of Experts: {first_block.mmoe.num_experts}")
                print(f"    Expert Names: {first_block.mmoe.expert_names}")

        # 临床先验信息
        if hasattr(model, 'use_clinical_prior'):
            print(f"  Clinical Prior: {'Enabled' if model.use_clinical_prior else 'Disabled'}")
            if model.use_clinical_prior:
                print(f"    Fusion Stage: {model.fusion_stage}")
                print(
                    f"    Fusion Type: {model.clinical_fusion.fusion_type if hasattr(model, 'clinical_fusion') else 'N/A'}")

    # Benchmark模式信息
    if hasattr(config, 'SINGLE_TASK') and hasattr(config.SINGLE_TASK, 'BENCHMARK'):
        if config.SINGLE_TASK.BENCHMARK.ENABLED:
            print(f"\nBenchmark Mode: ENABLED")
            print(f"  Dataset: {config.SINGLE_TASK.BENCHMARK.DATASET}")
            print(f"  Metrics: {config.SINGLE_TASK.BENCHMARK.METRICS}")

    print("=" * 80 + "\n")


def build_model_with_info(config, is_pretrain=False, verbose=True):
    """Build model and optionally print detailed information"""
    model = build_model(config, is_pretrain)

    if verbose:
        print_model_info(model, config)

    return model


if __name__ == "__main__":
    """Test model building"""
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config_single_task import get_single_task_config

    print("=" * 80)
    print("Testing Model Building")
    print("=" * 80)

    # Test binary classification model
    print("\n1. Binary Classification Model:")
    binary_config = get_single_task_config(task_type='binary', num_classes=2)
    binary_model = build_model_with_info(binary_config, verbose=True)

    # Test three-class model
    print("\n2. Three-class Classification Model:")
    three_config = get_single_task_config(task_type='diagnosis', num_classes=3)
    three_model = build_model_with_info(three_config, verbose=True)

    # Test benchmark mode
    print("\n3. Benchmark Mode Model:")
    benchmark_config = get_single_task_config(task_type='binary', num_classes=2)
    benchmark_config.defrost()
    benchmark_config.SINGLE_TASK.BENCHMARK.ENABLED = True
    benchmark_config.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR = False
    benchmark_config.freeze()
    benchmark_model = build_model_with_info(benchmark_config, verbose=True)

    print("Model building tests completed successfully!")