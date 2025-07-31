# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified for Nine Label Version (7 change classes)
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .swin_transformer_v2_mtad_ptft import SwinTransformerV2_AlzheimerMMoE
from .swin_transformer_v2_mtad_ptft_nine_labels_tokenmlp import SwinTransformerV2_AlzheimerMMoE_NineLabel
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

    elif model_type == 'swin_admoe':
        # 检查是否使用新版本（带ShiftedBlock）
        use_tokenmlp_version = getattr(config.MODEL.SWIN_ADMOE, 'USE_TOKENMLP_VERSION', False)

        if use_tokenmlp_version:
            from .swin_transformer_v2_mtad_ptft_tokenmlp import SwinTransformerV2_AlzheimerMMoE_ShiftedBlock
            model = SwinTransformerV2_AlzheimerMMoE_ShiftedBlock(
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
                is_pretrain=getattr(config.MODEL.SWIN_ADMOE, 'IS_PRETRAIN', True),
                # ===== Clinical先验参数 =====
                use_clinical_prior=getattr(config.MODEL.SWIN_ADMOE, 'USE_CLINICAL_PRIOR', True),
                prior_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_DIM', 3),
                prior_hidden_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_HIDDEN_DIM', 128),
                fusion_stage=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_STAGE', 2),
                fusion_type=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_TYPE', 'adaptive'),
                # ===== ShiftedBlock相关参数 =====
                use_shifted_last_layer=getattr(config.MODEL.SWIN_ADMOE, 'USE_SHIFTED_LAST_LAYER', False),
                shift_mlp_ratio=getattr(config.MODEL.SWIN_ADMOE, 'SHIFT_MLP_RATIO', 1.0)
            )
        else:
            # 使用原版本
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
                # ===== 新增MMoE参数 (仅基于原实现) =====
                num_experts=4,  # 专家数量，原实现中有
                temperature_init=1.0,  # 温度参数，原实现中forward时使用
                ## ===== 新增Clinical先验参数 =====
                use_clinical_prior=getattr(config.MODEL.SWIN_ADMOE, 'USE_CLINICAL_PRIOR', True),
                prior_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_DIM', 3),
                prior_hidden_dim=getattr(config.MODEL.SWIN_ADMOE, 'PRIOR_HIDDEN_DIM', 128),
                fusion_stage=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_STAGE', 2),
                fusion_type=getattr(config.MODEL.SWIN_ADMOE, 'FUSION_TYPE', 'adaptive')
            )

    elif model_type == 'swin_admoe_nine_label':
        # 新增：支持7分类change label的版本
        model = SwinTransformerV2_AlzheimerMMoE_NineLabel(
            img_size=config.DATA.IMG_SIZE,
            patch_size=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PATCH_SIZE', 4),
            in_chans=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'IN_CHANS', 3),
            # 双任务类别数设置
            num_classes_diagnosis=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'NUM_CLASSES_DIAGNOSIS', 3),
            num_classes_change=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'NUM_CLASSES_CHANGE', 7),  # 7分类
            embed_dim=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'EMBED_DIM', 96),
            depths=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'DEPTHS', [2, 2, 6, 2]),
            num_heads=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'NUM_HEADS', [3, 6, 12, 24]),
            window_size=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'WINDOW_SIZE', 7),
            mlp_ratio=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'MLP_RATIO', 4.),
            qkv_bias=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'QKV_BIAS', True),
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=getattr(config.MODEL, 'ATTN_DROP_RATE', 0.),
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            ape=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'APE', False),
            patch_norm=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PATCH_NORM', True),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PRETRAINED_WINDOW_SIZES',
                                            [0, 0, 0, 0]),
            is_pretrain=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'IS_PRETRAIN', True),
            # ===== Clinical先验参数 =====
            use_clinical_prior=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'USE_CLINICAL_PRIOR', True),
            prior_dim=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PRIOR_DIM', 3),
            prior_hidden_dim=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'PRIOR_HIDDEN_DIM', 128),
            fusion_stage=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'FUSION_STAGE', 2),
            fusion_type=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'FUSION_TYPE', 'adaptive'),
            # ===== 新增：ShiftedBlock相关参数 =====
            use_shifted_last_layer=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'USE_SHIFTED_LAST_LAYER', True),
            shift_mlp_ratio=getattr(config.MODEL.SWIN_ADMOE_NINE_LABEL, 'SHIFT_MLP_RATIO', 1.0)
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model


def build_model_from_name(model_name, config=None, **kwargs):
    """根据模型名称构建模型的便捷函数"""

    # 如果是 swin_admoe，根据配置决定使用哪个版本
    if model_name == 'swin_admoe' and config is not None:
        use_tokenmlp_version = getattr(config.MODEL.SWIN_ADMOE, 'USE_TOKENMLP_VERSION', False)
        if use_tokenmlp_version:
            from .swin_transformer_v2_mtad_ptft_tokenmlp import SwinTransformerV2_AlzheimerMMoE_ShiftedBlock
            model_class = SwinTransformerV2_AlzheimerMMoE_ShiftedBlock
        else:
            model_class = SwinTransformerV2_AlzheimerMMoE
    else:
        # 模型名称到类的映射
        model_registry = {
            'swin_transformer': SwinTransformer,
            'swin_transformer_v2': SwinTransformerV2,
            'swin_transformer_moe': SwinTransformerMoE,
            'swin_mlp': SwinMLP,
            'swin_admoe': SwinTransformerV2_AlzheimerMMoE,
            'swin_admoe_nine_label': SwinTransformerV2_AlzheimerMMoE_NineLabel,
        }

        if model_name not in model_registry:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(model_registry.keys())}")

        model_class = model_registry[model_name]

    # 如果提供了config，从config中提取参数
    if config is not None:
        if model_name == 'swin_admoe_nine_label':
            # 为nine label版本设置默认参数
            default_kwargs = {
                'img_size': config.DATA.IMG_SIZE,
                'patch_size': 4,
                'in_chans': 3,
                'num_classes_diagnosis': 3,
                'num_classes_change': 7,  # 7分类
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'mlp_ratio': 4.,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'ape': False,
                'patch_norm': True,
                'use_checkpoint': False,
                'pretrained_window_sizes': [0, 0, 0, 0],
                'is_pretrain': True,
                'use_clinical_prior': True,
                'prior_dim': 3,
                'prior_hidden_dim': 128,
                'fusion_stage': 2,
                'fusion_type': 'adaptive'
            }
            # 更新默认参数
            default_kwargs.update(kwargs)
            kwargs = default_kwargs

    return model_class(**kwargs)


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_complexity(model, input_shape=(1, 3, 224, 224)):
    """获取模型复杂度信息（FLOPs和参数量）"""
    try:
        from thop import profile
        import torch

        device = next(model.parameters()).device
        input_tensor = torch.randn(input_shape).to(device)

        # 对于双输出模型，需要特殊处理
        if hasattr(model, 'num_classes_diagnosis') and hasattr(model, 'num_classes_change'):
            # 为AlzheimerMMoE模型创建额外输入
            batch_size = input_shape[0]
            clinical_prior = torch.randn(batch_size, 3).to(device)
            flops, params = profile(model, inputs=(input_tensor, clinical_prior))
        else:
            flops, params = profile(model, inputs=(input_tensor,))

        return {
            'flops': flops,
            'params': params,
            'flops_readable': f"{flops / 1e9:.2f}G",
            'params_readable': f"{params / 1e6:.2f}M"
        }
    except Exception as e:
        print(f"Error calculating model complexity: {e}")
        return None


if __name__ == "__main__":
    """测试模型构建"""
    import torch
    from types import SimpleNamespace

    print("=" * 80)
    print("Testing Model Building")
    print("=" * 80)

    # 创建一个简单的配置对象
    config = SimpleNamespace()
    config.DATA = SimpleNamespace(IMG_SIZE=256)
    config.MODEL = SimpleNamespace(
        TYPE='swin_admoe_nine_label',
        NUM_CLASSES=3,
        DROP_RATE=0.0,
        DROP_PATH_RATE=0.1,
        ATTN_DROP_RATE=0.0
    )
    config.MODEL.SWIN_ADMOE_NINE_LABEL = SimpleNamespace(
        PATCH_SIZE=4,
        IN_CHANS=3,
        NUM_CLASSES_DIAGNOSIS=3,
        NUM_CLASSES_CHANGE=7,
        EMBED_DIM=96,
        DEPTHS=[2, 2, 6, 2],
        NUM_HEADS=[3, 6, 12, 24],
        WINDOW_SIZE=7,
        MLP_RATIO=4.,
        QKV_BIAS=True,
        APE=False,
        PATCH_NORM=True,
        PRETRAINED_WINDOW_SIZES=[0, 0, 0, 0],
        IS_PRETRAIN=True,
        USE_CLINICAL_PRIOR=True,
        PRIOR_DIM=3,
        PRIOR_HIDDEN_DIM=128,
        FUSION_STAGE=2,
        FUSION_TYPE='adaptive'
    )
    config.TRAIN = SimpleNamespace(USE_CHECKPOINT=False)
    config.FUSED_LAYERNORM = False
    config.FUSED_WINDOW_PROCESS = False

    # 测试构建nine label版本
    print("\n1. Testing swin_admoe_nine_label model:")
    try:
        model = build_model(config, is_pretrain=False)
        print("✓ Model built successfully!")

        # 打印模型信息
        param_info = count_parameters(model)
        print(f"   Total parameters: {param_info['total']:,}")
        print(f"   Trainable parameters: {param_info['trainable']:,}")

        # 测试前向传播
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        test_input = torch.randn(batch_size, 3, 256, 256).to(device)
        test_prior = torch.randn(batch_size, 3).to(device)

        with torch.no_grad():
            diag_out, change_out = model(test_input, clinical_prior=test_prior)

        print(f"   Diagnosis output shape: {diag_out.shape} (expected: [{batch_size}, 3])")
        print(f"   Change output shape: {change_out.shape} (expected: [{batch_size}, 7])")

    except Exception as e:
        print(f"✗ Error: {e}")

    # 测试通过名称构建模型
    print("\n2. Testing build_model_from_name:")
    try:
        model2 = build_model_from_name(
            'swin_admoe_nine_label',
            config,
            img_size=256,
            num_classes_diagnosis=3,
            num_classes_change=7
        )
        print("✓ Model built successfully from name!")

    except Exception as e:
        print(f"✗ Error: {e}")

    # 测试原始swin_admoe模型（3分类）
    print("\n3. Testing original swin_admoe model (3 classes):")
    config.MODEL.TYPE = 'swin_admoe'
    config.MODEL.SWIN_ADMOE = SimpleNamespace(
        PATCH_SIZE=4,
        IN_CHANS=3,
        NUM_CLASSES_DIAGNOSIS=3,
        NUM_CLASSES_CHANGE=3,  # 原始3分类
        EMBED_DIM=96,
        DEPTHS=[2, 2, 6, 2],
        NUM_HEADS=[3, 6, 12, 24],
        WINDOW_SIZE=7,
        MLP_RATIO=4.,
        QKV_BIAS=True,
        APE=False,
        PATCH_NORM=True,
        PRETRAINED_WINDOW_SIZES=[0, 0, 0, 0],
        IS_PRETRAIN=True,
        USE_CLINICAL_PRIOR=True,
        PRIOR_DIM=3,
        PRIOR_HIDDEN_DIM=128,
        FUSION_STAGE=2,
        FUSION_TYPE='adaptive'
    )

    try:
        model3 = build_model(config, is_pretrain=False)
        print("✓ Original model built successfully!")

        with torch.no_grad():
            diag_out, change_out = model3(test_input, clinical_prior=test_prior)

        print(f"   Diagnosis output shape: {diag_out.shape} (expected: [{batch_size}, 3])")
        print(f"   Change output shape: {change_out.shape} (expected: [{batch_size}, 3])")

    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 80)
    print("Model building tests completed!")
    print("=" * 80)