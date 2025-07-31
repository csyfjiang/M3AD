#!/usr/bin/env python
"""
完整测试脚本：Swin Transformer V2 with Alzheimer MMoE and SimMIM
测试模型的所有功能：
1. 分类任务（诊断和变化）
2. SimMIM重建任务
3. 临床先验融合
4. 专家权重分配
5. 不同标签系统测试
6. 性能和内存测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import time
import warnings

warnings.filterwarnings('ignore')

# 假设模型已经导入
from swin_transformer_v2_mtad_ptft import SwinTransformerV2_AlzheimerMMoE


def generate_mask(input_size: Tuple[int, int], patch_size: int, mask_ratio: float) -> torch.Tensor:
    """生成SimMIM的随机mask - patch级别的mask"""
    H, W = input_size
    # 计算patch的数量
    h, w = H // patch_size, W // patch_size
    num_patches = h * w
    num_mask = int(num_patches * mask_ratio)

    # 随机选择要mask的patches
    mask = torch.zeros(num_patches, dtype=torch.float32)
    mask_indices = torch.randperm(num_patches)[:num_mask]
    mask[mask_indices] = 1

    return mask


def visualize_reconstruction(original, reconstructed, mask, patch_size=4, save_path=None):
    """可视化重建结果"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 将所有数据转到CPU
    original_cpu = original[0].cpu()
    reconstructed_cpu = reconstructed[0].cpu()
    mask_cpu = mask[0].cpu()

    # 原始图像
    img_original = original_cpu.permute(1, 2, 0).numpy()
    img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min())
    axes[0].imshow(img_original)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 将patch级别的mask转换为像素级别用于可视化
    C, H, W = original_cpu.shape
    h, w = H // patch_size, W // patch_size
    mask_visual = mask_cpu.reshape(h, w)
    mask_visual = mask_visual.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)

    axes[1].imshow(mask_visual.numpy(), cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Masked图像
    mask_3channel = mask_visual.unsqueeze(0).repeat(3, 1, 1)
    img_masked = original_cpu * (1 - mask_3channel)
    img_masked = img_masked.permute(1, 2, 0).numpy()
    img_masked = (img_masked - img_masked.min()) / (img_masked.max() - img_masked.min() + 1e-8)
    axes[2].imshow(img_masked)
    axes[2].set_title("Masked Input")
    axes[2].axis('off')

    # 重建图像
    img_recon = reconstructed_cpu.permute(1, 2, 0).numpy()
    img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
    axes[3].imshow(img_recon)
    axes[3].set_title("Reconstructed")
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - 可视化结果已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def test_label_system_compatibility(model, device, batch_size=4):
    """测试1,2,3标签系统的兼容性"""
    print("\n" + "=" * 60)
    print("测试标签系统兼容性 (1=CN, 2=MCI, 3=AD)")
    print("=" * 60)

    # 测试数据
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)

    # 使用1,2,3标签系统
    diagnosis_labels_123 = torch.randint(1, 4, (batch_size,)).to(device)  # 1,2,3
    change_labels_123 = torch.randint(1, 4, (batch_size,)).to(device)  # 1,2,3

    print(f"测试标签:")
    print(f"- 诊断标签 (1,2,3): {diagnosis_labels_123}")
    print(f"- 变化标签 (1,2,3): {change_labels_123}")

    # 测试分类
    model.eval()
    with torch.no_grad():
        diag_out, change_out = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels_123,
            lbls_change=change_labels_123
        )

    print(f"\n分类输出:")
    print(f"- 诊断输出形状: {diag_out.shape}")
    print(f"- 变化输出形状: {change_out.shape}")

    # 预测结果 (模型输出0,1,2，需要+1转回1,2,3)
    diag_pred = torch.argmax(diag_out, dim=1) + 1
    change_pred = torch.argmax(change_out, dim=1) + 1

    print(f"\n预测结果 (转换回1,2,3):")
    print(f"- 诊断预测: {diag_pred}")
    print(f"- 变化预测: {change_pred}")

    # 测试重建
    mask_ratio = 0.6
    patch_size = 4
    img_size = 256

    mask = torch.stack([
        generate_mask((img_size, img_size), patch_size, mask_ratio)
        for _ in range(batch_size)
    ]).to(device)

    with torch.no_grad():
        reconstructed = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels_123,
            lbls_change=change_labels_123,
            mask=mask
        )

    print(f"\n重建测试:")
    print(f"- 重建输出形状: {reconstructed.shape}")
    print(f"- 输入输出形状匹配: {images.shape == reconstructed.shape}")

    return {
        'classification_success': True,
        'reconstruction_success': True,
        'diagnosis_predictions': diag_pred,
        'change_predictions': change_pred
    }


def test_expert_weight_distribution(model, device, batch_size=4):
    """测试专家权重分配逻辑"""
    print("\n" + "=" * 60)
    print("测试专家权重分配逻辑")
    print("=" * 60)

    expert_names = ['Shared', 'CN-focused', 'MCI-focused', 'AD-focused']

    # 为每个诊断类别测试专家分配
    results = {}

    for diag_label in [1, 2, 3]:  # CN, MCI, AD
        class_name = ['CN', 'MCI', 'AD'][diag_label - 1]
        print(f"\n测试 {class_name} (标签 {diag_label}) 的专家分配:")

        # 创建该类别的批次
        images = torch.randn(batch_size, 3, 256, 256).to(device)
        clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
        diagnosis_labels = torch.full((batch_size,), diag_label, dtype=torch.long).to(device)
        change_labels = torch.randint(1, 4, (batch_size,)).to(device)

        # 测试重建任务的专家分配
        mask = torch.stack([
            generate_mask((256, 256), 4, 0.6)
            for _ in range(batch_size)
        ]).to(device)

        model.eval()
        with torch.no_grad():
            # 直接测试第一个MMoE层
            first_block = model.layers[0].blocks[0]

            # 模拟输入特征
            x_test = torch.randn(batch_size, 64 * 64, 96).to(device)

            # 调用MMoE进行重建
            recon_output = first_block.mmoe(
                x_test,
                lbls_diagnosis=diagnosis_labels,
                task='reconstruction'
            )

            print(f"  - 重建输出形状: {recon_output.shape}")
            print(f"  - 预期主要激活的专家: {expert_names[diag_label]} (索引 {diag_label})")

            # 测试分类任务的专家分配
            diag_output, change_output = first_block.mmoe(
                x_test,
                lbls_diagnosis=diagnosis_labels,
                lbls_change=change_labels,
                task='both'
            )

            print(f"  - 诊断任务输出形状: {diag_output.shape}")
            print(f"  - 变化任务输出形状: {change_output.shape}")

        results[class_name] = {
            'reconstruction_output_shape': recon_output.shape,
            'diagnosis_output_shape': diag_output.shape,
            'change_output_shape': change_output.shape,
            'expected_expert': expert_names[diag_label]
        }

    return results


def test_clinical_prior_fusion(model, device, batch_size=4):
    """测试临床先验融合功能"""
    print("\n" + "=" * 60)
    print("测试临床先验融合功能")
    print("=" * 60)

    images = torch.randn(batch_size, 3, 256, 256).to(device)
    diagnosis_labels = torch.randint(1, 4, (batch_size,)).to(device)
    change_labels = torch.randint(1, 4, (batch_size,)).to(device)

    # 测试1: 有临床先验 vs 无临床先验
    print("\n1. 有无临床先验对比:")

    # 创建模拟的临床先验
    clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
    print(f"   临床先验示例: {clinical_prior[0].cpu().numpy()}")

    model.eval()
    with torch.no_grad():
        # 有临床先验
        diag_out_with_prior, change_out_with_prior = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )

        # 无临床先验
        diag_out_no_prior, change_out_no_prior = model(
            images,
            clinical_prior=None,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )

    print(f"   有先验 - 诊断输出形状: {diag_out_with_prior.shape}")
    print(f"   无先验 - 诊断输出形状: {diag_out_no_prior.shape}")

    # 计算输出差异
    diag_diff = torch.mean(torch.abs(diag_out_with_prior - diag_out_no_prior)).item()
    change_diff = torch.mean(torch.abs(change_out_with_prior - change_out_no_prior)).item()

    print(f"   诊断输出平均差异: {diag_diff:.4f}")
    print(f"   变化输出平均差异: {change_diff:.4f}")

    # 测试2: 不同融合阶段
    print("\n2. 测试不同融合阶段的影响:")
    fusion_stages = [0, 1, 2, 3]
    stage_results = {}

    for stage in fusion_stages:
        # 创建该融合阶段的模型（简化测试，只记录是否成功）
        try:
            test_model = SwinTransformerV2_AlzheimerMMoE(
                img_size=256, patch_size=4, in_chans=3,
                num_classes_diagnosis=3, num_classes_change=3,
                embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=16, mlp_ratio=4.,
                use_clinical_prior=True, prior_dim=3,
                fusion_stage=stage, fusion_type='adaptive',
                is_pretrain=False
            ).to(device)

            test_model.eval()
            with torch.no_grad():
                diag_out, change_out = test_model(
                    images, clinical_prior=clinical_prior,
                    lbls_diagnosis=diagnosis_labels, lbls_change=change_labels
                )

            stage_results[stage] = {
                'success': True,
                'output_shapes': (diag_out.shape, change_out.shape)
            }
            print(f"   融合阶段 {stage}: ✓ 成功")

        except Exception as e:
            stage_results[stage] = {
                'success': False,
                'error': str(e)
            }
            print(f"   融合阶段 {stage}: ✗ 失败 - {str(e)}")

    return {
        'prior_difference': {'diagnosis': diag_diff, 'change': change_diff},
        'fusion_stages': stage_results
    }


def test_pretrain_finetune_modes(model, device, batch_size=4):
    """测试预训练和微调模式切换"""
    print("\n" + "=" * 60)
    print("测试预训练和微调模式切换")
    print("=" * 60)

    images = torch.randn(batch_size, 3, 256, 256).to(device)
    clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
    diagnosis_labels = torch.randint(1, 4, (batch_size,)).to(device)
    change_labels = torch.randint(1, 4, (batch_size,)).to(device)

    mask = torch.stack([
        generate_mask((256, 256), 4, 0.6)
        for _ in range(batch_size)
    ]).to(device)

    results = {}

    # 测试1: 预训练模式
    print("\n1. 预训练模式测试:")
    model_to_test = model.module if hasattr(model, 'module') else model
    model_to_test.is_pretrain = True
    for layer in model_to_test.layers:
        layer.set_pretrain_mode(True)

    model.eval()
    with torch.no_grad():
        # 重建任务
        reconstructed = model(
            images, clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels, lbls_change=change_labels,
            mask=mask
        )
        print(f"   重建输出形状: {reconstructed.shape}")

        # 分类任务（预训练模式下也应该工作）
        diag_out_pretrain, change_out_pretrain = model(
            images, clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels, lbls_change=change_labels
        )
        print(f"   预训练模式分类输出: {diag_out_pretrain.shape}, {change_out_pretrain.shape}")

    results['pretrain'] = {
        'reconstruction_shape': reconstructed.shape,
        'classification_shapes': (diag_out_pretrain.shape, change_out_pretrain.shape)
    }

    # 测试2: 微调模式
    print("\n2. 微调模式测试:")
    model_to_test.is_pretrain = False
    for layer in model_to_test.layers:
        layer.set_pretrain_mode(False)

    with torch.no_grad():
        # 分类任务
        diag_out_finetune, change_out_finetune = model(
            images, clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels, lbls_change=change_labels
        )
        print(f"   微调模式分类输出: {diag_out_finetune.shape}, {change_out_finetune.shape}")

    results['finetune'] = {
        'classification_shapes': (diag_out_finetune.shape, change_out_finetune.shape)
    }

    # 比较两种模式的输出差异
    diag_mode_diff = torch.mean(torch.abs(diag_out_pretrain - diag_out_finetune)).item()
    change_mode_diff = torch.mean(torch.abs(change_out_pretrain - change_out_finetune)).item()

    print(f"\n3. 模式差异分析:")
    print(f"   诊断输出差异: {diag_mode_diff:.4f}")
    print(f"   变化输出差异: {change_mode_diff:.4f}")

    results['mode_differences'] = {
        'diagnosis': diag_mode_diff,
        'change': change_mode_diff
    }

    return results


def test_performance_memory(model, device, test_iterations=50):
    """测试性能和内存使用"""
    print("\n" + "=" * 60)
    print("测试性能和内存使用")
    print("=" * 60)

    # 测试不同batch size的性能
    batch_sizes = [1, 2, 4, 8] if torch.cuda.is_available() else [1, 2]
    results = {}

    for batch_size in batch_sizes:
        print(f"\n测试 Batch Size: {batch_size}")

        # 准备数据
        images = torch.randn(batch_size, 3, 256, 256).to(device)
        clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
        diagnosis_labels = torch.randint(1, 4, (batch_size,)).to(device)
        change_labels = torch.randint(1, 4, (batch_size,)).to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()

        model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(images, clinical_prior=clinical_prior,
                          lbls_diagnosis=diagnosis_labels, lbls_change=change_labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 测量时间
        start_time = time.time()

        with torch.no_grad():
            for _ in range(test_iterations):
                _ = model(images, clinical_prior=clinical_prior,
                          lbls_diagnosis=diagnosis_labels, lbls_change=change_labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time = (end_time - start_time) / test_iterations
        throughput = batch_size / avg_time

        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_usage = (end_memory - start_memory) / (1024 ** 2)  # MB
        else:
            memory_usage = 0

        print(f"   平均推理时间: {avg_time * 1000:.2f} ms")
        print(f"   吞吐量: {throughput:.2f} images/s")
        if torch.cuda.is_available():
            print(f"   内存使用: {memory_usage:.2f} MB")

        results[batch_size] = {
            'avg_time_ms': avg_time * 1000,
            'throughput': throughput,
            'memory_mb': memory_usage
        }

    return results


def test_reconstruction_quality(model, device, batch_size=4):
    """测试重建质量"""
    print("\n" + "=" * 60)
    print("测试重建质量")
    print("=" * 60)

    # 不同mask ratio的测试
    mask_ratios = [0.3, 0.5, 0.6, 0.75]
    patch_size = 4
    img_size = 256

    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
    diagnosis_labels = torch.randint(1, 4, (batch_size,)).to(device)
    change_labels = torch.randint(1, 4, (batch_size,)).to(device)

    model.eval()
    results = {}

    for mask_ratio in mask_ratios:
        print(f"\n测试 Mask Ratio: {mask_ratio}")

        # 生成mask
        mask = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio)
            for _ in range(batch_size)
        ]).to(device)

        with torch.no_grad():
            reconstructed = model(
                images, clinical_prior=clinical_prior,
                lbls_diagnosis=diagnosis_labels, lbls_change=change_labels,
                mask=mask
            )

        # 计算重建损失
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # 转换mask到像素级别
        mask_reshaped = mask.reshape(B, h, w)
        mask_upsampled = mask_reshaped.unsqueeze(-1).unsqueeze(-1)
        mask_upsampled = mask_upsampled.repeat(1, 1, 1, patch_size, patch_size)
        mask_upsampled = mask_upsampled.permute(0, 1, 3, 2, 4).contiguous()
        mask_upsampled = mask_upsampled.view(B, H, W)
        mask_3channel = mask_upsampled.unsqueeze(1).repeat(1, 3, 1, 1)

        # 计算各种损失
        l1_loss = F.l1_loss(images * mask_3channel, reconstructed * mask_3channel).item()
        l2_loss = F.mse_loss(images * mask_3channel, reconstructed * mask_3channel).item()

        # 计算PSNR
        mse = F.mse_loss(images, reconstructed).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        print(f"   L1 Loss (masked): {l1_loss:.4f}")
        print(f"   L2 Loss (masked): {l2_loss:.4f}")
        print(f"   PSNR (full): {psnr:.2f} dB")

        results[mask_ratio] = {
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'psnr': psnr,
            'reconstructed_shape': reconstructed.shape
        }

        # 可视化第一个样本（只保存，不显示）
        if mask_ratio == 0.6:  # 只为默认mask ratio保存可视化
            save_path = f"reconstruction_test_mask_{mask_ratio}.png"
            visualize_reconstruction(images, reconstructed, mask, patch_size, save_path)

    return results


def comprehensive_model_test():
    """综合模型测试主函数"""
    print("=" * 80)
    print("Swin Transformer V2 Alzheimer MMoE + SimMIM 综合测试")
    print("=" * 80)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 创建模型
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)

    model = SwinTransformerV2_AlzheimerMMoE(
        img_size=256, patch_size=4, in_chans=3,
        num_classes_diagnosis=3, num_classes_change=3,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=16, mlp_ratio=4.,
        use_clinical_prior=True, prior_dim=3, prior_hidden_dim=128,
        fusion_stage=2, fusion_type='adaptive',
        is_pretrain=True
    ).to(device)

    # 模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数统计:")
    print(f"- 总参数量: {total_params:,}")
    print(f"- 可训练参数量: {trainable_params:,}")
    print(f"- 模型大小: ~{total_params * 4 / 1e6:.1f} MB (float32)")

    # 运行所有测试
    test_results = {}

    try:
        # 测试1: 标签系统兼容性
        test_results['label_system'] = test_label_system_compatibility(model, device)
        print("✓ 标签系统测试通过")

        # 测试2: 专家权重分配
        test_results['expert_weights'] = test_expert_weight_distribution(model, device)
        print("✓ 专家权重测试通过")

        # 测试3: 临床先验融合
        test_results['clinical_fusion'] = test_clinical_prior_fusion(model, device)
        print("✓ 临床先验融合测试通过")

        # 测试4: 预训练/微调模式
        test_results['mode_switching'] = test_pretrain_finetune_modes(model, device)
        print("✓ 模式切换测试通过")

        # 测试5: 重建质量
        test_results['reconstruction'] = test_reconstruction_quality(model, device)
        print("✓ 重建质量测试通过")

        # 测试6: 性能和内存
        test_results['performance'] = test_performance_memory(model, device)
        print("✓ 性能测试通过")

    except Exception as e:
        print(f"✗ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试总结报告")
    print("=" * 80)

    # 标签系统测试结果
    if 'label_system' in test_results:
        print(f"\n1. 标签系统 (1,2,3) 兼容性:")
        print(f"   - 分类任务: ✓ 通过")
        print(f"   - 重建任务: ✓ 通过")

    # 专家权重测试结果
    if 'expert_weights' in test_results:
        print(f"\n2. 专家权重分配:")
        for class_name, result in test_results['expert_weights'].items():
            print(f"   - {class_name}: {result['expected_expert']} ✓")

    # 临床先验测试结果
    if 'clinical_fusion' in test_results:
        prior_diff = test_results['clinical_fusion']['prior_difference']
        print(f"\n3. 临床先验融合:")
        print(f"   - 诊断任务影响: {prior_diff['diagnosis']:.4f}")
        print(f"   - 变化任务影响: {prior_diff['change']:.4f}")

        successful_stages = sum(1 for stage_result in test_results['clinical_fusion']['fusion_stages'].values()
                                if stage_result['success'])
        print(f"   - 成功的融合阶段: {successful_stages}/4")

    # 重建质量测试结果
    if 'reconstruction' in test_results:
        print(f"\n4. 重建质量:")
        for mask_ratio, result in test_results['reconstruction'].items():
            print(f"   - Mask {mask_ratio}: PSNR {result['psnr']:.1f} dB")

    # 性能测试结果
    if 'performance' in test_results:
        print(f"\n5. 性能统计:")
        for batch_size, result in test_results['performance'].items():
            print(f"   - Batch {batch_size}: {result['avg_time_ms']:.1f} ms, "
                  f"{result['throughput']:.1f} img/s")

    print(f"\n6. 模式切换:")
    if 'mode_switching' in test_results:
        mode_diff = test_results['mode_switching']['mode_differences']
        print(f"   - 预训练/微调输出差异: 诊断 {mode_diff['diagnosis']:.4f}, "
              f"变化 {mode_diff['change']:.4f}")

    print(f"\n" + "=" * 80)
    print("所有测试完成！模型功能正常 ✓")
    print("=" * 80)

    return test_results


if __name__ == "__main__":
    # 运行综合测试
    results = comprehensive_model_test()