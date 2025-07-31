# --------------------------------------------------------
# Swin Transformer V2 with AlzheimerMMoE and SimMIM - Nine Label Version
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified for Alzheimer's Disease MMoE with SimMIM - 8 Experts, 7 Change Classes
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeatureLevelAttention(nn.Module):
    """Feature-level attention mechanism for gating network"""

    def __init__(self, dim, num_experts=8):
        super().__init__()
        self.dim = dim

        # Feature transformation network
        self.linear1 = nn.Linear(self.dim, int(self.dim * 0.5))
        self.feature_attention = nn.Linear(int(self.dim * 0.5), int(self.dim * 0.5))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(int(self.dim * 0.5), num_experts)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        attention_scores = self.feature_attention(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        x = x * attention_weights

        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AlzheimerMMoE(nn.Module):
    """Multi-gate Mixture of Experts for Alzheimer Disease Analysis - 8 Experts Version"""

    def __init__(self, dim, hidden_dim, num_experts=8, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # 8个专家：2个shared + 2个CN + 2个MCI + 2个AD
        self.experts = nn.ModuleList([
            Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
            for _ in range(num_experts)
        ])

        # 诊断任务的门控网络
        self.diagnosis_gate = FeatureLevelAttention(dim, num_experts)
        # 变化任务的门控网络
        self.change_gate = FeatureLevelAttention(dim, num_experts)

        # 任务特定的特征变换层
        self.diagnosis_transform = nn.Linear(dim, dim)
        self.change_transform = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lbls_diagnosis=None, lbls_change=None, is_pretrain=True, temperature=1.0, task='both'):
        """
        Args:
            x: 输入特征 [B, L, dim]
            lbls_diagnosis: 诊断标签 [B] (1: CN, 2: MCI, 3: AD)
            lbls_change: 变化标签 [B] (1-7: 详见注释)
            is_pretrain: 是否为预训练阶段
            temperature: softmax温度参数
            task: 'both', 'diagnosis', 'change', or 'reconstruction' (for SimMIM)
        Returns:
            tuple: (diagnosis_output, change_output) or single output for reconstruction

        专家索引定义（8个专家）：
            - 专家0-1: Shared experts (通用专家)
            - 专家2-3: CN-focused experts (对应诊断标签1)
            - 专家4-5: MCI-focused experts (对应诊断标签2)
            - 专家6-7: AD-focused experts (对应诊断标签3)

        Change标签定义：
            1: Stable NL to NL
            2: Stable MCI to MCI
            3: Stable Dementia to Dementia
            4: Conversion NL to MCI
            5: Conversion MCI to Dementia
            6: Conversion NL to Dementia
            7: Reversion MCI to NL
        """
        batch_size, seq_len, feature_dim = x.shape
        device = x.device
        dtype = x.dtype

        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [B, L, dim]
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, dim, num_experts]

        # ===== SimMIM重建任务 =====
        if task == 'reconstruction':
            if lbls_diagnosis is not None:
                # 基于诊断标签的专家选择（仅使用diagnosis label）
                reconstruction_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                                     device=device, dtype=dtype)
                # Shared experts保持30%权重
                reconstruction_weights[:, :, 0] = 0.15
                reconstruction_weights[:, :, 1] = 0.15

                # 根据诊断标签分配专家权重（每个类别对应的两个专家各35%）
                for i, lbl in enumerate(lbls_diagnosis):
                    if lbl == 1:  # CN -> CN专家(索引2,3)
                        reconstruction_weights[i, :, 2] = 0.35
                        reconstruction_weights[i, :, 3] = 0.35
                    elif lbl == 2:  # MCI -> MCI专家(索引4,5)
                        reconstruction_weights[i, :, 4] = 0.35
                        reconstruction_weights[i, :, 5] = 0.35
                    elif lbl == 3:  # AD -> AD专家(索引6,7)
                        reconstruction_weights[i, :, 6] = 0.35
                        reconstruction_weights[i, :, 7] = 0.35
            else:
                # 没有标签时，平均使用所有专家
                reconstruction_weights = torch.ones(batch_size, seq_len, self.num_experts,
                                                    device=device, dtype=dtype) / self.num_experts

            # 计算重建输出
            reconstruction_weights_expanded = reconstruction_weights.unsqueeze(2)  # [B, L, 1, num_experts]
            reconstruction_output = torch.sum(expert_outputs * reconstruction_weights_expanded, dim=-1)  # [B, L, dim]
            return reconstruction_output

        # ===== 双任务分类模式 =====

        # === 诊断任务专家路由 ===
        if is_pretrain and lbls_diagnosis is not None:
            # 预训练阶段：基于标签的先验路由
            diagnosis_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                            device=device, dtype=dtype)
            # Shared experts权重
            diagnosis_weights[:, :, 0] = 0.15
            diagnosis_weights[:, :, 1] = 0.15

            # 根据诊断标签分配专家权重
            for i, lbl in enumerate(lbls_diagnosis):
                if lbl == 1:  # CN -> CN专家(索引2,3)
                    diagnosis_weights[i, :, 2] = 0.35
                    diagnosis_weights[i, :, 3] = 0.35
                elif lbl == 2:  # MCI -> MCI专家(索引4,5)
                    diagnosis_weights[i, :, 4] = 0.35
                    diagnosis_weights[i, :, 5] = 0.35
                elif lbl == 3:  # AD -> AD专家(索引6,7)
                    diagnosis_weights[i, :, 6] = 0.35
                    diagnosis_weights[i, :, 7] = 0.35
        else:
            # 微调阶段：使用学习的门控网络
            diagnosis_transformed_x = self.diagnosis_transform(x)
            diagnosis_gate_logits = self.diagnosis_gate(diagnosis_transformed_x)
            diagnosis_weights = F.softmax(diagnosis_gate_logits / temperature, dim=-1)

        # === 变化任务专家路由 ===
        if is_pretrain and lbls_change is not None:
            # 预训练阶段：基于标签的先验路由
            change_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                         device=device, dtype=dtype)
            # Shared experts权重
            change_weights[:, :, 0] = 0.15
            change_weights[:, :, 1] = 0.15

            for i, lbl in enumerate(lbls_change):
                if lbl in [1, 2, 3]:  # 稳定类别
                    if lbl == 1:  # Stable CN to CN
                        change_weights[i, :, 2] = 0.35  # CN专家
                        change_weights[i, :, 3] = 0.35
                    elif lbl == 2:  # Stable MCI to MCI
                        change_weights[i, :, 4] = 0.35  # MCI专家
                        change_weights[i, :, 5] = 0.35
                    elif lbl == 3:  # Stable AD to AD
                        change_weights[i, :, 6] = 0.35  # AD专家
                        change_weights[i, :, 7] = 0.35

                elif lbl in [4, 5, 6]:  # 恶化类别
                    if lbl == 4:  # CN to MCI
                        change_weights[i, :, 2] = 0.175  # CN专家部分权重
                        change_weights[i, :, 3] = 0.175
                        change_weights[i, :, 4] = 0.175  # MCI专家部分权重
                        change_weights[i, :, 5] = 0.175
                    elif lbl == 5:  # MCI to AD
                        change_weights[i, :, 4] = 0.175  # MCI专家部分权重
                        change_weights[i, :, 5] = 0.175
                        change_weights[i, :, 6] = 0.175  # AD专家部分权重
                        change_weights[i, :, 7] = 0.175
                    elif lbl == 6:  # CN to AD (跨越式恶化)
                        change_weights[i, :, 2] = 0.117  # CN专家小权重
                        change_weights[i, :, 3] = 0.117
                        change_weights[i, :, 4] = 0.116  # MCI专家小权重（过渡）
                        change_weights[i, :, 5] = 0.116
                        change_weights[i, :, 6] = 0.117  # AD专家小权重
                        change_weights[i, :, 7] = 0.117

                elif lbl == 7:  # 改善类别 MCI to CN
                    change_weights[i, :, 4] = 0.175  # MCI专家部分权重
                    change_weights[i, :, 5] = 0.175
                    change_weights[i, :, 2] = 0.175  # CN专家部分权重
                    change_weights[i, :, 3] = 0.175
        else:
            # 微调阶段：使用学习的门控网络
            change_transformed_x = self.change_transform(x)
            change_gate_logits = self.change_gate(change_transformed_x)
            change_weights = F.softmax(change_gate_logits / temperature, dim=-1)

        # === 计算任务特定输出 ===
        # 诊断任务输出
        diagnosis_weights_expanded = diagnosis_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        diagnosis_output = torch.sum(expert_outputs * diagnosis_weights_expanded, dim=-1)  # [B, L, dim]

        # 变化任务输出
        change_weights_expanded = change_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        change_output = torch.sum(expert_outputs * change_weights_expanded, dim=-1)  # [B, L, dim]

        # 应用dropout
        diagnosis_output = self.dropout(diagnosis_output)
        change_output = self.dropout(change_output)

        return diagnosis_output, change_output

    def get_gate_weights(self, x, temperature=1.0):
        """获取两个门控网络的权重分布（用于分析和可视化）"""
        diagnosis_transformed_x = self.diagnosis_transform(x)
        change_transformed_x = self.change_transform(x)

        diagnosis_gate_logits = self.diagnosis_gate(diagnosis_transformed_x)
        change_gate_logits = self.change_gate(change_transformed_x)

        diagnosis_weights = F.softmax(diagnosis_gate_logits / temperature, dim=-1)
        change_weights = F.softmax(change_gate_logits / temperature, dim=-1)

        return {
            'diagnosis_weights': diagnosis_weights,
            'change_weights': change_weights,
            'diagnosis_logits': diagnosis_gate_logits,
            'change_logits': change_gate_logits
        }


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # 修复：确保clamp操作中的tensor在同一设备上
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device, dtype=self.logit_scale.dtype))
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock_ADMMoE(nn.Module):
    """Swin Transformer Block with Alzheimer Multi-gate MoE."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, is_pretrain=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_pretrain = is_pretrain

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_diagnosis = norm_layer(dim)  # 诊断任务的norm
        self.norm2_change = norm_layer(dim)  # 变化任务的norm
        self.norm2_reconstruction = norm_layer(dim)  # 重建任务的norm

        # 使用AlzheimerMMoE替换标准MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mmoe = AlzheimerMMoE(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            num_experts=8,  # 8个专家
            act_layer=act_layer,
            drop=drop
        )

        # 计算attention mask
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, lbls_diagnosis=None, lbls_change=None, task='both'):
        """
        Pre-norm Swin Transformer Block with Alzheimer MMoE and SimMIM support.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with MMoE
        x = shortcut + self.drop_path(x)

        if task == 'reconstruction':
            # For SimMIM reconstruction
            x_norm = self.norm2_reconstruction(x)
            reconstruction_output = self.mmoe(
                x_norm,
                lbls_diagnosis=lbls_diagnosis,
                lbls_change=lbls_change,
                is_pretrain=self.is_pretrain,
                temperature=1.0,
                task='reconstruction'
            )
            x = x + self.drop_path(reconstruction_output)
            return x
        else:
            # Original dual-task logic
            # 分别对两个任务应用norm
            x_diagnosis_norm = self.norm2_diagnosis(x)
            x_change_norm = self.norm2_change(x)

            # 使用平均norm结果作为MMoE输入
            mmoe_input = (x_diagnosis_norm + x_change_norm) / 2
            diagnosis_output, change_output = self.mmoe(
                mmoe_input,
                lbls_diagnosis=lbls_diagnosis,
                lbls_change=lbls_change,
                is_pretrain=self.is_pretrain,
                temperature=1.0,
                task=task
            )

            # 应用drop_path
            x_diagnosis = x + self.drop_path(diagnosis_output)
            x_change = x + self.drop_path(change_output)

            return x_diagnosis, x_change

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mmoe (估算，实际需要考虑MMoE的复杂度)
        flops += 8 * H * W * self.dim * self.dim * self.mlp_ratio  # 8个专家
        # norm2 (两个任务)
        flops += 2 * self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayerMMoE(nn.Module):
    """A basic Swin Transformer layer for one stage with Alzheimer MMoE."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, is_pretrain=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.is_pretrain = is_pretrain

        # build blocks with AlzheimerMMoE
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_ADMMoE(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                is_pretrain=is_pretrain)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, lbls_diagnosis=None, lbls_change=None, task='both'):
        if task == 'reconstruction':
            # For SimMIM reconstruction
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, lbls_diagnosis, lbls_change, task)
                else:
                    x = blk(x, lbls_diagnosis, lbls_change, task)

            if self.downsample is not None:
                x = self.downsample(x)

            return x
        else:
            # Original dual-task logic
            x_diagnosis, x_change = x, x

            for blk in self.blocks:
                if self.use_checkpoint:
                    x_diagnosis, x_change = checkpoint.checkpoint(blk, x_diagnosis, lbls_diagnosis, lbls_change)
                else:
                    x_diagnosis, x_change = blk(x_diagnosis, lbls_diagnosis, lbls_change)

            if self.downsample is not None:
                x_diagnosis = self.downsample(x_diagnosis)
                x_change = self.downsample(x_change)

            return x_diagnosis, x_change

    def set_pretrain_mode(self, is_pretrain):
        """设置预训练模式"""
        self.is_pretrain = is_pretrain
        for blk in self.blocks:
            blk.is_pretrain = is_pretrain

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ClinicalPriorEncoder(nn.Module):
    """
    临床先验编码器
    将3维的prior向量编码到与图像特征相同的维度
    """

    def __init__(self, prior_dim=3, hidden_dim=128, output_dim=384, dropout=0.1):
        """
        Args:
            prior_dim: 输入prior向量维度 (默认3)
            hidden_dim: MLP隐藏层维度
            output_dim: 输出维度，需要与Stage 2的特征维度匹配
            dropout: dropout率
        """
        super().__init__()

        # 简单的3层MLP
        self.mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, prior):
        """
        Args:
            prior: [B, 3] 临床先验向量
        Returns:
            encoded_prior: [B, output_dim] 编码后的特征
        """
        return self.mlp(prior)


class ClinicalImageFusion(nn.Module):
    """
    临床特征与图像特征融合模块
    支持多种融合策略
    """

    def __init__(self, image_dim, clinical_dim, fusion_type='adaptive', dropout=0.1):
        """
        Args:
            image_dim: 图像特征维度
            clinical_dim: 临床特征维度（应该与image_dim相同）
            fusion_type: 融合类型 ('adaptive', 'concat', 'add', 'hadamard')
            dropout: dropout率
        """
        super().__init__()

        assert clinical_dim == image_dim, "Clinical and image dimensions must match"

        self.fusion_type = fusion_type
        self.image_dim = image_dim

        if fusion_type == 'adaptive':
            # 自适应融合：学习融合权重
            self.gate = nn.Sequential(
                nn.Linear(image_dim * 2, image_dim),
                nn.ReLU(inplace=True),
                nn.Linear(image_dim, 2),
                nn.Softmax(dim=1)
            )
            self.fusion_proj = nn.Linear(image_dim, image_dim)

        elif fusion_type == 'concat':
            # 拼接后投影回原维度
            self.fusion_proj = nn.Sequential(
                nn.Linear(image_dim * 2, image_dim),
                nn.LayerNorm(image_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'add':
            # 简单相加，带可学习的缩放因子
            self.clinical_scale = nn.Parameter(torch.ones(1))
            self.image_scale = nn.Parameter(torch.ones(1))

        elif fusion_type == 'hadamard':
            # Hadamard积（逐元素乘积）+ 残差连接
            self.fusion_proj = nn.Linear(image_dim, image_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, clinical_features):
        """
        Args:
            image_features: [B, L, C] 图像特征 (L = H*W)
            clinical_features: [B, C] 临床特征（已经通过encoder编码到C维）
        Returns:
            fused_features: [B, L, C] 融合后的特征
        """
        B, L, C = image_features.shape

        # 将临床特征扩展到与图像特征相同的空间维度
        # clinical_features: [B, C] -> [B, 1, C] -> [B, L, C]
        clinical_features_expanded = clinical_features.unsqueeze(1).expand(B, L, C)

        if self.fusion_type == 'adaptive':
            # 计算自适应融合权重
            combined = torch.cat([image_features, clinical_features_expanded], dim=-1)
            weights = self.gate(combined.mean(dim=1))  # [B, 2]

            # 应用权重
            image_weight = weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
            clinical_weight = weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]

            fused = image_weight * image_features + clinical_weight * clinical_features_expanded
            fused = self.fusion_proj(fused)

        elif self.fusion_type == 'concat':
            # 拼接并投影
            combined = torch.cat([image_features, clinical_features_expanded], dim=-1)
            fused = self.fusion_proj(combined)

        elif self.fusion_type == 'add':
            # 加权相加
            fused = self.image_scale * image_features + self.clinical_scale * clinical_features_expanded

        elif self.fusion_type == 'hadamard':
            # Hadamard积 + 残差
            product = image_features * clinical_features_expanded
            fused = image_features + self.fusion_proj(product)

        return self.dropout(fused)


class SwinTransformerV2_AlzheimerMMoE_NineLabel(nn.Module):
    """
    扩展版Swin Transformer V2 with Alzheimer MMoE and SimMIM
    集成临床先验信息和SimMIM预训练
    支持8个专家和7分类的change label
    """

    def __init__(self,
                 # 原有参数
                 img_size=224, patch_size=4, in_chans=3,
                 num_classes_diagnosis=3, num_classes_change=7,  # 注意这里改为7
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 is_pretrain=True,
                 # 新增参数
                 use_clinical_prior=True,
                 prior_dim=3,
                 prior_hidden_dim=128,
                 fusion_stage=2,  # 在哪个stage后融合
                 fusion_type='adaptive',
                 **kwargs):

        super().__init__()

        # ===== 保存所有原有初始化代码 =====
        self.num_classes_diagnosis = num_classes_diagnosis
        self.num_classes_change = num_classes_change
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.is_pretrain = is_pretrain

        # 临床先验相关参数
        self.use_clinical_prior = use_clinical_prior
        self.fusion_stage = fusion_stage

        # For SimMIM
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ===== 临床先验编码器 =====
        if self.use_clinical_prior:
            # 计算融合stage的特征维度
            # 注意：如果在stage i后融合，实际维度是 embed_dim * 2^(i+1)
            # 因为PatchMerging会使维度翻倍
            if fusion_stage < self.num_layers - 1:
                # 如果不是最后一个stage，需要考虑PatchMerging的维度翻倍
                fusion_dim = int(embed_dim * 2 ** (fusion_stage + 1))
            else:
                # 最后一个stage没有PatchMerging
                fusion_dim = int(embed_dim * 2 ** fusion_stage)

            # 临床先验编码器
            self.clinical_encoder = ClinicalPriorEncoder(
                prior_dim=prior_dim,
                hidden_dim=prior_hidden_dim,
                output_dim=fusion_dim,
                dropout=drop_rate
            )

            # 融合模块
            self.clinical_fusion = ClinicalImageFusion(
                image_dim=fusion_dim,
                clinical_dim=fusion_dim,
                fusion_type=fusion_type,
                dropout=drop_rate
            )

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerMMoE(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                is_pretrain=is_pretrain)
            self.layers.append(layer)

        # Task-specific norms and heads
        self.norm_diagnosis = norm_layer(self.num_features)
        self.norm_change = norm_layer(self.num_features)
        self.norm = norm_layer(self.num_features)  # For SimMIM
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head_diagnosis = nn.Linear(self.num_features, num_classes_diagnosis)
        self.head_change = nn.Linear(self.num_features, num_classes_change)  # 7 classes

        # For SimMIM decoder
        # 计算encoder的总下采样倍数
        self.encoder_stride = patch_size * (2 ** (self.num_layers - 1))  # patch_size=4, 4个stage，总共下采样32倍

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_features,
                out_channels=self.encoder_stride ** 2 * 3,  # 32^2 * 3 = 3072
                kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),  # 上采样32倍
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'mask_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'cpb_mlp'}

    def forward_features(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None, mask=None):
        """
        Args:
            x: 图像输入
            clinical_prior: [B, 3] 临床先验向量
            lbls_diagnosis: 诊断标签
            lbls_change: 变化标签 (1-7)
            mask: SimMIM mask [B, num_patches]
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Apply mask for SimMIM
        if mask is not None:
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            # mask应该是[B, L]的形状
            if mask.dim() == 2 and mask.shape[1] == L:
                w = mask.unsqueeze(-1).type_as(mask_tokens)
            else:
                # 如果mask维度不对，尝试reshape
                w = mask.view(B, L).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 编码临床先验
        if self.use_clinical_prior and clinical_prior is not None:
            clinical_features = self.clinical_encoder(clinical_prior)
        else:
            clinical_features = None

        # For SimMIM reconstruction
        if mask is not None:
            # 通过各层进行重建
            for i_layer, layer in enumerate(self.layers):
                x = layer(x, lbls_diagnosis, lbls_change, task='reconstruction')

                # 在指定stage后融合临床特征
                if self.use_clinical_prior and clinical_features is not None and i_layer == self.fusion_stage:
                    x = self.clinical_fusion(x, clinical_features)

            # Final normalization for reconstruction
            x = self.norm(x)

            # Reshape for decoder
            x = x.transpose(1, 2)
            B, C, L = x.shape
            H = W = int(L ** 0.5)
            x = x.reshape(B, C, H, W)

            return x

        # Original dual-task forward
        x_diagnosis, x_change = x, x

        for i_layer, layer in enumerate(self.layers):
            # 通过当前层
            x_diagnosis, x_change = layer(x_diagnosis, lbls_diagnosis, lbls_change)

            # 在指定stage后融合临床特征
            if self.use_clinical_prior and clinical_features is not None and i_layer == self.fusion_stage:
                # 对两个任务分支都进行融合
                x_diagnosis = self.clinical_fusion(x_diagnosis, clinical_features)
                x_change = self.clinical_fusion(x_change, clinical_features)

        # Final normalization and pooling
        x_diagnosis = self.norm_diagnosis(x_diagnosis)
        x_change = self.norm_change(x_change)

        x_diagnosis = self.avgpool(x_diagnosis.transpose(1, 2))
        x_change = self.avgpool(x_change.transpose(1, 2))

        x_diagnosis = torch.flatten(x_diagnosis, 1)
        x_change = torch.flatten(x_change, 1)

        return x_diagnosis, x_change

    def forward(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None, return_features=False, mask=None):
        """
        前向传播
        Args:
            x: 输入图像
            clinical_prior: [B, 3] 临床先验向量
            lbls_diagnosis: 诊断标签
            lbls_change: 变化标签 (1-7)
            return_features: 是否返回特征
            mask: SimMIM mask
        """
        if mask is not None:
            # SimMIM reconstruction
            z = self.forward_features(x, clinical_prior, lbls_diagnosis, lbls_change, mask)
            x_rec = self.decoder(z)
            return x_rec

        # Classification
        features_diagnosis, features_change = self.forward_features(
            x, clinical_prior, lbls_diagnosis, lbls_change
        )

        # 双任务输出
        output_diagnosis = self.head_diagnosis(features_diagnosis)
        output_change = self.head_change(features_change)

        if return_features:
            return output_diagnosis, output_change, features_diagnosis, features_change
        else:
            return output_diagnosis, output_change

    def get_expert_utilization(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None):
        """获取专家利用率（用于分析MMoE的工作情况）"""
        gate_weights_list = []

        # 设置hook来收集门控权重
        def hook_fn(module, input, output):
            if hasattr(module, 'mmoe') and hasattr(module.mmoe, 'get_gate_weights'):
                # 获取MMoE模块的门控权重
                mmoe_input = input[0]  # 假设输入是tuple的第一个元素
                gate_weights = module.mmoe.get_gate_weights(mmoe_input)
                gate_weights_list.append(gate_weights)

        hooks = []
        for layer in self.layers:
            for block in layer.blocks:
                if hasattr(block, 'mmoe'):
                    hook = block.register_forward_hook(hook_fn)
                    hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            _ = self.forward(x, clinical_prior, lbls_diagnosis, lbls_change)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        return gate_weights_list

    def flops(self):
        """计算FLOPs"""
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        # 两个分类头的FLOPs
        flops += self.num_features * self.num_classes_diagnosis
        flops += self.num_features * self.num_classes_change
        return flops


if __name__ == "__main__":
    """测试8专家7分类版本"""
    import torch

    print("=" * 80)
    print("Testing SwinTransformerV2_AlzheimerMMoE_NineLabel")
    print("8 Experts, 7 Change Classes")
    print("=" * 80)

    # 测试配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    img_size = 256
    patch_size = 4
    in_chans = 3
    prior_dim = 3

    # 模型配置
    model = SwinTransformerV2_AlzheimerMMoE_NineLabel(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes_diagnosis=3,  # CN(1), MCI(2), AD(3)
        num_classes_change=7,  # 7 detailed change classes
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        is_pretrain=False,
        use_clinical_prior=True,
        prior_dim=prior_dim,
        prior_hidden_dim=128,
        fusion_stage=2,
        fusion_type='adaptive'
    ).to(device)

    # 准备测试数据
    test_image = torch.randn(batch_size, in_chans, img_size, img_size).to(device)
    test_prior = torch.randn(batch_size, prior_dim).to(device)
    test_diag_labels = torch.randint(1, 4, (batch_size,)).to(device)  # 1-3
    test_change_labels = torch.randint(1, 8, (batch_size,)).to(device)  # 1-7

    print(f"\nTest data shapes:")
    print(f"  Image: {test_image.shape}")
    print(f"  Prior: {test_prior.shape}")
    print(f"  Diagnosis labels: {test_diag_labels.shape} - Values: {test_diag_labels.tolist()}")
    print(f"  Change labels: {test_change_labels.shape} - Values: {test_change_labels.tolist()}")

    # 测试分类模式
    print("\n1. Testing classification mode:")
    with torch.no_grad():
        diag_out, change_out = model(
            test_image,
            clinical_prior=test_prior,
            lbls_diagnosis=test_diag_labels - 1,  # Convert to 0-indexed
            lbls_change=test_change_labels - 1
        )

    print(f"  ✓ Diagnosis output: {diag_out.shape} (expected: [{batch_size}, 3])")
    print(f"  ✓ Change output: {change_out.shape} (expected: [{batch_size}, 7])")

    # 测试SimMIM重建模式
    print("\n2. Testing SimMIM reconstruction mode:")
    mask = torch.randint(0, 2, (batch_size, (img_size // patch_size) ** 2)).float().to(device)
    with torch.no_grad():
        recon_out = model(
            test_image,
            clinical_prior=test_prior,
            lbls_diagnosis=test_diag_labels - 1,
            lbls_change=test_change_labels - 1,
            mask=mask
        )
    print(f"  ✓ Reconstruction output: {recon_out.shape} (expected: [{batch_size}, 3, {img_size}, {img_size}])")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n3. Model parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 测试专家利用率分析
    print("\n4. Testing expert utilization analysis:")
    try:
        expert_weights = model.get_expert_utilization(
            test_image,
            clinical_prior=test_prior,
            lbls_diagnosis=test_diag_labels - 1,
            lbls_change=test_change_labels - 1
        )
        print(f"  ✓ Got expert utilization data from {len(expert_weights)} layers")

        # 分析第一层的专家权重
        if expert_weights:
            first_layer_weights = expert_weights[0]
            if 'diagnosis_weights' in first_layer_weights:
                diag_weights = first_layer_weights['diagnosis_weights'].mean(
                    dim=[0, 1])  # Average over batch and sequence
                change_weights = first_layer_weights['change_weights'].mean(dim=[0, 1])

                print(f"\n  Expert weight distribution (Layer 0):")
                expert_names = ['Shared1', 'Shared2', 'CN1', 'CN2', 'MCI1', 'MCI2', 'AD1', 'AD2']
                print(f"  Diagnosis task:")
                for i, name in enumerate(expert_names):
                    print(f"    {name}: {diag_weights[i].item():.4f}")
                print(f"  Change task:")
                for i, name in enumerate(expert_names):
                    print(f"    {name}: {change_weights[i].item():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

    # 测试不同的change label对专家分配的影响
    print("\n5. Testing expert assignment for different change labels:")
    model.eval()

    # 设置为预训练模式以查看基于标签的专家分配
    model.is_pretrain = True
    for layer in model.layers:
        layer.set_pretrain_mode(True)

    change_label_descriptions = {
        1: "Stable: CN to CN",
        2: "Stable: MCI to MCI",
        3: "Stable: AD to AD",
        4: "Conversion: CN to MCI",
        5: "Conversion: MCI to AD",
        6: "Conversion: CN to AD",
        7: "Reversion: MCI to CN"
    }

    print("\n  Expected expert assignments in pretrain mode:")
    for label, desc in change_label_descriptions.items():
        print(f"  Change label {label} ({desc}):")
        if label in [1, 2, 3]:  # Stable
            if label == 1:
                print(f"    → CN experts (35% each) + Shared (15% each)")
            elif label == 2:
                print(f"    → MCI experts (35% each) + Shared (15% each)")
            elif label == 3:
                print(f"    → AD experts (35% each) + Shared (15% each)")
        elif label in [4, 5, 6]:  # Conversion
            if label == 4:
                print(f"    → CN & MCI experts (17.5% each) + Shared (15% each)")
            elif label == 5:
                print(f"    → MCI & AD experts (17.5% each) + Shared (15% each)")
            elif label == 6:
                print(f"    → CN, MCI & AD experts (~11.7% each) + Shared (15% each)")
        elif label == 7:  # Reversion
            print(f"    → MCI & CN experts (17.5% each) + Shared (15% each)")

    # 测试内存使用
    if torch.cuda.is_available():
        print("\n6. Memory usage test:")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        large_batch = 8
        large_image = torch.randn(large_batch, in_chans, img_size, img_size).to(device)
        large_prior = torch.randn(large_batch, prior_dim).to(device)

        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1024 ** 3  # GB

        with torch.no_grad():
            _ = model(large_image, clinical_prior=large_prior)

        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated() / 1024 ** 3  # GB

        print(f"  ✓ Memory usage for batch size {large_batch}:")
        print(f"    - Start: {start_mem:.2f} GB")
        print(f"    - End: {end_mem:.2f} GB")
        print(f"    - Increase: {end_mem - start_mem:.2f} GB")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)