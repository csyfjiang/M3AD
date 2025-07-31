# --------------------------------------------------------
# Swin Transformer V2 with AlzheimerMMoE and SimMIM - Single Task Degraded Version
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified for Alzheimer's Disease MMoE with SimMIM
# Degraded to Single Task for Validation
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

    def __init__(self, dim, num_experts=4):
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


class AlzheimerMMoE_SingleTask(nn.Module):
    """Single Task Mixture of Experts for Alzheimer Disease Analysis - Fixed Version"""

    def __init__(self, dim, hidden_dim, num_classes=2, act_layer=nn.GELU, drop=0.):
        """
        Args:
            dim: feature dimension
            hidden_dim: hidden dimension for experts
            num_classes: number of classes (2 for binary, 3 for CN/MCI/AD)
            act_layer: activation layer
            drop: dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # 根据类别数动态设置专家数量
        if num_classes == 2:
            # 二分类：1个shared + 2个类别特定专家 = 3个专家
            self.num_experts = 3
            self.expert_names = ['Shared', 'Class_0', 'Class_1']
        elif num_classes == 3:
            # 三分类：1个shared + 3个类别特定专家 = 4个专家
            self.num_experts = 4
            self.expert_names = ['Shared', 'Class_0', 'Class_1', 'Class_2']
        else:
            # 多分类：1个shared + num_classes个类别特定专家
            self.num_experts = num_classes + 1
            self.expert_names = ['Shared'] + [f'Class_{i}' for i in range(num_classes)]

        # 创建专家网络
        self.experts = nn.ModuleList([
            Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
            for _ in range(self.num_experts)
        ])

        # 单任务门控网络
        self.gate = FeatureLevelAttention(dim, self.num_experts)

        # 任务特定的特征变换层
        self.task_transform = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, labels=None, is_pretrain=True, temperature=1.0, task='classification'):
        """
        Args:
            x: 输入特征 [B, L, dim]
            labels: 分类标签 [B] (MUST be 0-based indexing: 0, 1, 2, ...)
            is_pretrain: 是否为预训练阶段
            temperature: softmax温度参数
            task: 'classification' or 'reconstruction' (for SimMIM)
        Returns:
            output: 专家融合后的特征 [B, L, dim]
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
            if labels is not None:
                # 基于标签的专家选择
                reconstruction_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                                     device=device, dtype=dtype)
                reconstruction_weights[:, :, 0] = 0.4  # shared expert保持40%权重

                # 根据标签分配专家权重 - 假设labels已经是0-based
                for i, lbl in enumerate(labels):
                    label_val = lbl.item() if hasattr(lbl, 'item') else lbl

                    # 安全检查：确保标签在有效范围内
                    if 0 <= label_val < self.num_classes:
                        # 0-based label直接映射到对应专家：label -> expert_index = label + 1
                        expert_idx = label_val + 1
                        if expert_idx < self.num_experts:
                            reconstruction_weights[i, :, expert_idx] = 0.6
                    else:
                        # 如果标签无效，使用平均权重
                        print(
                            f"Warning: Invalid label {label_val} for {self.num_classes} classes. Using uniform weights.")
                        reconstruction_weights[i, :, :] = 1.0 / self.num_experts
            else:
                # 没有标签时，平均使用所有专家
                reconstruction_weights = torch.ones(batch_size, seq_len, self.num_experts,
                                                    device=device, dtype=dtype) / self.num_experts

            # 计算重建输出
            reconstruction_weights_expanded = reconstruction_weights.unsqueeze(2)  # [B, L, 1, num_experts]
            reconstruction_output = torch.sum(expert_outputs * reconstruction_weights_expanded, dim=-1)  # [B, L, dim]
            return reconstruction_output

        # ===== 分类任务 =====
        if is_pretrain and labels is not None:
            # 预训练阶段：基于标签的先验路由
            task_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                       device=device, dtype=dtype)
            task_weights[:, :, 0] = 0.4  # shared expert权重

            # 根据标签分配专家权重 - 假设labels已经是0-based
            for i, lbl in enumerate(labels):
                label_val = lbl.item() if hasattr(lbl, 'item') else lbl

                # 安全检查：确保标签在有效范围内
                if 0 <= label_val < self.num_classes:
                    # 0-based label直接映射到对应专家：label -> expert_index = label + 1
                    expert_idx = label_val + 1
                    if expert_idx < self.num_experts:
                        task_weights[i, :, expert_idx] = 0.6
                else:
                    # 如果标签无效，使用平均权重
                    print(f"Warning: Invalid label {label_val} for {self.num_classes} classes. Using uniform weights.")
                    task_weights[i, :, :] = 1.0 / self.num_experts
        else:
            # 微调阶段：使用学习的门控网络
            task_transformed_x = self.task_transform(x)
            gate_logits = self.gate(task_transformed_x)
            task_weights = F.softmax(gate_logits / temperature, dim=-1)

        # 计算任务输出
        task_weights_expanded = task_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        task_output = torch.sum(expert_outputs * task_weights_expanded, dim=-1)  # [B, L, dim]

        # 应用dropout
        task_output = self.dropout(task_output)

        return task_output

    def get_gate_weights(self, x, temperature=1.0):
        """获取门控网络的权重分布（用于分析和可视化）"""
        task_transformed_x = self.task_transform(x)
        gate_logits = self.gate(task_transformed_x)
        task_weights = F.softmax(gate_logits / temperature, dim=-1)

        return {
            'task_weights': task_weights,
            'gate_logits': gate_logits,
            'expert_names': self.expert_names
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


class SwinTransformerBlock_SingleTask(nn.Module):
    """Swin Transformer Block with Single Task MoE."""

    def __init__(self, dim, input_resolution, num_heads, num_classes=2, window_size=7, shift_size=0,
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
        self.num_classes = num_classes

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
        self.norm2 = norm_layer(dim)  # 单任务只需要一个norm

        # 使用单任务MMoE替换标准MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mmoe = AlzheimerMMoE_SingleTask(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
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

    def forward(self, x, labels=None, task='classification'):
        """
        Single Task Swin Transformer Block with MMoE and SimMIM support.
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

        # FFN with Single Task MMoE
        x = shortcut + self.drop_path(x)
        x_norm = self.norm2(x)

        mmoe_output = self.mmoe(
            x_norm,
            labels=labels,
            is_pretrain=self.is_pretrain,
            temperature=1.0,
            task=task
        )

        x = x + self.drop_path(mmoe_output)
        return x

    def set_pretrain_mode(self, is_pretrain):
        """设置预训练模式"""
        self.is_pretrain = is_pretrain

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mmoe (估算)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
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


class BasicLayerSingleTask(nn.Module):
    """A basic Swin Transformer layer for one stage with Single Task MoE."""

    def __init__(self, dim, input_resolution, depth, num_heads, num_classes=2, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, is_pretrain=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.is_pretrain = is_pretrain
        self.num_classes = num_classes

        # build blocks with Single Task MoE
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_SingleTask(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                num_classes=num_classes,
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

    def forward(self, x, labels=None, task='classification'):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, labels, task)
            else:
                x = blk(x, labels, task)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def set_pretrain_mode(self, is_pretrain):
        """设置预训练模式"""
        self.is_pretrain = is_pretrain
        for blk in self.blocks:
            blk.set_pretrain_mode(is_pretrain)

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


class SwinTransformerV2_SingleTask(nn.Module):
    """
    Single Task Swin Transformer V2 with MoE and SimMIM - Degraded Version for Validation
    """

    def __init__(self,
                 # 基础参数
                 img_size=224, patch_size=4, in_chans=3,
                 num_classes=2,  # 单任务：2分类或3分类
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 is_pretrain=True,
                 # 临床先验参数
                 use_clinical_prior=True,
                 prior_dim=3,
                 prior_hidden_dim=128,
                 fusion_stage=2,
                 fusion_type='adaptive',
                 **kwargs):

        super().__init__()

        # ===== 基础初始化 =====
        self.num_classes = num_classes
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
            if fusion_stage < self.num_layers - 1:
                fusion_dim = int(embed_dim * 2 ** (fusion_stage + 1))
            else:
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
            layer = BasicLayerSingleTask(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_classes=num_classes,
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

        # Task-specific norm and head
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        # For SimMIM decoder
        self.encoder_stride = patch_size * (2 ** (self.num_layers - 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_features,
                out_channels=self.encoder_stride ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
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

    def forward_features(self, x, clinical_prior=None, labels=None, mask=None):
        """
        Args:
            x: 图像输入
            clinical_prior: [B, 3] 临床先验向量
            labels: 分类标签 (0-based indexing)
            mask: SimMIM mask [B, num_patches]
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Apply mask for SimMIM
        if mask is not None:
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            if mask.dim() == 2 and mask.shape[1] == L:
                w = mask.unsqueeze(-1).type_as(mask_tokens)
            else:
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
            for i_layer, layer in enumerate(self.layers):
                x = layer(x, labels, task='reconstruction')

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

        # Classification forward
        for i_layer, layer in enumerate(self.layers):
            x = layer(x, labels, task='classification')

            # 在指定stage后融合临床特征
            if self.use_clinical_prior and clinical_features is not None and i_layer == self.fusion_stage:
                x = self.clinical_fusion(x, clinical_features)

        # Final normalization and pooling
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        return x

    def forward(self, x, clinical_prior=None, labels=None, return_features=False, mask=None):
        """
        前向传播
        Args:
            x: 输入图像
            clinical_prior: [B, 3] 临床先验向量
            labels: 分类标签 (0-based indexing)
            return_features: 是否返回特征
            mask: SimMIM mask
        """
        if mask is not None:
            # SimMIM reconstruction
            z = self.forward_features(x, clinical_prior, labels, mask)
            x_rec = self.decoder(z)
            return x_rec

        # Classification
        features = self.forward_features(x, clinical_prior, labels)
        output = self.head(features)

        if return_features:
            return output, features
        else:
            return output

    def get_expert_utilization(self, x, clinical_prior=None, labels=None):
        """获取专家利用率（用于分析MoE的工作情况）"""
        gate_weights_list = []

        # 设置hook来收集门控权重
        def hook_fn(module, input, output):
            if hasattr(module, 'mmoe') and hasattr(module.mmoe, 'get_gate_weights'):
                mmoe_input = input[0]
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
            _ = self.forward(x, clinical_prior, labels)

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
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":
    """测试单任务退化版本"""
    import torch

    print("=" * 80)
    print("Testing SwinTransformerV2_SingleTask - Degraded Version")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    img_size = 256
    patch_size = 4
    in_chans = 3
    prior_dim = 3

    # 测试二分类和三分类
    test_configs = [
        {'num_classes': 2, 'name': 'Binary Classification (0-1)'},
        {'num_classes': 3, 'name': 'Three-class Classification (0-1-2)'}
    ]

    base_config = {
        'img_size': img_size,
        'patch_size': patch_size,
        'in_chans': in_chans,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 16,
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'pretrained_window_sizes': [0, 0, 0, 0],
        'is_pretrain': False,
        'use_clinical_prior': True,
        'prior_dim': prior_dim,
        'prior_hidden_dim': 128,
        'fusion_stage': 2,
        'fusion_type': 'adaptive',
    }

    for config in test_configs:
        print(f"\n{'=' * 60}")
        print(f"Testing {config['name']}")
        print(f"{'=' * 60}")

        num_classes = config['num_classes']
        model_config = base_config.copy()
        model_config['num_classes'] = num_classes

        # 准备测试数据
        test_image = torch.randn(batch_size, in_chans, img_size, img_size).to(device)
        test_prior = torch.randn(batch_size, prior_dim).to(device)
        test_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

        print(f"Test data:")
        print(f"  Image: {test_image.shape}")
        print(f"  Prior: {test_prior.shape}")
        print(f"  Labels: {test_labels} (range: 0-{num_classes - 1})")

        try:
            # 创建模型
            model = SwinTransformerV2_SingleTask(**model_config).to(device)

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\nModel Info:")
            print(f"  Total params: {total_params:,}")
            print(f"  Trainable params: {trainable_params:,}")

            # 检查专家配置
            first_mmoe = model.layers[0].blocks[0].mmoe
            print(f"  Expert configuration:")
            print(f"    Number of experts: {first_mmoe.num_experts}")
            print(f"    Expert names: {first_mmoe.expert_names}")

            with torch.no_grad():
                # 测试分类模式
                output = model(
                    test_image,
                    clinical_prior=test_prior,
                    labels=test_labels
                )

                print(f"\nClassification Test:")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected shape: [{batch_size}, {num_classes}]")
                print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

                # 测试SimMIM重建模式
                mask = torch.randint(0, 2, (batch_size, (img_size // patch_size) ** 2)).float().to(device)
                recon_output = model(
                    test_image,
                    clinical_prior=test_prior,
                    labels=test_labels,
                    mask=mask
                )

                print(f"\nSimMIM Reconstruction Test:")
                print(f"  Reconstruction shape: {recon_output.shape}")
                print(f"  Expected shape: [{batch_size}, {in_chans}, {img_size}, {img_size}]")

                # 测试专家利用率
                expert_weights = model.get_expert_utilization(
                    test_image,
                    clinical_prior=test_prior,
                    labels=test_labels
                )

                print(f"\nExpert Utilization Test:")
                print(f"  Number of layers with MoE: {len(expert_weights)}")
                if expert_weights:
                    print(f"  First layer expert weights shape: {expert_weights[0]['task_weights'].shape}")

                print(f"  ✓ {config['name']} - All tests passed!")

        except Exception as e:
            print(f"  ✗ {config['name']} - Failed: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("Single Task Testing Complete!")
    print(f"{'=' * 80}")