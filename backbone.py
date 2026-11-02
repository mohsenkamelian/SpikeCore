# =============================================================================
# SpikeCore — Research Code
#
# Paper:
#   "Timing Is All You Need: SpikeCore, Learnable Delays,
#    and Gain Control for Neuromorphic Classification"
#   Advances in Neural Information Processing Systems (NeurIPS), 2026.
#
# Authors:
#   Mohsen Kamelian Rad, Sotiris Moschoyiannis, Roman Bauer
#
# Repository:
#   https://github.com/mohsenkamelian/SpikeCore
#
# Citation:
#   If you use this code in your research, please cite the accompanying paper.
#   See README.md for the full reference and BibTeX entry.
# File: backbone.py
# Description:
#   Dataset-specific model architectures for SSC, SHD, and DVSGesture,
#   together with temporal delay modules, multi-branch feature extraction,
#   squeeze-and-excitation blocks, and Parameter-Free Temporal Gain Control.
# =============================================================================


# Full architecture including delay layers, MBSE, Parameter-Free Temporal Gain Control (PFTGC), and SpikeCore layers.
# Three feature extractors are defined for SSC, DVSGesture, and SHD respectively. PFTGC module, delay layers, the MBSE structure, and SpikeCore layers are similar in all of them.
# The difference is that for SSC, we use LearnableDelayConv module, while LearnableDelayLayer is used for SHD and DVSGesture. Also 2D variations of each module are used for DVSGesture, as it's a 2D data.

import torch
import torch.nn as nn
from SpikeCore import SpikeCore, StepFunction
import torch.nn.functional as F

class LearnableDelayLayer(nn.Module):
    def __init__(self, num_channels, max_delay=50):
        super().__init__()
        self.num_channels = num_channels
        self.max_delay = max_delay
        # Initialize delays to low values for stability
        self.delays = nn.Parameter(torch.empty(num_channels).uniform_(0, 10))  # Start small, learn via STE

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        device = x.device
        int_delays = self.delays.round().clamp(0, self.max_delay).long()  # STE for gradients
        output = torch.zeros_like(x)
        for c in range(C):
            d = int_delays[c]
            if d > 0:
                output[:, d:, c] = x[:, :-d, c]  # Shift forward (delay), pad zeros at start
            else:
                output[:, :, c] = x[:, :, c]
        return output
        
class LearnableDelayConv(nn.Module):
    #-- fixed small set of delays, learned mixing weights. ----
    def __init__(self, channels, max_delay=5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1,
                      groups=channels, padding=0)
            for _ in range(max_delay + 1)
        ])
        # Learnable soft selection over delays
        self.mix = nn.Parameter(torch.zeros(max_delay + 1, channels))

    def forward(self, x):
        # x: [B, C, T]
        weights = torch.softmax(self.mix, dim=0)         # [D, C]
        out = torch.zeros_like(x)
        for d, conv in enumerate(self.convs):
            if d == 0:
                shifted = x
            else:
                shifted = F.pad(x, (d, 0))[..., :x.size(-1)]  # delay by d steps
            out = out + weights[d].unsqueeze(0).unsqueeze(-1) * conv(shifted)
        return out
# ────────────────────────────────────────────────
# New: Lightweight Parameter-free Temporal Gain Control (PFTGC, applied over time dimension)
        
class PFTGC(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x: [B, T, C]
        mu = x.mean(dim=1, keepdim=True)           # [B, 1, C]
        sigma = x.std(dim=1, keepdim=True) + self.epsilon
        
        # Z-score each timestep, then gate — properly input-dependent
        z = (x - mu) / sigma                        # [B, T, C]
        attn = torch.sigmoid(z)                     # soft gate per (t, c)
        
        return x * attn + x  # residual gating

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w

class Extractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv1d(in_ch, out_ch//4, kernel_size=1)
        self.branch2 = nn.Conv1d(in_ch, out_ch//4, kernel_size=3, padding=1)
        self.branch3 = nn.Conv1d(in_ch, out_ch//4, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch//4, kernel_size=1)
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        # added SE
        self.se = SEBlock(out_ch)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        out = self.se(out)    # SE recalibrates channels
        return self.relu(out)
        
        
class InceptionTemporalModule(nn.Module):
    def __init__(self, in_channels, nb_filters=128, bottleneck_size=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1)

        # No padding in conv — handled manually in forward for exact same-length output
        self.conv10 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=10, padding=0)
        self.conv15 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=15, padding=0)
        self.conv20 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=20, padding=0)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_bottleneck = nn.Conv1d(in_channels, nb_filters, kernel_size=1)

        self.norm = nn.BatchNorm1d(nb_filters * 4)
        self.act  = nn.GELU()
        self.residual = nn.Conv1d(in_channels, nb_filters * 4, kernel_size=1)

    def _causal_pad_and_conv(self, conv, x):
        """Left-pad x so conv output matches input T exactly."""
        pad = conv.kernel_size[0] - 1
        x = F.pad(x, (pad, 0))
        return conv(x)

    def forward(self, x):
        residual = self.residual(x)
        bn = self.bottleneck(x)
        out = torch.cat([
            self._causal_pad_and_conv(self.conv10, bn),
            self._causal_pad_and_conv(self.conv15, bn),
            self._causal_pad_and_conv(self.conv20, bn),
            self.mp_bottleneck(self.maxpool(x)),
        ], dim=1)
        out = self.norm(out)
        out = self.act(out)
        return out + residual
        
# MBSE + Delay + Inception Temporal Module + PFTGC + SpikeCore
class feature_extractor(nn.Module):
    def __init__(self, sizes, unit_args, input_size): 
        super().__init__()
        
        #self.delay_layer = LearnableDelayLayer(input_size, max_delay=100)  # bumped for deeper features

        self.mbse = nn.Sequential(
            Extractor(input_size, 256),  # instead of 64 => 256
            Extractor(256, 512),
            Extractor(512, 1024),
        )
        backbone_out_channels = 1024
        # Optional PfA after residuals
        self.pfa = PFTGC()
        self.proj = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1),
            nn.GroupNorm(8, 512),
            nn.GELU())
            
        self.inception_temporal = InceptionTemporalModule(
            in_channels=512,
            nb_filters=128,      # 128 × 4 branches = 512 output channels
            bottleneck_size=32
        )
        self.delay= LearnableDelayConv(1024, max_delay=5)

        # Stacked SpikeCore layers (keep as-is, but input now from residuals)
        self.layers = nn.ModuleList()
        in_size = 512  # after residuals
        for size in sizes:
            layer = SpikeCore(size, **unit_args)
            layer.build(in_size)
            self.layers.append(layer)
            in_size = size

    def forward(self, x):
        x = x.float()
        #print("SHape of ibput before reshaping:", x.shape)
        if x.dim() > 3:
            x = x.flatten(start_dim=2)  # [B, T, input_size]
        #print(x.shape)
        #x = self.delay_layer(x)  # [B, T, input_size]
        
        x = x.permute(0, 2, 1)  # [B, input_size, T] for conv
        
        # Backbone capture intermediates

        # Multi-Branch Spatial Feature Extractor with Channel Recalibration (MBSE)
        x = self.mbse(x)
        x = self.delay(x)
        x = self.proj(x)   # [B, 512, T]
        x = self.inception_temporal(x)           # [B, 512, T]
        x = x.permute(0, 2, 1)  # back to [B, T, sizes[0]]
        x = self.pfa(x)  # Parameter-free Temporal Gain Control (PFTGC)

        # Proceed to SpikeCore layers
        B, T, _ = x.shape
        device = x.device
        states = [layer.init_state(B, device=device) for layer in self.layers]
        h = x
        final_layer_states = []
        for li, layer in enumerate(self.layers):
            h, last_step_state = layer.forward_sequence(h, states[li])
            final_layer_states.append(last_step_state)
        
        return h, final_layer_states
    
    
    
    
# The following is the feature extractor's modification for 2D DVSGesture data

class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # [B, C, 1, 1]
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w                                          # channel-wise rescaling

# This is MBSE for 2D data of DVSGesture
class MBSE2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0, "out_ch must be divisible by 4"
        branch_ch = out_ch // 4

        # Branch 1: 1x1 conv (pointwise, cross-channel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch)
        )

        # Branch 2: 1x1 then 3x3 (local spatial features)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch)
        )

        # Branch 3: 1x1 then 5x5 (larger receptive field)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch)
        )

        # Branch 4: MaxPool then 1x1 (pooling path)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch)
        )

        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se   = SEBlock2D(out_ch)

    def forward(self, x):
        # x: [B*T, C, H, W]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)   # [B*T, out_ch, H, W]
        out = self.bn(out)
        out = self.se(out)                           # SE recalibrates channels
        return self.relu(out)
        
# Feature_extractor for DVSGesture
class feature_extractor_DVS(nn.Module):
    def __init__(self, sizes, unit_args, input_size, num_res_blocks=2): # tune num_res_blocks=2-4
        super().__init__()
        self.delay_layer = LearnableDelayLayer(
            num_channels=2,
            max_delay= 50
        )
        self.feature_extractor2 = nn.Sequential(
            # Stage 1: early spatial compression (like your ResidualConv2DBlock stem)
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 128->64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 2: MBSE blocks with progressive channel growth
            MBSE2D(32,  64),                    # [B*T, 64,  64, 64]
            nn.MaxPool2d(2, stride=2),                     # [B*T, 64,  32, 32]

            MBSE2D(64,  128),                   # [B*T, 128, 32, 32]
            nn.MaxPool2d(2, stride=2),                     # [B*T, 128, 16, 16]

            MBSE2D(128, 256),                   # [B*T, 256, 16, 16]
            nn.MaxPool2d(2, stride=2),                     # [B*T, 256,  8,  8]

            MBSE2D(256, input_size),            # [B*T, input_size, 8, 8]

            nn.AdaptiveAvgPool2d(1),                       # [B*T, input_size, 1, 1]
            nn.Flatten(start_dim=1)                        # [B*T, input_size]
        )

        # Optional PFTGC after residuals
        self.pfa = PFTGC()
        # Stacked SpikeCore layers (keep as-is, but input now from residuals)
        self.layers = nn.ModuleList()
        in_size = sizes[0] # after residuals
        for size in sizes:
            layer = SpikeCore(size, **unit_args)
            layer.build(in_size)
            self.layers.append(layer)
            in_size = size
    def forward(self, x):
        x = x.float()
        x = self.delay_layer(x)     # added delay layers
        
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 2, 128, 128]
        x = self.feature_extractor2(x)  # [B*T, input_size]
        x = x.view(B, T, -1)  # [B, T, input_size]
        #print(x.shape)
        x = self.pfa(x) # optional PfA
       
        # Proceed to SpikeCore layers
        B, T, _ = x.shape
        device = x.device
        states = [layer.init_state(B, device=device) for layer in self.layers]
        h = x
        final_layer_states = []
        for li, layer in enumerate(self.layers):
            h, last_step_state = layer.forward_sequence(h, states[li])
            final_layer_states.append(last_step_state)
       
        return h, final_layer_states




# And here's the pipeline's modification for SHD dataset
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.LayerNorm(out_channels)  # or BatchNorm1d for larger batches
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Shortcut (if channels change)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.norm1(out.permute(0, 2, 1)).permute(0, 2, 1)  # norm on channel dim after permute
        
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.norm2(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        out += residual  # residual add
        return F.leaky_relu(out)
        
class feature_extractor_SHD(nn.Module):
    def __init__(self, sizes, unit_args, input_size, num_res_blocks=2):  # tune num_res_blocks=2-4
        super().__init__()
        
        self.delay_layer = LearnableDelayLayer(input_size, max_delay=100)  # bumped for deeper features
        # Residual conv blocks (deeper frontend)
        res_channels = [input_size] + [sizes[0]] * num_res_blocks  # e.g., 700 → 256 → 256 → ...
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResidualConvBlock(res_channels[i], res_channels[i+1]))
        self.backbone = nn.Sequential(
            Extractor(input_size, 64),  #
            Extractor(64, 128),
            Extractor(128, 256),
        )
        backbone_out_channels = 256
        # Optional PfA after residuals
        self.pfa = PFTGC()
        # Stacked SpikeCore layers (keep as-is, but input now from residuals)
        self.layers = nn.ModuleList()
        in_size = sizes[0]  # after residuals
        for size in sizes:
            layer = SpikeCore(size, **unit_args)
            layer.build(in_size)
            self.layers.append(layer)
            in_size = size

    def forward(self, x):
        x = x.float()
        #print("SHape of ibput before reshaping:", x.shape)
        if x.dim() > 3:
            x = x.flatten(start_dim=2)  # [B, T, input_size]
        #print(x.shape)
        x = self.delay_layer(x)  # [B, T, input_size]
        
        x = x.permute(0, 2, 1)  # [B, input_size, T] for conv
        
        # Apply residual blocks (instead of the MBSE backbone
        #for block in self.res_blocks:
        #    x = block(x)  # [B, sizes[0], T] after all
        # MBSE instead of ResBlocks
        x = self.backbone(x)
        
        x = x.permute(0, 2, 1)  # back to [B, T, sizes[0]]
        
        x = self.pfa(x)  # optional PfA
        
        # Proceed to SpikeCore layers
        B, T, _ = x.shape
        device = x.device
        states = [layer.init_state(B, device=device) for layer in self.layers]
        h = x
        final_layer_states = []
        for li, layer in enumerate(self.layers):
            h, last_step_state = layer.forward_sequence(h, states[li])
            final_layer_states.append(last_step_state)
        
        return h, final_layer_states


# Usage notes:
# 