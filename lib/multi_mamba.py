# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
import logging


try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None

from .local_scan import LocalScanTriton, LocalReverseTriton, local_scan, local_scan_bchw, local_reverse
        
class FrequencyAwareModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 使用更稳定的权重生成结构
        self.conv = nn.Sequential(
            nn.Linear(in_channels*2, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # 可学习的残差缩放系数 (初始值更小)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
        # 添加正则化参数
        self.eps = 1e-8
    
    def forward(self, x):
        B, C, H_orig, W_orig = x.shape
        
        # 1. 傅里叶变换 (添加数值保护)
        x_fft = torch.fft.rfft2(x.detach() + self.eps, norm='ortho')
        mag = torch.abs(x_fft) + self.eps  # 避免零值
        phase = torch.angle(x_fft)
        
        # 2. 频率分区 (添加边界检查)
        B, C, H, W = mag.shape
        h_cut = max(1, H // 4)
        w_cut = max(1, W // 4)
        
        low_freq = mag[:, :, :h_cut, :w_cut]
        high_freq = mag[:, :, h_cut:, w_cut:]
        
        # 3. 能量计算 (添加梯度裁剪)
        low_energy = torch.mean(low_freq, dim=(2, 3)).detach()
        high_energy = torch.mean(high_freq, dim=(2, 3)).detach()
        
        # 4. 权重生成 (使用线性层替代卷积)
        energy = torch.cat([low_energy, high_energy], dim=1)
        weights = self.conv(energy).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # 5. 安全的频域重建
        weighted_mag = mag * weights
        
        # 使用更稳定的复数重建方法
        with torch.no_grad():
            # 分离计算图防止梯度传播问题
            real_part = weighted_mag * torch.cos(phase)
            imag_part = weighted_mag * torch.sin(phase)
            enhanced_fft = torch.complex(real_part, imag_part)
        
        # 6. 逆变换 (添加数值保护)
        reconstructed = torch.fft.irfft2(
            enhanced_fft, 
            s=(H_orig, W_orig),
            norm='ortho'
        )
        
        # 7. 严格的数值稳定处理
        reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=10.0, neginf=-10.0)
        reconstructed = torch.clamp(reconstructed, -5.0, 5.0)
        
        # 8. 带保护的残差连接
        return x + self.alpha * reconstructed


class MultiScan(nn.Module):

    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

    def __init__(self, dim, choices=None, token_size=(14, 14)):
        super().__init__()
        self.token_size = token_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
            # return LocalReverseTriton.apply(x, K, flip, H, W)
        else:
            raise RuntimeError(f'Direction {direction} not found.')    
        
    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        return ori_x * attn

class MultiScanVSSM(MultiScan):
    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None, deformable=True, freq_aware=True):
        super().__init__(dim, choices=choices, token_size=None)
        self.attn = BiAttn(dim)
        self.deformable = deformable
        self.freq_aware = freq_aware
        self.direction_attn = nn.Sequential(
            nn.Linear(dim, len(choices)),
            nn.Softmax(dim=-1)
        )
        # 频域感知模块
        if freq_aware:
            self.freq_module = FrequencyAwareModule(dim)
        
        # 可变形卷积参数
        if deformable:
            self.offset_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, 2, kernel_size=1)
            )
            # 初始化偏移量为0
            nn.init.constant_(self.offset_conv[-1].weight, 0)
            nn.init.constant_(self.offset_conv[-1].bias, 0)

    def deformable_scan(self, x, direction):
        """可变形局部扫描"""
        B, C, H, W = x.shape
        
        # 生成偏移场
        offset = self.offset_conv(x)
        # 新增约束：使用tanh限制偏移范围在[-1,1]内，然后缩放0.1倍
        offset = torch.tanh(offset) * 0.1  # 约束偏移幅度
        
        # 新增：偏移平滑正则化
        offset = F.avg_pool2d(offset, kernel_size=3, stride=1, padding=1)
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device)
        )
        grid = torch.stack((grid_x, grid_y), 0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 应用偏移 (限制偏移范围)
        #offset_grid = grid + offset * 0.1
        offset_grid = grid + offset
    
        # 新增：边界保护（防止采样点超出图像边界）
        offset_grid = torch.clamp(offset_grid, -1.5, 1.5)
        
        # 可变形采样
        x_deform = F.grid_sample(
            x, 
            offset_grid.permute(0, 2, 3, 1), 
            mode='bilinear', 
             padding_mode='reflection',  # 改为反射填充，保护边界padding_mode='zeros',
            align_corners=False
        )
        
        # 应用原始扫描
        return super().scan(x_deform, direction)
    
    def multi_scan(self, x):
        # 频域感知
        if self.freq_aware:
            x = self.freq_module(x)
        
        B, C, H, W = x.shape
        self.token_size = (H, W)
        
        xs = []
        for direction in self.choices:
            if self.deformable and direction.startswith('w'):
                # 可变形窗口扫描
                xs.append(self.deformable_scan(x, direction))
            else:
                # 常规扫描
                xs.append(super().scan(x, direction))
        
        # 记录扫描长度
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)
        
        # 填充到相同长度
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
            
        return torch.stack(new_xs, 1)
        
    def merge(self, xs):
        # If xs is a tensor, convert to list of tensors
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
     #   xs = [x.transpose(-2, -1) for x in xs]
        x = super().forward(xs)

        return x
    def merge_new(self, xs):
        if self.freq_aware:
            # 提取各路径频域特征
            freq_weights = []
            for i in range(len(self.choices)):
                # 获取当前路径特征 (移除填充)
                x_path = xs[:, i, :, :self.scan_lengths[i]]
                
                # 反扫描恢复空间结构
                x_path = self.reverse(x_path, self.choices[i])
                
                # 计算频域能量权重
                if len(x_path.shape) == 3:  # [B, D, L]
                    x_path = rearrange(x_path, 'b d (h w) -> b d h w', 
                                      h=self.token_size[0], w=self.token_size[1])
                
                # 傅里叶变换提取频域信息
                x_fft = torch.fft.rfft2(x_path, norm='ortho')
                magnitude = torch.abs(x_fft)
                
                # 计算低频能量权重
                B, C, H, W = magnitude.shape
                low_freq = magnitude[:, :, :H//4, :W//4]
                weight = torch.mean(low_freq, dim=(2, 3))  # [B, C]
                freq_weights.append(weight)
            
            # 归一化权重
            freq_weights = torch.stack(freq_weights, dim=1)  # [B, K, C]
            freq_weights = F.softmax(freq_weights, dim=1)
        else:
            freq_weights = torch.ones(
                xs.size(0), len(self.choices), xs.size(2), 
                device=xs.device) / len(self.choices)
        
        # 应用频域权重融合特征
        weighted_xs = []
        for i in range(len(self.choices)):
            # 移除填充并应用权重
            x_path = xs[:, i, :, :self.scan_lengths[i]]
            x_path = x_path * freq_weights[:, i].unsqueeze(-1)
            weighted_xs.append(x_path)
        
        # 反扫描恢复原始空间结构
        xs = super().multi_reverse(weighted_xs)
        
        # 应用注意力机制
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = super().forward(xs)
        
        return x
    
    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')
        
        

    
        

class MultiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        directions=None,
        token_size=(14, 14),
        use_middle_cls_token=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size
        self.use_middle_cls_token = use_middle_cls_token

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()


        #self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        self.multi_scan = MultiScanVSSM(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        if self.use_middle_cls_token:
            """
            Steps to use middle cls token
            # 1. split cls token out
            # 2. do 2d scan
            # 3. append cls token to the middle
            # 4. ssm
            # 5. split cls token out
            # 6. reverse tokens
            # 7. append cls token to the middle
            """
            cls_position = (xz.shape[1] - 1) // 2
            cls_token = xz[:, cls_position:cls_position+1]
            xz = torch.cat([xz[:, :cls_position], xz[:, cls_position+1:]], dim=1)

        xs = self.multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        if self.use_middle_cls_token:
            # step 3
            xs = [torch.cat([x[:, :, :cls_position], cls_token.transpose(-2, -1), x[:, :, cls_position:]], dim=2) for x in xs]

        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')
            
            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
                delta_softplus=True,
            )
            outs.append(out)

        if self.use_middle_cls_token:
            # step 5
            new_outs = []
            cls_tokens = []
            for out in outs:
                cls_tokens.append(out[:, :, cls_position:cls_position+1])
                new_outs.append(torch.cat([out[:, :, :cls_position], out[:, :, cls_position+1:]], dim=2))
            outs = new_outs

        outs = self.multi_scan.multi_reverse(outs)

        if self.use_middle_cls_token:
            # step 7
            new_outs = []
            for out, cls_token in zip(outs, cls_tokens):
                new_outs.append(torch.cat([out[:, :, :cls_position], cls_token, out[:, :, cls_position:]], dim=2))
            outs = new_outs

        outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


try:
    import selective_scan_cuda_oflex
except:
    selective_scan_cuda_oflex = None

class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None


class MultiVMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        directions=None,
        token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()


       # self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        self.multi_scan = MultiScanVSSM(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch_size, seq_len, dim = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=2)
        z = self.act(z)

        xs = self.multi_scan.multi_scan(x)
        outs = []
        for i, xz in enumerate(xs):
            xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            x_proj = getattr(self, f'x_proj_{i}')
            conv1d = getattr(self, f'conv1d_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

            xz = conv1d(xz)[:, :, :seq_len]
            xz = self.act(xz)

            N = A.shape[-1]
            R = dt_proj.weight.shape[-1]

            x_dbl = F.linear(rearrange(xz, 'b d l -> b l d'), x_proj.weight)
            dts, B, C = torch.split(x_dbl, [R, N, N], dim=2)
            dts = F.linear(dts, dt_proj.weight)

            dts = rearrange(dts, 'b l d -> b d l')
            B = rearrange(B, 'b l d -> b 1 d l')
            C = rearrange(C, 'b l d -> b 1 d l')
            D = D.float()
            delta_bias = dt_proj.bias.float()

            out = SelectiveScanOflex.apply(xz.contiguous(), dts.contiguous(), A.contiguous(), B.contiguous(), C.contiguous(), D.contiguous(), delta_bias, True, True)

            outs.append(rearrange(out, "b d l -> b l d"))

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(out) for out in outs]
        out = self.multi_scan(outs)
        out = out * z
        out = self.out_proj(out)

        return out