
import torch.nn.functional as F
import math
import einops
import torch
import timm
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from natten.functional import na2d_av
from mmengine.runner import load_checkpoint
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
import itertools
try:
    from natten.functional import na2d_qk, na2d_av
    NATTEN_AVAILABLE = True
except ImportError:
    NATTEN_AVAILABLE = False
    print("Warning: natten not available, using manual implementation")
from timm.models.layers import to_2tuple, trunc_normal_

def get_conv2d(in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               padding, 
               dilation, 
               groups, 
               bias,
               attempt_use_lk_impl=True):
    
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding, 
                     dilation=dilation, 
                     groups=groups, 
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        Conv2d_BN(in_chans, embed_dim//2, ks=3, stride=2, pad=1),
        nn.GELU(),
        Conv2d_BN(embed_dim//2, embed_dim//2, ks=3, stride=1, pad=1),
        nn.GELU(),
        Conv2d_BN(embed_dim//2, embed_dim//2, ks=1, stride=1, pad=0),
        nn.GELU(),
        Conv2d_BN(embed_dim//2, embed_dim, ks=3, stride=2, pad=1),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(embed_dim)  
    )

def downsample(in_dim, out_dim):
    return nn.Sequential(
        Conv2d_BN(in_dim, out_dim, ks=3, stride=2, pad=1), 
    )        



class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class MulitPriorContext(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.dw_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dw_conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dw_conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        
     
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    
        self.act = nn.GELU()
        
    def forward(self, x):
       
        x3 = self.dw_conv3(x)
        x5 = self.dw_conv5(x)
        x7 = self.dw_conv7(x)
        
      
        x = (x3 + x5 + x7) / 3.0
        
       
        x = self.conv1x1(x)
        x = self.act(x)
        
        return x
        
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x
    
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class context_offset_guide(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.pre_conv = nn.Conv2d(context_dim, context_dim, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.post_conv = nn.Conv2d(context_dim, 32 , kernel_size=1, bias=False)
        self.norm = LayerNorm2d(context_dim)
    
    def forward(self, context_prior):
        x = self.gelu(self.pre_conv(context_prior))
        
        avg_feat = self.avg_pool(x)  
    
        
        output = self.norm(avg_feat)

        output = self.post_conv(output)
        
        return output
            
class ContextGuidedDeformableAttention(nn.Module):
   
    def __init__(self, H, W, local_dim, context_dim, x_deformable_dim, num_heads, offset_range_factor=1.0):
        super().__init__()
        self.H = H
        self.W = W
        self.dim = local_dim
        self.num_heads = num_heads
        self.head_dim = local_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.offset_range_factor = offset_range_factor
        self.num_sampling_points = 4
       
    
        self.local_q_proj = nn.Conv2d(local_dim, local_dim, kernel_size=1, bias=False)

        self.context_k_proj = nn.Conv2d(context_dim, local_dim, kernel_size=1, bias=False)
        self.context_v_proj = nn.Conv2d(x_deformable_dim, x_deformable_dim, kernel_size=1, bias=False)
       
        
        self.context_offset_guide = context_offset_guide(context_dim)
        
        self.local_offset_base = nn.Sequential(
            nn.Conv2d(local_dim,local_dim, 3, 1, 1, groups=local_dim),
            LayerNorm2d(local_dim),
            nn.GELU(),
            nn.Conv2d(local_dim, 32, 1)
        )
        
      
        self.offset_proj = nn.Conv2d(
            64, 
            num_heads * self.num_sampling_points * 2,  # 遵循deformable detr生成四个参考点
            kernel_size=1, 
            bias=True
        )
   
        self.proj = Conv2d_BN(local_dim, local_dim, bn_weight_init=0)
  
        self.apply_deformable_detr_init()
        
        points = list(itertools.product(range(self.H), range(self.W)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]     
    '''
    #copy from deformable detr
    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
    '''       
    def apply_deformable_detr_init(self):
        with torch.no_grad():
            self.offset_proj.weight.data.zero_()#对于权重初始化为0，采样仅依靠来自bias的偏移
            device = next(self.parameters()).device
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32, device=device
            ) * (2.0 * math.pi / self.num_heads)#[0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            '''
         
                    grid_init = [
            [1.0,  0.0],     # 0°
            [0.707, 0.707],  # 45°  
            [0.0,  1.0],     # 90°
            [-0.707, 0.707], # 135°
            [-1.0, 0.0],     # 180°
            [-0.707, -0.707],# 225°
            [0.0, -1.0],     # 270°
            [0.707, -0.707]  # 315°
            ]
            '''
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])#normalize 
            grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_sampling_points, 1)# [8,2]->[8, 1, 2]->[8, 4, 2]
            
            for i in range(self.num_sampling_points):
        
                grid_init[:, i, :] *= (i + 1)  
            self.offset_proj.bias.data = grid_init.view(-1)
            '''
            print(f"[DEBUG] offset_proj.bias initialized:")
            print(f"  Shape: {self.offset_proj.bias.shape}")
            print(f"  Values: {self.offset_proj.bias.data}")
            print(f"  First 8 values: {self.offset_proj.bias.data[:8]}")
            print(f"  two 8 values: {self.offset_proj.bias.data[8:16]}")
            print(f"  three 8 values: {self.offset_proj.bias.data[16:24]}")
            print(f"  four 8 values: {self.offset_proj.bias.data[24:32]}")
            print(f"  five 8 values: {self.offset_proj.bias.data[32:40]}")
            print(f"  six 8 values: {self.offset_proj.bias.data[40:48]}")
            print(f"  seven 8 values: {self.offset_proj.bias.data[48:56]}")
            print(f"  eight 8 values: {self.offset_proj.bias.data[56:64]}")
            print(f"  Range: [{self.offset_proj.bias.data.min():.4f}, {self.offset_proj.bias.data.max():.4f}]")
            print(f"  offset_proj.bias.data.shape: {self.offset_proj.bias.data.shape}")
            '''
    
    '''
    copy from deformable detr
    reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    '''
    @torch.no_grad()
    def _get_reference_points(self, H, W, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref = torch.stack((ref_x, ref_y), -1)
        ref = ref.reshape(H, W, 2)
        return ref 
    def forward(self, local_feat, context_prior, deformable_x):
      
        B, C, H, W = local_feat.shape
        device = local_feat.device
        
        q = self.local_q_proj(local_feat)#q local_dim
        context_guide = self.context_offset_guide(context_prior)#max pool avg pool  7 7 c=32 
        
        if context_guide.shape[-2:] != (H, W):
            context_guide = F.interpolate(context_guide, size=(H, W), mode='bilinear', align_corners=False)
        
        local_offset = self.local_offset_base(local_feat)   # 14 14 32    
        
 
        fused_features = torch.cat([context_guide, local_offset], dim=1)  #14 14 64
        offset = self.offset_proj(fused_features)  # 14 14 32
      
        offset = offset.permute(0, 2, 3, 1)#B H W point*head*2
        offset = offset.view(B, H, W, self.num_heads, self.num_sampling_points, 2)
        offset_normalizer = torch.tensor([W, H], device=device, dtype=torch.float32)
        offset_normalized = offset / offset_normalizer.view(1, 1, 1, 1, 1, 2)# B H W head point 2
        reference_points = self._get_reference_points(H, W, device)#基础参考点，参考deformable detr做法  H，W，2
        reference_points = reference_points[None, :, :, None, None, :].expand(#1, H, W, 1, 1, 2
    B, -1, -1, self.num_heads, self.num_sampling_points, -1
)
   
        sampling_positions = reference_points + offset_normalized
        sampling_grid = sampling_positions * 2.0 - 1.0#B H W head point 2
       
        
        C_ctx = context_prior.shape[1]
        context_per_head = C_ctx // self.num_heads
        context_multi_head = context_prior.view(B, self.num_heads, context_per_head, H, W)
        context_multi_head = context_multi_head.reshape(B * self.num_heads, context_per_head, H, W)#B*head,context_per_head,h,w
        sampling_grid_reshaped = sampling_grid.permute(0, 3, 1, 2, 4, 5) 
        sampling_grid_reshaped = sampling_grid_reshaped.reshape(B * self.num_heads, H * W, self.num_sampling_points, 2) # [B* num_heads, H * W, num_sampling_points, 2]
       

        k_sampled = F.grid_sample(context_multi_head, sampling_grid_reshaped, mode='bilinear', padding_mode='zeros', align_corners=False)#B * num_heads, context_per_head, H * W, num_sampling_points
        k_sampled = k_sampled.view(B, self.num_heads, context_per_head, H, W, self.num_sampling_points)
        k_sampled = k_sampled.sum(dim=-1)#B，head,context_per_head,h,w
        k = self.context_k_proj(k_sampled.view(B, -1, H, W))#B,context_dim,h,w
       
        v = self.context_v_proj(deformable_x)
        
        q_heads = q.view(B, self.num_heads, self.head_dim, H*W).transpose(2, 3)  # [B, H, N, D]
        k_heads = k.view(B, self.num_heads, self.head_dim, H*W).transpose(2, 3)  # [B, H, N, D]  
        v_heads = v.view(B, self.num_heads, self.head_dim, H*W).transpose(2, 3)  # [B, H, N, D]
        q_heads = q_heads * self.scale
        
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))   # [B, H, N, N]
        attn_scores = attn_scores + (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        
        out = torch.matmul(attn_weights, v_heads)  # [B, H, N, D]
        out = out.transpose(2, 3).contiguous().view(B, -1, H, W)  # [B, C, H, W]
        
        return self.proj(out)#proj_out

class MultiScaleWeightGenerator(nn.Module):
  
    def __init__(self, local_dim, context_dim, kernel_sizes, num_heads):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = num_heads
        self.scale = (local_dim // num_heads) ** -0.5
        
        self.qk_comodulator = DynamicQKCoModulator(local_dim, context_dim)
        
 
        self.spatial_weaver = SpatialAwareMultiScaleWeaver(kernel_sizes, num_heads)
        

    def forward(self, local_feat, context_prior):
        B, C, H, W = local_feat.shape
        local_feat = local_feat * self.scale

        final_q, final_k = self.qk_comodulator(local_feat, context_prior)
        
      
        query = rearrange(final_q, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(final_k, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        
        qk_affinity = einsum(query, key, 'b g c n, b g c l -> b g n l')
        qk_affinity = rearrange(qk_affinity, 'b g n l -> b l g n').contiguous()#2，49，8，196
        

        weights = self.spatial_weaver(qk_affinity)
        return weights
  
class SpatialScaleWeaver(nn.Module):
  
    def __init__(self, kernel_sizes, num_heads):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = num_heads
        self.self_scales = nn.ModuleDict()
        self.cross_scales = nn.ModuleDict()
        for ks in kernel_sizes:
            ks_sq = ks * ks
            self.self_scales[str(ks)] = LayerScale(ks_sq, init_value=1.0)
            self.cross_scales[str(ks)] = LayerScale(ks_sq, init_value=1e-4)  # 插值权重初始化较小

    
    def forward(self, spatial_weights, H, W):
       
        B, num_heads, H, W, total_weights = spatial_weights.shape
        weights_split = torch.split(spatial_weights, 
                                   [ks*ks for ks in self.kernel_sizes], dim=-1)
        woven_weights = []
        for i, (target_weights, target_ks) in enumerate(zip(weights_split, self.kernel_sizes)):
            # target_weights: [B, num_heads, H, W, target_ks_sq]
            cross_scale_contribution = torch.zeros_like(target_weights)
            for j, (source_weights, source_ks) in enumerate(zip(weights_split, self.kernel_sizes)):
                if i == j:
                    continue
                mapped_weights = self._spatial_cross_scale_mapping(
                    source_weights, source_ks, target_ks
                )
                cross_scale_contribution += mapped_weights
                
            target_reshaped = rearrange(target_weights, 'b g h w c -> (b g) c h w')
            cross_reshaped = rearrange(cross_scale_contribution, 'b g h w c -> (b g) c h w')
            self_scaled = self.self_scales[str(target_ks)](target_reshaped)
            cross_scaled = self.cross_scales[str(target_ks)](cross_reshaped)
            final_reshaped = self_scaled + cross_scaled  # [B*num_heads, ks², H, W]
            final_weights = rearrange(final_reshaped, '(b g) c h w -> b g h w c', b=B, g=self.num_heads)
            woven_weights.append(final_weights)
        return woven_weights
    

    
    def _spatial_cross_scale_mapping(self, source_weights, source_ks, target_ks):
       
        if source_ks == target_ks:
            return source_weights
        
        B, num_heads, H, W, source_ks_sq = source_weights.shape
        target_ks_sq = target_ks * target_ks
        
       
        source_2d = source_weights.view(B, num_heads, H, W, source_ks, source_ks)
        
        if source_ks < target_ks:
       
            pad_size = (target_ks - source_ks) // 2
            target_2d = F.pad(source_2d, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            
        elif source_ks > target_ks:
   
            start = (source_ks - target_ks) // 2
            end = start + target_ks
            target_2d = source_2d[:, :, :, :, start:end, start:end].contiguous()
            
        else:
       
            target_2d = source_2d
        
      
        mapped_weights = target_2d.view(B, num_heads, H, W, target_ks_sq).contiguous()
        return mapped_weights
          
class SpatialAwareMultiScaleWeaver(nn.Module):
  
    def __init__(self, kernel_sizes, num_heads):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_heads = num_heads
        
       # self.spatial_context_processor = nn.Conv2d(49, sum([ks*ks for ks in kernel_sizes]), kernel_size=1)  
        self.spatial_context_processor = nn.Sequential(
            nn.Conv2d(49, 49, kernel_size=3, padding=1,groups=49),
            nn.Conv2d(49,sum([ks*ks for ks in kernel_sizes]), kernel_size=1),
        )
        self.SpatialScaleWeaver = SpatialScaleWeaver(kernel_sizes, num_heads)
    
    def forward(self, qk_affinity):
      
        B, _, num_heads, HW = qk_affinity.shape
        H = W = int(HW ** 0.5)
        merged_affinity = qk_affinity.permute(0, 2, 1, 3)  # [B, num_heads, 49, H*W]
        merged_affinity = merged_affinity.contiguous().view(B * num_heads, 49, H, W)  # [B*num_heads, 49, H, W]
        all_spatial_weights = self.spatial_context_processor(merged_affinity)  # [B*num_heads, total_weights, H, W]
        all_spatial_weights = all_spatial_weights.view(B, num_heads, -1, H, W)  # [B, num_heads, total_weights, H, W]
        all_heads_weights = all_spatial_weights.permute(0, 1, 3, 4, 2)  # [B, num_heads, H, W, total_weights]
        woven_weights = self.SpatialScaleWeaver(all_heads_weights, H, W)
        
        return woven_weights
    
class DynamicQKCoModulator(nn.Module):
  
    def __init__(self, local_dim, context_dim):
        super().__init__()
      
        self.init_q_gen = Conv2d_BN(local_dim, local_dim//4, ks=1)
        self.init_k_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            Conv2d_BN(context_dim, local_dim//4, ks=1)
        )
        
   
        self.k_modulates_q = nn.Sequential(
            nn.Conv2d(local_dim//4, local_dim//4, 1),
            nn.GELU(),  
            nn.Conv2d(local_dim//4, local_dim//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, local_feat, context_prior):
     
        B, C, H, W = local_feat.shape 
        init_q = self.init_q_gen(local_feat) 
        init_k = self.init_k_gen(context_prior) 
        k_mod_signal = self.k_modulates_q(init_k)
        
        # 将K的调制信号广播到Q的空间尺度
        if k_mod_signal.shape[-2:] != (H, W):
            k_mod_broadcast = F.interpolate(k_mod_signal, 
                                           size=(H, W), 
                                           mode='bilinear', align_corners=False)
        else:
            k_mod_broadcast = k_mod_signal
            
        modulated_q = init_q * k_mod_broadcast 
        
       
        final_q = modulated_q + init_q  
        final_k = init_k  
        
        return final_q, final_k

class MultiScaleNeighborhoodAttention(nn.Module):
 
    def __init__(self, local_dim, context_dim, x_neighborhood_dim, num_heads, kernel_sizes=[5, 7], dilation=1):
        super().__init__()
        self.local_dim = local_dim
        self.context_dim = context_dim
        self.x_dim = x_neighborhood_dim
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilation = dilation
    
        if NATTEN_AVAILABLE:
            self.rpbs = nn.ParameterList()
            for ks in kernel_sizes:
                rpb = nn.Parameter(torch.zeros(num_heads, 2 * ks - 1, 2 * ks - 1))
                trunc_normal_(rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
                self.rpbs.append(rpb)
        
 
        self.branch_fusion = Conv2d_BN(local_dim , local_dim, ks=1 , bn_weight_init=0)
        
        self.MultiScaleWeight_generator =  MultiScaleWeightGenerator(
                           local_dim=local_dim,
                         context_dim=context_dim, 
                        kernel_sizes=kernel_sizes,
                        num_heads=num_heads
                        )
   

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)
    
    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
        """
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb
    
    
    def forward(self, local_feat, context_prior, x):
    
        
        B, C, H, W = local_feat.shape
  
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        
        branch_outputs = []
        attn_scores_group = []  
        
        hierarchical_weights = self.MultiScaleWeight_generator(local_feat, context_prior)
        for idx, kernel_size in enumerate(self.kernel_sizes):
               
                rpb_idx = self.generate_idx(kernel_size)
               
                attn_scores = self.apply_rpb(hierarchical_weights[idx], self.rpbs[idx], H, W, kernel_size, *rpb_idx)
                attn_scores = attn_scores.softmax(dim=-1)
                attn_scores_group.append(attn_scores)
        out1 = na2d_av(attn_scores_group[0].contiguous(), value[0].contiguous(), kernel_size=self.kernel_sizes[0])
        out2 = na2d_av(attn_scores_group[1].contiguous(), value[1].contiguous(), kernel_size=self.kernel_sizes[1])
        out = torch.cat([out1, out2], dim=1)
        out = rearrange(out, 'b g h w c -> b (g c) h w', h=H, w=W)
        out = self.branch_fusion(out)
        return out

class ContextGuidedAdaptiveAttention(nn.Module):
  
    def __init__(
        self,
        H=7,
        W=7,
        local_dim: int = 256,      
        context_dim: int = 512,       
        x_dim: int = 256,
        num_heads: int = 8,            
        kernel_sizes: list = [5, 7],   
        offset_range_factor: float = 1.0,
        dilation: int = 1,
       
    ):
        super().__init__()
        
        self.local_dim = local_dim
        self.context_dim = context_dim
 
        self.ContextGuidedDeformableAttention_local_dim = local_dim // 2
        self.MultiScaleNeighborhoodAttention_local_dim = local_dim // 2
      
        self.deformable_branch = ContextGuidedDeformableAttention(
            H=H,
            W=W,
            local_dim=self.ContextGuidedDeformableAttention_local_dim,
            context_dim=context_dim,
            x_deformable_dim=x_dim // 2,
            num_heads=num_heads,
            offset_range_factor=offset_range_factor
        )
        
        self.neighborhood_branch = MultiScaleNeighborhoodAttention(
            local_dim=self.MultiScaleNeighborhoodAttention_local_dim,
            context_dim=context_dim,
            x_neighborhood_dim=x_dim // 2,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilation=dilation
        )

        self.dyconv_proj = nn.Sequential(
            Conv2d_BN(local_dim, local_dim, ks=1, pad=0),
    
        )
        
    def forward(self, local_features, context_features=None, x=None):
     
        deformable_features, neighborhood_features = torch.split(
            local_features, 
            [self.ContextGuidedDeformableAttention_local_dim, self.MultiScaleNeighborhoodAttention_local_dim], 
            dim=1
        )
        deformable_x, neighborhood_x = torch.split(x, [self.ContextGuidedDeformableAttention_local_dim, self.MultiScaleNeighborhoodAttention_local_dim], dim=1)
        deform_output = self.deformable_branch(deformable_features, context_features, deformable_x)
        neighbor_output = self.neighborhood_branch(neighborhood_features, context_features, neighborhood_x)
        x = torch.cat([deform_output, neighbor_output], dim=1)
        x = self.dyconv_proj(x)
        return x
    
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'): # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
       

class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        
        self.x_proj = nn.Sequential(
            Conv2d_BN(dim, h_dim, ks=3, stride=2, pad=1)
        )
        self.h_proj = nn.Sequential(
            Conv2d_BN(h_dim//4, h_dim//4, ks=3, stride=2, pad=1)
        )
    

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv2d):

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):
        x = x + super().forward(x)
        return x

class AttnGate(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid, kernel=3):
        super().__init__()
        inner_dim = max(16, dim // red)
        #global
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.se_act = inner_act()
        self.se_expand = nn.Conv2d(inner_dim, dim, kernel_size=1)
        self.se_gate = out_act()
        

        self.glu_proj = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size = kernel, padding=kernel//2, groups=dim)
        self.act = inner_act()
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
 
        se_w = self.se_gate(self.se_expand(self.se_act(self.se_reduce(self.pool(x)))))
        x = x * se_w
        identity = x
        
 
        glu_x, feat = self.glu_proj(x).chunk(2, dim=1)
        glu_x = self.dwconv(glu_x)
        glu_x = self.act(glu_x)

        glu_x = glu_x * feat
        out = self.out_proj(glu_x)

        out = out + identity
        
        return out

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )
        
    def forward(self, x):
        x = x * self.proj(x)
        return x
    
class ConvolutionalGLU(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = int(2 * dim / 3)
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv1(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=6,
                 H=7,
                 W=7):
        super().__init__()
        self.num_heads = num_heads 
        self.scale = key_dim ** -0.5 
        self.key_dim = key_dim 
        self.nh_kd = nh_kd = key_dim * num_heads  
        self.d = int(attn_ratio * key_dim) 
        
        self.dh = dim 
        self.attn_ratio = attn_ratio 
        h = self.dh + nh_kd * 2 
        self.qkv = Conv2d_BN(dim, h, ks=1) 
    
        self.proj = Conv2d_BN(self.dh, dim, bn_weight_init=0)
 
        points = list(itertools.product(range(H), range(W)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))
  
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)#64 64 384
  
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)#弄成B H C L
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)   #弄成B C H W
        x = self.proj(x)
        return x
    
class Pre_BaseBlock(nn.Module):

    def __init__(self, 
                 dim=64,
              
                 mlp_ratio=4,
             
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
             
                 kernel=3,
                 use_checkpoint=False):
        super().__init__()

        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
      
            AttnGate(dim, kernel=kernel),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            #nn.Conv2d(mlp_dim, mlp_dim, kernel_size=3, padding=1, groups=mlp_dim),
            ResDWConv(mlp_dim, kernel_size=3),
            nn.GELU(),
            #ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        
        #self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        
        if self.res_scale:
        
            x = self.proj(x) + x
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x
    
    def forward(self, x):
        
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False) + x 
        else:
            x = self.forward_features(x)
        
        return x

class BaseBlock(nn.Module):

    def __init__(self, 
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 kernel=3,
                 use_checkpoint=False):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=kernel)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            #SEModule(dim),
            AttnGate(dim, kernel=kernel),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
          
            ResDWConv(mlp_dim, kernel_size=kernel),
            nn.GELU(),
         
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x
    
    def forward(self, x):
        
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
        return x

class AdaptiveGuidance(nn.Module):
    def __init__(self, 
                 dim=64,
               
                 drop_path=0,
                 norm_layer=LayerNorm2d,
               
                 use_checkpoint=False,
                 atten_rate=6,
                 H=7,
                 W=7):
        super().__init__()
 
        self.use_checkpoint = use_checkpoint
  
        self.sequeeze = nn.Conv2d(dim, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
            nn.Sigmoid()
        )
        

        self.norm_after_attention = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        key_dim = 16 
        self.attn = Attention(
            dim=dim,
            key_dim=key_dim,
            num_heads=4,
            attn_ratio=atten_rate,
            H=H,
            W=W,
        )
        
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim, 1, 1))
        
   
        self.conv3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1, groups=dim // 4)
        self.conv5 = nn.Conv2d(dim // 4, dim // 4, kernel_size=5, stride=1, padding=2, groups=dim // 4)
        self.conv7 = nn.Conv2d(dim // 4, dim // 4, kernel_size=7, stride=1, padding=3, groups=dim // 4)
        self.channel_compress = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.proj = nn.Sequential(
             nn.Conv2d(dim // 4, dim //4, kernel_size=1),
             nn.GELU(),
             nn.Dropout(0.1),
             nn.Conv2d(dim // 4, dim, kernel_size=1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
      
        self.ffn = ConvolutionalGLU(dim, act_layer=nn.GELU, drop=0.1)
        self.dwconv = ResDWConv(dim, kernel_size=3)
    def forward_features(self, x):
      
  
        identitysa = x

        x = identitysa + identitysa * self.sigmoid(self.sequeeze(x))

        x = x + x * self.spatial_attention((F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1))* 0.5)

        x = self.norm_after_attention(x)
        x = self.dwconv(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))

        identity = x
        x = self.channel_compress(self.norm2(x) * self.gamma + x * self.gammax)
        x = x + (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        x = identity + self.proj(x)

        x = x + self.drop_path(self.ffn(self.norm3(x)))
    
        return x
    
    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
        return x
    
class CompressConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ExpandDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Fusion(nn.Module):
    def __init__(self, in_channels, gate_kernel=3, feat_kernels=[3, 5]):
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1, bias=False)
        self.gate_dw = nn.Conv2d(in_channels, in_channels, kernel_size=gate_kernel, stride=1, padding=gate_kernel//2, groups=in_channels, bias=False)
        self.act = nn.GELU()
        self.feat_dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=feat_kernels[0], stride=1, padding=feat_kernels[0]//2, groups=in_channels, bias=False)
        self.feat_dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=feat_kernels[1], stride=1, padding=feat_kernels[1]//2, groups=in_channels, bias=False)
        self.feat_reduce = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, bias=False)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x1, x2):
        x = x1 + x2
        identity = x
        x = self.expand_conv(x)
        feat1, feat2 = torch.chunk(x, 2, dim=1)
        gate = self.act(self.gate_dw(feat1))
        feat_3x3 = self.feat_dw3(feat2)
        feat_5x5 = self.feat_dw5(feat2)
        feat_identity = feat2 
        feat_concat = torch.cat([feat_3x3, feat_5x5, feat_identity], dim=1)
        feat = self.act(self.feat_reduce(feat_concat))
        result = feat * gate
        feature = self.final_conv(result)
        feature = identity + feature
        return feature


class MutilScaleBrideBlock(nn.Module):
    def __init__(self, input_channels, kernel=[3, 5, 7, 7]):
        super().__init__()

        self.compress_conv_1 = CompressConv(input_channels[2], input_channels[1]) 
    
        self.fusion_1 = Fusion(input_channels[1], gate_kernel=kernel[0], feat_kernels=[kernel[0], kernel[1]])
        
        self.compress_conv_2 = CompressConv(input_channels[1], input_channels[0]) 
     
        self.fusion_2 = Fusion(input_channels[0], gate_kernel=kernel[1], feat_kernels=[kernel[1], kernel[2]])
        
        self.ExpandDown_pan1 = ExpandDownBlock(input_channels[0],input_channels[1])
        self.ExpandDown_pan2 = ExpandDownBlock(input_channels[1],input_channels[2])

        self.fusion_3 = Fusion(input_channels[1], gate_kernel=kernel[1], feat_kernels=[kernel[1], kernel[2]])

        self.fusion_4 = Fusion(input_channels[2], gate_kernel=kernel[0], feat_kernels=[kernel[0], kernel[1]])

    def forward(self, outs):
      
        fpn_results = [outs[2]]  
        
    
        compressed_feat = self.compress_conv_1(outs[2])#112 14 14 
        
  
        upsampled_feat = F.interpolate(compressed_feat, scale_factor=2., mode='nearest')
        
   
        fpn1 = self.fusion_1(upsampled_feat, outs[1])
        
     
        fpn_results.append(fpn1)
        

        compressed_fpn1 = self.compress_conv_2(fpn1)
        

        upsampled_fpn1 = F.interpolate(compressed_fpn1, scale_factor=2., mode='nearest')
        

        fpn2 = self.fusion_2(upsampled_fpn1, outs[0])
   

        ExpandDown_pan1 = self.ExpandDown_pan1(fpn2)

        pan1 = self.fusion_3(ExpandDown_pan1, fpn_results[1])

        ExpandDown_pan2 = self.ExpandDown_pan2(pan1)

        pan2 = self.fusion_4(ExpandDown_pan2, fpn_results[0])
  
        return pan2

class MultiScaleContextualDynamicBlock(nn.Module):
    def __init__(self,
                 H=7,
                 W=7,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 num_heads=2,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 kernel_sizes_group: list = [5, 7],
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim*mlp_ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
 
        self.num_heads = num_heads * 2
 
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint
        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)
        
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(
            Conv2d_BN(dim, dim, ks=1, pad=0),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            ResDWConv(mlp_dim, kernel_size=3),
            nn.GELU(),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
    
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.model = MultiScaleNeighborhoodAttention(
            local_dim=dim,
            context_dim=ctx_dim,
            x_neighborhood_dim = dim,
            num_heads=self.num_heads,
            kernel_sizes=kernel_sizes_group,
            dilation=1
        )
    def _forward_inner(self, x, h_x, h_r):
             
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape
        
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)
        
        local_features, context_features = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        output = self.model(local_features, context_features, x)
        
        x = output + lepe
        x = self.se_layer(x)
        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))
         
        x = self.dwconv2(x)
         
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))

        if self.is_last:
            return (x, None)
        else:
            l_x, h_x = torch.split(x, split_size_or_sections=[C, C_h], dim=1)
            return (l_x, h_x)
    
    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x

class MultiScaleContextualDeformableDynamicBlock(nn.Module):
    def __init__(self,
                 H=7,
                 W=7,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 num_heads=2,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 kernel_sizes_group: list = [5, 7],
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim*mlp_ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm

        self.num_heads = num_heads * 2

        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)
        
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(
            Conv2d_BN(dim, dim, ks=1, pad=0),
            nn.SiLU(),
             
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )
        
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            ResDWConv(mlp_dim, kernel_size=3),
            nn.GELU(),
       
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.model = ContextGuidedAdaptiveAttention(
            H=H,
            W=W,
            local_dim=dim,
            context_dim=ctx_dim, 
            x_dim=dim,
            num_heads=self.num_heads,
            kernel_sizes=kernel_sizes_group,
       
        )  
    def _forward_inner(self, x, h_x, h_r):
             
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape
        
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)
        
        local_features, context_features = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        output = self.model(local_features, context_features, x)
        x = output + lepe
        #x = self.seglu(x)
        x = self.se_layer(x)
        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))
         
        x = self.dwconv2(x)
         
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))

        if self.is_last:
            return (x, None)
        else:
            l_x, h_x = torch.split(x, split_size_or_sections=[C, C_h], dim=1)
            return (l_x, h_x)
    
    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x

class CGD(nn.Module):
    def __init__(self, 
                 image_size=224,
                 depth=[2, 2, 2, 2],
                 sub_depth=[4, 2],
                 in_chans=3, 
                 embed_dim=[96, 192, 384, 768],
                 kernel_size=[7, 7, 7, 7],
                 mlp_ratio=[4, 4, 4, 4],
                 sub_mlp_ratio=[4, 4],
                 sub_num_heads=[4, 8],
                 ls_init_value=[None, None, 1, 1],
                 res_scale=True,
                 smk_size=5,
                 deploy=False,
                 use_gemm=True,
                 use_ds=True,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_layer=LayerNorm2d,
                 projection=1024,
                 num_classes=1000,
                 use_checkpoint=[0, 0, 0, 0],
                 kernel=[3, 5, 7, 7],
                 atten_rate=6,
            ):
 
        super().__init__()
        
        fusion_dim = embed_dim[-1] + embed_dim[-1]//4
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1]//4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()
        self.mulit_prior_context = MulitPriorContext(embed_dim[3] // 4, embed_dim[2])
        
        self.projX = nn.Sequential(
            nn.Conv2d(embed_dim[2], embed_dim[2], kernel_size=1),
            nn.GELU(),
        )
        self.projX_multi = nn.Sequential(
            nn.Conv2d(embed_dim[2], embed_dim[2], kernel_size=1),
        )
        self.X_gate_final_norm = nn.BatchNorm2d(embed_dim[2])

        self.segate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim[3] // 4, embed_dim[3] // 4 // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim[3] // 4 // 8, embed_dim[2], kernel_size=1),
            nn.GELU(),
        )

        self.ls_identity = LayerScale(embed_dim[2], init_value=ls_init_value[2]) if ls_init_value is not None else nn.Identity()
        H = image_size // 4
        W = image_size // 4
        for i in range(depth[0]):
            self.blocks1.append(
                Pre_BaseBlock(
                    dim=embed_dim[0],
           
                    mlp_ratio=mlp_ratio[0],
               
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
               
                    use_checkpoint=(i<use_checkpoint[0]),
                    kernel=kernel[0],##3
                )
            )
        H = image_size // 8
        W = image_size // 8
        for i in range(depth[1]):
            self.blocks2.append(
                BaseBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i+depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[1]),
                    kernel=kernel[1],
                )
            )
        H = image_size // 16
        W = image_size // 16
        for i in range(depth[2]):
            self.blocks3.append(
                BaseBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[2]),
                    kernel=kernel[2],
                )
            )
        H = image_size // 32
        W = image_size // 32
        for i in range(depth[3]):
            self.blocks4.append(
                AdaptiveGuidance(
                    dim=embed_dim[3],
                
                    drop_path=dpr[i+sum(depth[:3])],
                    norm_layer=norm_layer,
                  
                    atten_rate=atten_rate,
                 
                    H=H,
                    W=W,
                )
            )
        H = image_size // 16
        W = image_size // 16
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                MultiScaleContextualDynamicBlock(
                    H=H,
                    W=W,
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
              
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)],
                    norm_layer=norm_layer,
              
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i==0),
                    use_checkpoint=(i<use_checkpoint[2]),
                    kernel_sizes_group=[5,13]
                )
            )
        H = image_size // 32
        W = image_size // 32
        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                MultiScaleContextualDeformableDynamicBlock(
                    H=H,
                    W=W,
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
               
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)+sub_depth[0]],
                    norm_layer=norm_layer,
             
                    is_first=False,
                    is_last=(i==sub_depth[1]-1),
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                    kernel_sizes_group=[5,7],
                )
            )

  
        if use_ds:
            self.aux_head = nn.Sequential(
                nn.BatchNorm2d(embed_dim[-1]),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
            )
        
    
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        )
        
        self.apply(self._init_weights)
        
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

        self.extra_norm = nn.ModuleList()
        for idx in range(3):
            dim = embed_dim[idx]
            self.extra_norm.append(norm_layer(dim))

        self.Mscale_bridge = MutilScaleBrideBlock(embed_dim[:3], kernel)
        self.scale_layer = LayerScale(embed_dim[2], init_value=1e-4)
        



    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward_pre_features(self, x):
        
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        self.outs.append(self.extra_norm[0](x))   
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        self.outs.append(self.extra_norm[1](x))
        return x
    
    
    def forward_base_features(self, x):
        
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        self.outs.append(self.extra_norm[2](x))
        
        feat = self.scale_layer(self.Mscale_bridge(self.outs))
        x = x + feat
        ctx = self.patch_embed4(x)
        
        for blk in self.blocks4:
            ctx = blk(ctx)

        return (x, ctx)
    
    def forward_sub_features(self, x, ctx):

        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, scale_factor=2, mode='bilinear', align_corners=False)
        
        identity_X = x
        multi=self.mulit_prior_context(ctx_up)
        X_multi = self.projX(x) * multi
        X_multi_final = self.projX_multi(X_multi) 
        x = self.X_gate_final_norm(self.segate(ctx_up) * X_multi_final) + self.ls_identity(identity_X)

        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)
        
        return (x, ctx_cls)

    def forward_features(self, x):
        self.outs = []
        x = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x)
        x, ctx_cls = self.forward_sub_features(x, ctx)

        return (x, ctx_cls)

    def forward(self, x):
        
        x, ctx = self.forward_features(x)
        x = self.head(x).flatten(1)

        if hasattr(self, 'aux_head') and self.training:
            ctx = self.aux_head(ctx).flatten(1)
            return dict(main=x, aux=ctx)
        
        return x


def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',  
        'mean': timm.data.IMAGENET_DEFAULT_MEAN,
        'std': timm.data.IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }


@register_model
def CGD(pretrained=False, pretrained_cfg=None, **kwargs):
    
    model = CGD(
        image_size=224,
        depth=[2, 2, 3, 2],
        sub_depth=[6, 2],
        embed_dim=[56, 112, 256, 320],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 5],
        sub_mlp_ratio=[3, 3],
        kernel=[3, 5, 7],
        atten_rate=6,
        **kwargs
    )

    model.default_cfg = _cfg(crop_pct=0.925)

    if pretrained:
        pretrained = None
        load_checkpoint(model, pretrained)

    return model



