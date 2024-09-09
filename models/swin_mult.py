import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.swin_transformer import Mlp

import numpy as np
from .swin_transformer import window_partition, window_reverse
from ptflops import get_model_complexity_info
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import json


# Several modules here are based on https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch


def get_head(task, quantization_bits, backbone_channels, num_outputs, input_res, config, task_map_file=None):
    task_dec_map = {
        'semseg': 'aspp',
        'normals': 'aspp',
        'sal': 'aspp',
        'human_parts': 'aspp',
    }
    if task_map_file is not None:
        with open(task_map_file, 'r') as f:
            task_dec_map = json.load(f)

    if task_dec_map[task] == 'hrnet':
        print(f"Using hrnet for task {task}")
        from models.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, num_outputs)
    elif task_dec_map[task] == 'swin':
        from models.transformer_head import SwinDecoderHead
        print(f"Using Swin for task {task}")
        return SwinDecoderHead(channels=backbone_channels, window_size=config.MODEL.SWIN.WINDOW_SIZE, resolution=input_res, num_classes=num_outputs)
    else:
        print(f"Using deeplab head for task {task}")
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, num_outputs, quantization_bits)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class Distillation(nn.Module):
    def __init__(self, tasks, auxilary_tasks, channels):
        super(Distillation, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict(
                {a: SpatialAttention(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % (
            a)]) for a in self.auxilary_tasks if a != t} for t in self.tasks}
        out = {t: x['features_%s' % (t)] + torch.sum(torch.stack(
            [v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class MultiTaskDistillation(nn.Module):
    def __init__(self, tasks, auxilary_tasks, channels, window_size=7, num_heads=9):
        super(MultiTaskDistillation, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        self.window_size = window_size

        self.attn = nn.ModuleDict({t:
                                   InterTaskWindowAttnetion(dim=channels,
                                                            window_size=to_2tuple(
                                                                self.window_size),
                                                            num_heads=num_heads,
                                                            qkv_bias=True) for t in self.tasks})

    def forward(self, x):
        out = {}
        for t in self.tasks:
            x_q = x[f"features_{t}"]
            x_k = torch.sum(torch.stack(
                [x[f"features_{a}"] for a in self.tasks if a != t]), dim=0)
            x_v = x_k.clone()
            out[t] = self.attn[t](x_q, x_k, x_v)
        return out


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2, 3))).view(B, C, 1, 1)
        return torch.mul(x, squeeze)


class ResidualConv(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResidualConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'ResidualConv only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in ResidualConv")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class InterTaskWindowAttnetion(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., mlp_ratio=4.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=0.)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, C, H, W = x_q.shape
        x_q = x_q.contiguous().permute(0, 2, 3, 1)
        x_k = x_k.contiguous().permute(0, 2, 3, 1)
        x_v = x_v.contiguous().permute(0, 2, 3, 1)
        shortcut = x_q

        x_q = window_partition(
            x_q, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[1], C)
        x_k = window_partition(
            x_k, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[1], C)
        x_v = window_partition(
            x_v, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[1], C)

        B_, N, C = x_q.shape

        x_q = self.norm1(x_q)
        x_k = self.norm1(x_k)
        x_v = self.norm1(x_v)
        q = self.q(x_q).reshape(B_, N, self.num_heads, C //
                                self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(B_, N, self.num_heads, C //
                                self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(B_, N, self.num_heads, C //
                                self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)

        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 3, 1, 2)
        return x


class InterscaleTaskPrediction(nn.Module):
    def __init__(self, num_outputs, auxilary_tasks, input_channels, task_channels):
        super(InterscaleTaskPrediction, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(ResidualConv(
                channels, channels), ResidualConv(channels, channels)) for task in self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False),
                                           nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(ResidualConv(input_channels, task_channels, downsample=downsample),
                                              ResidualConv(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict({task: nn.Conv2d(
            task_channels, num_outputs[task], 1) for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scales=[]):
        if len(features_prev_scales) > 0:
            x = {}
            for t in self.auxilary_tasks:
                temp = features_curr_scale
                for idx, f_p_s in enumerate(features_prev_scales):
                    scale = features_curr_scale.shape[-1] // f_p_s[t].shape[-1]
                    temp = torch.cat((temp, F.interpolate(
                        f_p_s[t], scale_factor=scale, mode='bilinear')), 1)
                x[t] = temp
        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % (t)])

        return out


class InitialTaskPrediction(nn.Module):
    def __init__(self, num_outputs, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPrediction, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(ResidualConv(
                channels, channels), ResidualConv(channels, channels)) for task in self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False),
                                           nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(ResidualConv(input_channels, task_channels, downsample=downsample),
                                              ResidualConv(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict({task: nn.Conv2d(
            task_channels, num_outputs[task], 1) for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = {t: torch.cat((features_curr_scale, F.interpolate(
                features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % (t)])

        return out


class FeaturePropagation(nn.Module):
    def __init__(self, auxilary_tasks, per_task_channels, window_size=7, num_heads=9, mha=False):
        super(FeaturePropagation, self).__init__()
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N*per_task_channels)
        self.window_size = window_size
        self.num_heads = num_heads
        self.mha = mha

        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//4, 1, bias=False),
                                   nn.BatchNorm2d(self.shared_channels//4))
        self.non_linear = nn.Sequential(ResidualConv(self.shared_channels, self.shared_channels//4, downsample=downsample),
                                        ResidualConv(
                                            self.shared_channels//4, self.shared_channels//4),
                                        nn.Conv2d(self.shared_channels//4, self.shared_channels, 1))

        if self.mha:
            self.self_attn = InterTaskWindowAttnetion(dim=self.shared_channels, window_size=to_2tuple(
                self.window_size), num_heads=self.num_heads)

        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = ResidualConv(self.shared_channels, self.per_task_channels,
                                                     downsample=downsample)

        self.se = nn.ModuleDict(
            {task: SqueezeAndExcitation(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        concat = torch.cat([x['features_%s' % (task)]
                           for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim=2)
        shared = torch.mul(mask, concat.view(
            B, C//self.N, self.N, H, W)).view(B, -1, H, W)

        shared = self.dimensionality_reduction(shared)

        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' % (task)]

        return out


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Sigmoid(nn.Module):
    def __init__(self, *args):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Softmax(nn.Module):
    def __init__(self, *args):
        super(Softmax, self).__init__()

    def forward(self, x):
        return torch.softmax(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class DecoderGroup(nn.Module):
    def __init__(self, tasks, mixed_precision_bits, num_outputs, channels, out_size, input_res, config, task_map_file=None):
        super(DecoderGroup, self).__init__()
        self.tasks = tasks
        self.num_outputs = num_outputs
        self.channels = channels
        self.decoders = nn.ModuleDict()
        self.out_size = out_size
        for i, task in enumerate(self.tasks, start=0):
            self.decoders[task] = get_head(task, mixed_precision_bits[i],
                                           self.channels, self.num_outputs[task], input_res, config, task_map_file=task_map_file)

    def forward(self, x):
        result = {
            task: F.interpolate(self.decoders[task](
                x[task]), self.out_size, mode='bilinear')
            for task in self.tasks
        }
        return result

    def freeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = True

    def freeze_all(self):
        for task in self.tasks:
            self.freeze_task(task)

    def unfreeze_all(self):
        for task in self.tasks:
            self.unfreeze_task(task)


class InterscaleDistillation(nn.Module):
    def __init__(self, tasks, dims, num_outputs, input_res, channels, window_size):
        super(InterscaleDistillation, self).__init__()
        self.tasks = tasks
        self.dims = dims
        self.num_outputs = num_outputs
        self.channels = channels
        self.input_res = input_res
        self.downsample_0 = torch.nn.Conv2d(
            self.dims[0], self.channels[0], 1, bias=False)
        self.downsample_1 = torch.nn.Conv2d(
            self.dims[1], self.channels[1], 1, bias=False)
        self.downsample_2 = torch.nn.Conv2d(
            self.dims[2], self.channels[2], 1, bias=False)
        self.downsample_3 = torch.nn.Conv2d(
            self.dims[3], self.channels[3], 1, bias=False)

        self.fpm_scale_3 = FeaturePropagation(
            self.tasks, self.channels[3], window_size=window_size)
        self.fpm_scale_2 = FeaturePropagation(
            self.tasks, self.channels[2], window_size=window_size)
        self.fpm_scale_1 = FeaturePropagation(
            self.tasks, self.channels[1], window_size=window_size)

        self.scale_0 = InterscaleTaskPrediction(
            self.num_outputs, self.tasks, self.channels[0] + self.channels[1] + self.channels[2] + self.channels[3], self.channels[0])
        self.scale_1 = InterscaleTaskPrediction(
            self.num_outputs, self.tasks, self.channels[1] + self.channels[2] + self.channels[3], self.channels[1])
        self.scale_2 = InterscaleTaskPrediction(
            self.num_outputs, self.tasks, self.channels[2] + self.channels[3], self.channels[2])
        self.scale_3 = InterscaleTaskPrediction(
            self.num_outputs, self.tasks, self.channels[3], self.channels[3])

        self.distillation_scale_0 = MultiTaskDistillation(
            self.tasks, self.tasks, self.channels[0], window_size=window_size)
        self.distillation_scale_1 = MultiTaskDistillation(
            self.tasks, self.tasks, self.channels[1], window_size=window_size)
        self.distillation_scale_2 = MultiTaskDistillation(
            self.tasks, self.tasks, self.channels[2], window_size=window_size)
        self.distillation_scale_3 = MultiTaskDistillation(
            self.tasks, self.tasks, self.channels[3], window_size=window_size)

    def forward(self, x):
        s_3 = x[3].view(-1, self.input_res[3],
                        self.input_res[3], self.dims[3]).permute(0, 3, 1, 2)

        s_2 = x[2].view(-1, self.input_res[2],
                        self.input_res[2], self.dims[2]).permute(0, 3, 1, 2)
        s_1 = x[1].view(-1, self.input_res[1],
                        self.input_res[1], self.dims[1]).permute(0, 3, 1, 2)
        s_0 = x[0].view(-1, self.input_res[0],
                        self.input_res[0], self.dims[0]).permute(0, 3, 1, 2)

        s_3 = self.downsample_3(s_3)
        s_2 = self.downsample_2(s_2)
        s_1 = self.downsample_1(s_1)
        s_0 = self.downsample_0(s_0)

        x_3 = self.scale_3(s_3)
        x_3_fpm = self.fpm_scale_3(x_3)

        x_2 = self.scale_2(s_2, [x_3_fpm])
        x_2_fpm = self.fpm_scale_2(x_2)

        x_1 = self.scale_1(s_1, [x_2_fpm, x_3_fpm])
        x_1_fpm = self.fpm_scale_1(x_1)

        x_0 = self.scale_0(s_0, [x_1_fpm, x_2_fpm, x_3_fpm])

        features_0 = self.distillation_scale_0(x_0)

        features_1 = self.distillation_scale_1(x_1)

        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {t: [
            features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}

        multi_scale_initial_predictions = {
            t: [x_0[t], x_1[t], x_2[t], x_3[t]] for t in self.tasks}
        return multi_scale_features, multi_scale_initial_predictions


class MultiTaskSwin(nn.Module):
    def __init__(self, encoder, decoder_config, config, mixed_precision_bits, task_map_file=None):
        super(MultiTaskSwin, self).__init__()

        self.backbone = encoder
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        self.decoder_config = decoder_config
        self.embed_dim = decoder_config['embed_dim']
        self.num_decoders = len(decoder_config['depths'])
        if hasattr(self.backbone, 'patch_embed'):
            patches_resolution = self.backbone.patch_embed.patches_resolution
            embed_dim = self.backbone.embed_dim
            num_layers = self.backbone.num_layers
            self.dims = [int((embed_dim * 2 ** ((i+1) if i < num_layers - 1 else i)))
                         for i in range(num_layers)]
            self.input_res = [patches_resolution[0] //
                              (2 ** ((i+1) if i < num_layers - 1 else i)) for i in range(num_layers)]
            self.window_size = self.backbone.layers[0].blocks[0].window_size
            self.img_size = self.backbone.patch_embed.img_size
        else:
            self.input_res = [28, 14, 7, 7]
            self.dims = [192, 384, 768, 768]
            self.window_size = config.MODEL.SWIN.WINDOW_SIZE
            self.img_size = config.DATA.IMG_SIZE
        self.channels = [18, 36, 72, 144]

        self.task_interaction = InterscaleDistillation(
            self.tasks, self.dims, self.num_outputs, self.input_res, self.channels, self.window_size)
        self.decoders = DecoderGroup(
            self.tasks, mixed_precision_bits, self.num_outputs, channels=self.channels, out_size=self.img_size, 
            input_res=self.input_res, config=config, task_map_file=task_map_file,
            )

    def forward(self, x, return_activation_stats=False, task=None):
        multi_scale_initial_predictions = None
        shared_representation = self.backbone(x, return_stages=True)
        multi_scale_features, multi_scale_initial_predictions = self.task_interaction(
            shared_representation)
        result = self.decoders(multi_scale_features)
        return result, multi_scale_initial_predictions

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self, task):
        self.decoders.freeze_task(task)

    def unfreeze_task(self, task):
        self.decoders.unfreeze_task(task)

    def freeze_backbone(self):
        for param in self.backbone():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone():
            param.requires_grad = True
