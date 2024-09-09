#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, quantization_bits):
            quant_desc_weights = QuantDescriptor(num_bits=quantization_bits[0], fake_quant=True, axis=(0), unsigned=False)
            quant_desc_inputs = QuantDescriptor(num_bits=quantization_bits[1], fake_quant=True, axis=(0), unsigned=False)
            super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], quantization_bits),
            quant_nn.Conv2d(256, 256, 3, padding=1, bias=False,
                            quant_desc_input=quant_desc_inputs,
                            quant_desc_weight=quant_desc_weights),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            quant_nn.Conv2d(256, num_classes, 1,
                            quant_desc_input=quant_desc_inputs,
                            quant_desc_weight=quant_desc_weights)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, quantization_bits):
        quant_desc_weights = QuantDescriptor(num_bits=quantization_bits[0], fake_quant=True, axis=(0), unsigned=False)
        quant_desc_inputs = QuantDescriptor(num_bits=quantization_bits[1], fake_quant=True, axis=(0), unsigned=False)
        modules = [
            quant_nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False,
                        quant_desc_input=quant_desc_inputs,
                        quant_desc_weight=quant_desc_weights),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, quantization_bits):
        quant_desc_weights = QuantDescriptor(num_bits=quantization_bits[0], fake_quant=True, axis=(0), unsigned=False)
        quant_desc_inputs = QuantDescriptor(num_bits=quantization_bits[1], fake_quant=True, axis=(0), unsigned=False)
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            quant_nn.Conv2d(in_channels, out_channels, 1, bias=False,
                            quant_desc_weight = quant_desc_weights,
                            quant_desc_input = quant_desc_inputs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, quantization_bits):
        super(ASPP, self).__init__()
        
        if isinstance(in_channels, (list, tuple)):
            in_channels = sum(in_channels)
        
        out_channels = 256
        quant_desc_weights = QuantDescriptor(num_bits=quantization_bits[0], fake_quant=True, axis=(0), unsigned=False)
        quant_desc_inputs = QuantDescriptor(num_bits=quantization_bits[1], fake_quant=True, axis=(0), unsigned=False)
            
        modules = []
        modules.append(nn.Sequential(
            quant_nn.Conv2d(in_channels, out_channels, 1, bias=False,
                            quant_desc_input=quant_desc_inputs,
                            quant_desc_weight=quant_desc_weights),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, quantization_bits))
        modules.append(ASPPConv(in_channels, out_channels, rate2, quantization_bits))
        modules.append(ASPPConv(in_channels, out_channels, rate3, quantization_bits))
        modules.append(ASPPPooling(in_channels, out_channels, quantization_bits))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            quant_nn.Conv2d(5 * out_channels, out_channels, 1, bias=False,
                            quant_desc_input = quant_desc_inputs,
                            quant_desc_weight = quant_desc_weights),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

