
"""
Author:
    Yiqun Chen
Docs:
    Encoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from collections import OrderedDict
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

_ENCODER = {}

def add_encoder(encoder):
    _ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class DPDEncoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDEncoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1 = self._build_block(2, 6, 64, dropout=0.0)
        self.block_2 = self._build_block(2, 64, 128, dropout=0.0)
        self.block_3 = self._build_block(2, 128, 256, dropout=0.0)
        self.block_4 = self._build_block(2, 256, 512, dropout=0.4)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _build_block(self, num_conv, in_channels, out_channels, dropout=0.0):
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(("relu_"+str(idx), nn.ReLU()))
        if dropout:
            layer_list.append(
                ("dropout", nn.Dropout(dropout))
            )
        block = nn.Sequential(OrderedDict(layer_list))
        return block

    def forward(self, inp):
        enc_1 = self.block_1(inp)
        
        enc_2 = self.max_pool(enc_1)
        enc_2 = self.block_2(enc_2)

        enc_3 = self.max_pool(enc_2)
        enc_3 = self.block_3(enc_3)

        enc_4 = self.max_pool(enc_3)
        enc_4 = self.block_4(enc_4)

        bottleneck = self.max_pool(enc_4)

        return enc_1, enc_2, enc_3, enc_4, bottleneck


@add_encoder
class ResNet50Encoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(ResNet50Encoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    def forward(self, data, *args, **kwargs):
        output = self.model(data)
        raise NotImplementedError("Method ResNet50Encoder.forward is not implemented.")


@add_encoder
class ResNext50Encoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(ResNext50Encoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)

    def forward(self, data, *args, **kwargs):
        output = self.model(data)
        raise NotImplementedError("Method RextNet50Encoder.forward is not implemented.")


@add_encoder
class DeepLabV3Encoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DeepLabV3Encoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True).backbone

    def forward(self, data, *args, **kwargs):
        output = self.model(data)
        raise NotImplementedError("Method DeepLabV3Encoder.forward is not implemented.")


@add_encoder
class FCNEncoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(FCNEncoder, self).__init()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True).backbone

    def forward(self, data, *args, **kwargs):
        output = self.model(data)
        raise NotImplementedError("Method FCNEncoder.forward is not implemented.")


@add_encoder
class UNetEncoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(UNetEncoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet', 
            in_channels=3, out_channels=1, init_features=32, pretrained=True
        )
        self.model = nn.ModuleDict({
            "encoder1": model.encoder1,
            "encoder2": model.encoder2, 
            "encoder3": model.encoder3,
            "encoder4": model.encoder4,
            "bottleneck": model.bottleneck, 
            "pool1": model.pool1, 
            "pool2": model.pool2, 
            "pool3": model.pool3, 
            "pool4": model.pool4, 
        })

    def forward(self, data, *args, **kwargs):
        enc1 = self.model["encoder1"](data)
        enc2 = self.model["encoder2"](self.model["pool1"](enc1))
        enc3 = self.model["encoder3"](self.model["pool2"](enc2))
        enc4 = self.model["encoder4"](self.model["pool3"](enc3))
        bottleneck = self.bottleneck(self.model["pool4"](enc4))
        raise NotImplementedError("Method UNetEncoder.forward is not implemented.")
        return enc1, enc2, enc3, enc4, bottleneck
        


if __name__ == "__main__":
    print(_ENCODER)