import os, sys
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Type, Union, cast
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from model.running_mean_and_var import RunningMeanAndVar
import torch.nn.functional as F


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1
) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )

class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x

        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: Optional[Sequential] = None,
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes,
            planes,
            ngroups,
            stride,
            self.expansion,
            groups=cardinality,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x: Tensor) -> Tensor:
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(x)

Block = Union[Type[Bottleneck], Type[BasicBlock]]


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_planes: int,
        n_groups: int,
        block: Block,
        layers: List[int],
        cardinality: int = 1,
    ) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(n_groups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, n_groups, base_planes, layers[0])
        self.layer2 = self._make_layer(block, n_groups, base_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, n_groups, base_planes * 2 * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, n_groups, base_planes * 2 * 2 * 2, layers[3], stride=2)

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(
        self,
        block: Block,
        ngroups: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                cardinality=self.cardinality,
            )
        )
        self.inplanes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = cast(Tensor, x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])

    return model

# 目前只考虑输入的是depth图像，且在进入网络的时候已经走过normalize了
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        baseplanes: int = 32,
        ngroups: int = 32,
        input_img_h: int = 256,
        input_img_w: int = 256,
        make_backbone = None,
        normalize_visual_inputs: bool = False,
    ):
        super(ResNetEncoder, self).__init__()
        self._n_input_channels = in_channels
        if normalize_visual_inputs:
            self.running_mean_and_var: nn.module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        self.backbone = make_backbone(
            self._n_input_channels, baseplanes, ngroups
        )

        spatial_size_h = input_img_h // 2
        spatial_size_w = input_img_w // 2

        final_spatial_h = int(np.ceil(spatial_size_h * self.backbone.final_spatial_compress))
        final_spatial_w = int(np.ceil(spatial_size_w * self.backbone.final_spatial_compress))
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(
                after_compression_flat_size
                / (final_spatial_h * final_spatial_w)
            )
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )


        self.output_shape = (
            num_compression_channels,
            final_spatial_h,
            final_spatial_w,
        )

        self.out_linear = nn.Linear(self.output_shape[0] * self.output_shape[1] * self.output_shape[2], 128)



    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observation):
        cnn_input = []

        x = observation.clone()
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        x = self.out_linear(x.reshape(x.shape[0], -1))
        return x


def _build_bottleneck_branch(
    inplanes: int,
    planes: int,
    ngroups: int,
    stride: int,
    expansion: int,
    groups: int = 1,
) -> Sequential:
    return nn.Sequential(
        conv1x1(inplanes, planes),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion),
        nn.GroupNorm(ngroups, planes * expansion),
    )



if __name__ == '__main__':
    net = resnet18(3, 32, 32)
    img = torch.randn((4, 3, 256, 256))
    net = ResNetEncoder(make_backbone=resnet18, normalize_visual_inputs=False)
    ans = net(img)
    print('In the file resnet')