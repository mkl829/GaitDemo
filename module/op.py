
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels=5,
            out_channels=64,
            groups: int = 1
    ) -> None:
        super(InputBlock, self).__init__()

        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # batch, in_channels, 64, 30
        # batch, out_channels, 32, 15
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # batch, out_channels, 16, 8
        x = self.maxpool(x)

        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int = 64,
            planes: int = 64,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # batch, inplanes, 16, 8
        # batch, planes, 8, 4
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # batch, planes, 8, 4
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SharedMLP(nn.Module):
    def __init__(self, input_channel: int,
                 layer_units: list,
                 activation=nn.ReLU(inplace=True)):
        # expect input: batch, channel, pts_num
        super(SharedMLP, self).__init__()
        self.activation = activation
        layers = []
        layers.append(nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=layer_units[0], kernel_size=1),
            nn.BatchNorm1d(layer_units[0]),
            self.activation
        ))
        for i, in_channel in enumerate(layer_units[:-1]):
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=layer_units[i + 1], kernel_size=1),
                nn.BatchNorm1d(layer_units[i + 1]),
                self.activation
            ))
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_mlp(x)

        return x

# Refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py
class STN3d(nn.Module):
    # mini_PointNet, regressses to a 3x3 matrix
    # according to STN, the transform is customized
    def __init__(self):
        # expect input: batch, channel, pts_num
        # shared MLP(64, 128, 1024)
        # maxPooling
        # FC 512,256
        # add regularization loss to make matrix close to orthogonal
        super(STN3d, self).__init__()
        self.shared_mlp = SharedMLP(input_channel=3, layer_units=[64, 128, 1024])
        self.activation = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation
        )
        self.fc = nn.Linear(256, 9)

    def forward(self, x):
        batchsize = x.size()[0]
        # batch, 64, pts_num
        x = self.shared_mlp(x)
        # batch, 1024
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.mlp(x)
        x = self.fc(x)
        # the output matrix is initialized as an identity matrix
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    # the seconde transform, regresses to kxk matrix
    def __init__(self, k=64):
        # expect input: batch, 64, pts_num
        super(STNkd, self).__init__()
        self.share_mlp = SharedMLP(input_channel=k, layer_units=[64, 128, 1024])
        self.activation = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation
        )
        self.fc = nn.Linear(256, k * k)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.share_mlp(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.mlp(x)
        x = self.fc(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
