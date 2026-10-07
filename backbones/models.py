import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Type, Callable, Union, List, Optional
from .resnet import *

def get_act_layer(act='relu', **config):
    if act == 'relu':
        return nn.ReLU()
    if act == 'selu':
        return nn.SELU()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'gelu':
        return nn.GELU()
    if act == 'relu6':
        return nn.ReLU6()
    if act == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=config.get('lrelu_a', 1e-2))
    if act == 'rrelu':
        return nn.RReLU()
    if act == 'elu':
        return nn.ELU()
    if act == 'hardtanh':
        return nn.Hardtanh()
    if act == 'softplus':
        return nn.Softplus()
    if act == 'silu':
        return nn.SiLU()

    raise RuntimeError(f'Unknown activation "{act}"')

class MLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=[], act='relu', n_outputs: int = 1, **config):
        super().__init__()
        layer_sizes = [n_features] + hidden_sizes + [n_outputs]
        layers = []
        for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(get_act_layer(act))
        self.last_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)
        self.init_weights(self.last_layer)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, repr=False):
        rep = self.model(x)
        out = self.last_layer(rep)
        if repr:
            return out, rep
        else:
            return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_dropout: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        bn1 = norm_layer(self.inplanes)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0])
        layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = nn.Sequential(conv1, bn1, relu, maxpool, *layer1.children(), *layer2.children(), *layer3.children(), *layer4.children(), avgpool)
        self.last_layer = nn.Linear(512 * block.expansion, num_classes)

        self.drop = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, repr: bool = False) -> Tensor:
        x = self.model(x)
        z = torch.flatten(x, 1)
        if self.use_dropout:
            z = self.drop(z)
        out = self.last_layer(z)
        if repr:
            return out, z
        return out