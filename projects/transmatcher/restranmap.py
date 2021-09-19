"""Class for the Transformer based feature map, upon ResNet and IBN-Net
    Shengcai Liao and Ling Shao, "Transformer-Based Deep Image Matching for Generalizable Person Re-identification." 
    In arXiv preprint, arXiv:2105.14432, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.0
        May 25, 2021
    """

from __future__ import absolute_import
import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, ModuleList
import torchvision
from torch.nn.modules import TransformerEncoderLayer

fea_dims_small = {'layer2': 128, 'layer3': 256, 'layer4': 512}
fea_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        outputs = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            for i in len(outputs):
                outputs[i] = self.norm(outputs[i])

        outputs = torch.cat(outputs, dim=-1)
        return outputs


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, ibn_type=None, final_layer='layer3', neck=512, 
                nhead=1, num_encoder_layers=2, dim_feedforward=2048, dropout=0., pretrained=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.final_layer = final_layer
        self.neck = neck
        self.pretrained = pretrained

        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth: ", depth)
        if ibn_type is not None and depth == 152:
            raise KeyError("Unsupported IBN-Net depth: ", depth)

        if ibn_type is None:
            # Construct base (pretrained) resnet
            print('\nCreate ResNet model ResNet-%d.\n' % depth)
            self.base = ResNet.__factory[depth](pretrained=pretrained)
        else:
            # Construct base (pretrained) IBN-Net
            model_name = 'resnet%d_ibn_%s' % (depth, ibn_type)
            print('\nCreate IBN-Net model %s.\n' % model_name)
            self.base = torch.hub.load('XingangPan/IBN-Net', model_name, pretrained=pretrained)

        if depth < 50:
            out_planes = fea_dims_small[final_layer]
        else:
            out_planes = fea_dims[final_layer]

        if neck > 0:
            self.neck_conv = nn.Conv2d(out_planes, neck, kernel_size=3, padding=1)
            out_planes = neck

        self.encoder = None
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(out_planes, nhead, dim_feedforward, dropout)
            encoder_norm = None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.num_features = out_planes

    def forward(self, inputs):
        x = inputs
        for name, module in self.base._modules.items():
            x = module(x)
            if name == self.final_layer:
                break

        if self.neck > 0:
            x = self.neck_conv(x)

        out = x.permute(0, 2, 3, 1)  # [b, h, w, c]
        if self.encoder is not None:
            b, c, h, w = x.size()
            y = x.view(b, c, -1).permute(2, 0, 1)  # [hw, b, c]
            y = self.encoder(y)
            y = y.permute(1, 0, 2).reshape(b, h, w, -1)  # [b, h, w, c]
            out = torch.cat((out, y), dim=-1)

        return out


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        If True, will use ImageNet pretrained model.
        Default: True
    final_layer : str
        Which layer of the resnet model to use. Can be either of 'layer2', 'layer3', or 'layer4'.
        Default: 'layer3'
    neck : int
        The number of convolutional channels appended to the final layer. Negative number or 0 means skipping this.
        Default: 128
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == '__main__':
    inputs = torch.rand([64, 3, 384, 128])
    net = create('resnet50', final_layer='layer3', neck=128, num_encoder_layers=6)
    print(net)
    out = net(inputs)
    print(out.size())
