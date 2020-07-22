from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torchvision

fea_dims_small = {'layer2': 128, 'layer3': 256, 'layer4': 512}
fea_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, final_layer='layer3', neck=128, pretrained=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.final_layer = final_layer
        self.neck = neck
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if depth < 50:
            out_planes = fea_dims_small[final_layer]
        else:
            out_planes = fea_dims[final_layer]

        if neck > 0:
            self.neck_conv = nn.Conv2d(out_planes, neck, kernel_size=3, padding=1, bias=False)
            out_planes = neck
            self.neck_bn = nn.BatchNorm2d(out_planes)

        self.num_features = out_planes

    def forward(self, inputs):
        x = inputs
        for name, module in self.base._modules.items():
            x = module(x)
            if name == self.final_layer:
                break

        if self.neck > 0:
            x = self.neck_conv(x)
            x = self.neck_bn(x)

        x = F.normalize(x)

        return x


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
