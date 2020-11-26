from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torchvision

fea_dims = [64, 128, 256, 512]


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, final_layer='layer3', ibn_layers=(0, 1, 2), neck=128, pretrained=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.final_layer = final_layer
        self.ibn_layers = ibn_layers
        self.neck = neck
        self.pretrained = pretrained
        
        if depth < 50:
            self.expansion = 1
        else:
            self.expansion = 4

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        for layer in ibn_layers:
            if layer == 0:
                self.base.bn1 = nn.InstanceNorm2d(fea_dims[0])
            else:
                module = getattr(self.base, 'layer%d' % layer)
                module[-1].in1 = nn.InstanceNorm2d(fea_dims[layer - 1] * self.expansion)

        out_planes = fea_dims[int(final_layer[-1]) - 1] * self.expansion

        if neck > 0:
            self.neck_conv = nn.Conv2d(out_planes, neck, kernel_size=3, padding=1, bias=False)
            out_planes = neck
            self.neck_bn = nn.BatchNorm2d(out_planes)

        self.num_features = out_planes

    @staticmethod
    def basic_ibnb(module, x):
        identity = x

        out = module.conv1(x)
        out = module.bn1(out)
        out = module.relu(out)

        out = module.conv2(out)
        out = module.bn2(out)

        if module.downsample is not None:
            identity = module.downsample(x)

        out += identity
        out = module.in1(out)
        out = module.relu(out)

        return out

    @staticmethod
    def bottleneck_ibnb(module, x):
        identity = x

        out = module.conv1(x)
        out = module.bn1(out)
        out = module.relu(out)

        out = module.conv2(out)
        out = module.bn2(out)
        out = module.relu(out)

        out = module.conv3(out)
        out = module.bn3(out)

        if module.downsample is not None:
            identity = module.downsample(x)

        out += identity
        out = module.in1(out)
        out = module.relu(out)

        return out

    def forward(self, inputs):
        x = inputs
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        for layer in range(1, int(self.final_layer[-1]) + 1):
            module = getattr(self.base, 'layer%d' % layer)
            if layer in self.ibn_layers:
                for sub_module in module[:-1]:
                    x = sub_module(x)
                if self.expansion == 1:
                    x = self.basic_ibnb(module[-1], x)
                else:
                    x = self.bottleneck_ibnb(module[-1], x)
            else:
                x = module(x)

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
