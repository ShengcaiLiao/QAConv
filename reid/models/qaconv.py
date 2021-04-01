"""Class for the Query-Adaptive Convolution (QAConv)
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.2
        Mar. 31, 2021
    """

import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F


class QAConv(Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
        """
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width * 2, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.kernel = None

    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.fc.reset_parameters()
        self.logit_bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def make_kernel(self, features): # probe features
        kernel = features.permute([0, 2, 3, 1])  # [p, h, w, d]
        kernel = kernel.reshape(-1, self.num_features, 1, 1)  # [phw, d, 1, 1]
        self.kernel = kernel

    def forward(self, features):  # gallery features
        self._check_input_dim(features)

        hw = self.height * self.width
        batch_size = features.size(0)

        score = F.conv2d(features, self.kernel)  # [g, phw, h, w]
        score = score.view(batch_size, -1, hw, hw)
        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1)

        score = score.view(-1, 1, 2 * hw)
        score = self.bn(score).view(-1, 2 * hw)
        score = self.fc(score)
        score = self.logit_bn(score)
        score = score.view(batch_size, -1).t() # [p, g]

        return score
