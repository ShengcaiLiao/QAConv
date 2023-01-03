"""Class for the Query-Adaptive Convolution (QAConv)
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.3.1
        July 1, 2021
    """

import torch
from torch import nn
from torch.nn import Module


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
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.reset_parameters()

    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, prob_fea, gal_fea):
        hw = self.height * self.width
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)
        score = torch.einsum('p c s, g c r -> p g r s', prob_fea, gal_fea)
        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1)

        score = score.view(-1, 1, hw)
        score = self.bn(score).view(-1, hw)
        score = self.fc(score)
        score = score.view(-1, 2).sum(dim=1, keepdim=True)
        score = self.logit_bn(score)
        score = score.view(prob_size, gal_size)

        return score
