"""Class for the Query-Adaptive Convolution (QAConv) in the evaluation phase with matching indices
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.0
        12-12-2019
    """

import torch
from torch.nn import Module
from torch.nn import functional as F


class QAConvMatch(Module):
    """
    Inputs:
        gal_fea: gallery feature map in size [g, d, h, w]
        qaconv_layer: the learned QAConv layer
    Outputs:
        score: similarity score matrix in size [p, g]
        prob_score: max score of each probe location matching over all gallery locations
        index_in_gal: the corresponding maximum index in gallery for the prob_score
        gal_score: max score of each gallery location matching over all probe locations
        index_in_prob: the corresponding maximum index in probe for the gal_score
    """

    def __init__(self, gal_fea, qaconv_layer):
        super(QAConvMatch, self).__init__()
        self.num_gals = gal_fea.size(0)
        self.height = gal_fea.size(2)
        self.width = gal_fea.size(3)
        self.hw = self.height * self.width
        self.fea_dims = gal_fea.size(1)
        self.bn = qaconv_layer.bn
        self.fc = qaconv_layer.fc
        self.logit_bn = qaconv_layer.logit_bn
        kernel = F.normalize(gal_fea)  # [g, c, h, w]
        kernel = kernel.permute([0, 2, 3, 1])  # [g, h, w, c]
        kernel = kernel.contiguous().view(-1, kernel.size(-1), 1, 1)
        self.register_buffer('kernel', kernel)  # [ghw, c, 1, 1]

    def forward(self, prob_fea):
        prob_f = F.normalize(prob_fea)  # [p, c, h, w]
        score = F.conv2d(prob_f, self.kernel)  # [p, ghw, h, w]

        num_probs = prob_fea.size(0)
        score = score.view(num_probs, self.num_gals, self.hw, self.hw)
        prob_score, index_in_gal = score.max(dim=2)
        gal_score, index_in_prob = score.max(dim=3)
        score = torch.cat((prob_score, gal_score), dim=-1).view(num_probs, 1, self.num_gals * self.hw * 2)
        score = self.bn(score).view(num_probs * self.num_gals, 2 * self.hw)

        weight = self.fc.weight.view(1, -1)
        out_score = score * weight
        out_score = out_score.view(num_probs, self.num_gals, 2, self.height, self.width)
        prob_score = out_score[:, :, 0, :, :].squeeze()
        gal_score = out_score[:, :, 1, :, :].squeeze()

        score = self.fc(score).view(num_probs, self.num_gals)
        score = self.logit_bn(score.unsqueeze(1)).view(num_probs, self.num_gals)
        # scale matching scores to make them visually more recognizable
        score = torch.sigmoid(score / 10)

        prob_score = prob_score.view(num_probs, self.num_gals, self.height, self.width)
        index_in_gal = index_in_gal.view(num_probs, self.num_gals, self.height, self.width)
        gal_score = gal_score.view(num_probs, self.num_gals, self.height, self.width)
        index_in_prob = index_in_prob.view(num_probs, self.num_gals, self.height, self.width)

        return score, prob_score, index_in_gal, gal_score, index_in_prob
