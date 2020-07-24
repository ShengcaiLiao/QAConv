"""Class for the Query-Adaptive Convolution (QAConv) in the evaluation phase
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.1
        July 13, 2020
    """

import torch
from torch.nn import Module
from torch.nn import functional as F


class QAConv(Module):
    def __init__(self, prob_fea, qaconv_layer, gal_batch_size=8, prob_batch_size=256):
        """
        Inputs:
            prob_fea: probe feature map in size [p, d, h, w]
            qaconv_layer: the learned QAConv layer
        """
        super(QAConv, self).__init__()
        self.num_probs = prob_fea.size(0)
        self.hw = prob_fea.size(2) * prob_fea.size(3)
        self.fea_dims = prob_fea.size(1)
        self.bn = qaconv_layer.bn
        self.fc = qaconv_layer.fc
        self.logit_bn = qaconv_layer.logit_bn
        kernel = prob_fea.permute([0, 2, 3, 1])  # [p, h, w, d]
        kernel = kernel.contiguous().view(self.num_probs, self.hw, self.fea_dims, 1, 1)
        self.register_buffer('kernel', kernel)  # [p, hw, d, 1, 1]
        self.gal_batch_size = gal_batch_size
        self.prob_batch_size = prob_batch_size

    def forward(self, gal_fea):
        """
            gal_fea: gallery feature map in size [g, d, h, w]
            gal_batch_size: QAConv gallery batch size during testing. Reduce this if you encounter a gpu memory
            overflow.
            prob_batch_size: QAConv probe batch size (as kernel) during testing. For prob_batch_size >= p (number of
            probe images), that is, doing convolution at once with all the probe samples, the computation would be
            faster, however, in the cost of possibly large GPU memory. Reduce this  if you encounter a gpu memory
            overflow.
        """
        num_gals = gal_fea.size(0)
        score = torch.zeros(num_gals, self.num_probs, 2 * self.hw, device=gal_fea.device)

        for k in range(0, self.num_probs, self.prob_batch_size):
            k2 = min(k + self.prob_batch_size, self.num_probs)
            kernel = self.kernel[k: k2]
            kernel = kernel.view(-1, self.fea_dims, 1, 1)
            for i in range(0, num_gals, self.gal_batch_size):
                j = min(i + self.gal_batch_size, num_gals)
                s = F.conv2d(gal_fea[i: j], kernel)  # [j - i, (k2-k)*hw, h, w]
                s = s.view(j - i, k2 - k, self.hw, self.hw)
                score[i: j, k: k2, :] = torch.cat((s.max(dim=2)[0], s.max(dim=3)[0]), dim=-1)  # [j - i, k2 - k, 2 * hw]

        score = score.view(-1, 1, self.num_probs * 2 * self.hw)
        score = self.bn(score).view(-1, 2 * self.hw)
        score = self.fc(score).view(num_gals, self.num_probs)
        score = self.logit_bn(score.unsqueeze(1)).view(num_gals, self.num_probs)
        # scale matching scores to make them visually more recognizable
        score = torch.sigmoid(score / 10)

        return score
