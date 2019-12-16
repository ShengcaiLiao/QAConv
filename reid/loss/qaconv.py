"""Class for the Query-Adaptive Convolution (QAConv) in the evaluation phase
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-identification with Query-adaptive
    Convolution and Temporal Lifting." In arXiv preprint, arXiv:1904.10424, 2019.
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


class QAConv(Module):
    def __init__(self, prob_fea, qaconv_layer):
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
        kernel = F.normalize(prob_fea)  # [p, d, h, w]
        kernel = kernel.permute([0, 2, 3, 1])  # [p, h, w, d]
        kernel = kernel.contiguous().view(self.num_probs, self.hw, self.fea_dims, 1, 1)
        self.register_buffer('kernel', kernel)  # [p, hw, d, 1, 1]

    def forward(self, gal_fea, gal_batch_size=16, prob_batch_size=4096):
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
        gal_f = F.normalize(gal_fea).contiguous()  # [g, d, h, w]
        score = torch.zeros(num_gals, self.num_probs).to(gal_fea.device)

        for i in range(0, num_gals, gal_batch_size):
            j = min(i + gal_batch_size, num_gals)
            if prob_batch_size < self.num_probs:
                score_ = torch.zeros(j - i, self.num_probs, 2 * self.hw).to(gal_fea.device)
                for k in range(0, self.num_probs, prob_batch_size):
                    k2 = min(k + prob_batch_size, self.num_probs)
                    kernel = self.kernel[k: k2]
                    kernel = kernel.view(-1, self.fea_dims, 1, 1)
                    s = F.conv2d(gal_f[i: j, :, :, :], kernel)  # [j - i, (k2-k)*hw, h, w]
                    s = s.view(j - i, k2 - k, self.hw, self.hw)
                    score_[:, k: k2, :] = torch.cat((s.max(dim=2)[0], s.max(dim=3)[0]), dim=-1)  # [j - i, k2 - k, 2 * hw]
            else:
                score_ = F.conv2d(gal_f[i: j, :, :, :], self.kernel.view(-1, self.fea_dims, 1, 1))  # [j - i, phw, h, w]
                score_ = score_.view(j - i, self.num_probs, self.hw, self.hw)
                score_ = torch.cat((score_.max(dim=2)[0], score_.max(dim=3)[0]), dim=-1)
            score_ = score_.view(-1, 1, self.num_probs * 2 * self.hw)
            score_ = self.bn(score_).view(-1, 2 * self.hw)
            score[i: j, :] = self.fc(score_).view(-1, self.num_probs)

        score = self.logit_bn(score.unsqueeze(1)).view(num_gals, self.num_probs)
        score = torch.sigmoid(score / 10.)

        return score
