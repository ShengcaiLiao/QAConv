"""Class for the pairwise matching loss
    Shengcai Liao and Ling Shao, "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification." In arXiv preprint, arXiv:2104.01546, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.1
        Jan 3, 2023
    """

import torch
from torch.nn import Module
from torch.nn import functional as F


class PairwiseMatchingLoss(Module):
    def __init__(self, matcher):
        """
        Inputs:
            matcher: a class for matching pairs of images
        """
        super(PairwiseMatchingLoss, self).__init__()
        self.matcher = matcher

    def reset_running_stats(self):
        self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target):
        self._check_input_dim(feature)
        # self.matcher.make_kernel(feature)

        score = self.matcher(feature, feature)  # [b, b]

        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()
        loss = F.binary_cross_entropy_with_logits(score, pair_labels, reduction='none')
        loss = loss.sum(-1)

        with torch.no_grad():
            min_pos = torch.min(score * pair_labels + 
                    (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
            max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
            acc = (min_pos > max_neg).float()

        return loss, acc
