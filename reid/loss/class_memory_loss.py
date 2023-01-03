"""Class for the class memory based loss for QAConv
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.1
        Jan 3, 2023
    """

import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F


class ClassMemoryLoss(Module):
    def __init__(self, matcher, num_classes, num_features, height, width, mem_batch_size=16):
        """
        Inputs:
            matcher: a class for matching pairs of images
            num_classes: the number of classes in the training set.
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
            mem_batch_size: batch size of the class memory for query-adaptive convolution. For
            mem_batch_size >= num_classes, that is, doing convolution at once with all the class memory, the
            computation would be faster, however, in the cost of possibly large GPU memory.
        """
        super(ClassMemoryLoss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.height = height
        self.width = width
        self.mem_batch_size = mem_batch_size
        self.matcher = matcher
        self.register_buffer('class_memory', torch.zeros(num_classes, num_features, height, width))
        self.register_buffer('valid_class', torch.zeros(num_classes))

    def reset_running_stats(self):
        self.class_memory.zero_()
        self.valid_class.zero_()
        self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target):
        self._check_input_dim(feature)
        batch_size = target.size(0)
        # self.matcher.make_kernel(feature)

        if self.mem_batch_size < self.num_classes:
            score = torch.zeros(batch_size, self.num_classes, device=feature.device)
            for i in range(0, self.num_classes, self.mem_batch_size):
                j = min(i + self.mem_batch_size, self.num_classes)
                s = self.matcher(feature, self.class_memory[i: j, :, :, :].detach().clone())  # [b, m]
                score[:, i: j] = s
        else:
            score = self.matcher(feature, self.class_memory.detach().clone())  # [b, c]

        target1 = target.unsqueeze(1)
        onehot_labels = torch.zeros_like(score).scatter(1, target1, 1)
        loss = F.binary_cross_entropy_with_logits(score, onehot_labels, reduction='none')
        prob = score.sigmoid()
        weight = torch.pow(torch.where(onehot_labels.byte(), 1. - prob, prob), 2.)
        loss = loss * weight * self.valid_class.detach().clone().unsqueeze(0)
        loss = loss.sum(-1)

        with torch.no_grad():
            _, preds = torch.max(score, 1)
            acc = (preds == target).float()
            self.class_memory[target] = feature
            self.valid_class[target] = 1.0

        return loss, acc
