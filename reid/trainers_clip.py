from __future__ import print_function, absolute_import
import time
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, clip_value=512.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.clip_value = clip_value

    def train(self, epoch, data_loader, optimizer):
        self.model.train()
        self.criterion.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, acc = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(acc.item(), targets.size(0))

            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(self.model.parameters(), self.clip_value)
            clip_grad_norm_(self.criterion.parameters(), self.clip_value)
            
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{}][{}/{}]. '
                  'Time: {:.3f} ({:.3f}). '
                  'Data: {:.3f} ({:.3f}). '
                  'Loss: {:.3f} ({:.3f}). '
                  'Prec: {:.2%} ({:.2%}).'
                  .format(epoch + 1, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          precisions.val, precisions.avg), end='\r', file=sys.stdout.console)
            
        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        feature = self.model(inputs)
        loss, acc = self.criterion(feature, targets)
        loss = torch.mean(loss)
        acc = torch.mean(acc)
        return loss, acc
