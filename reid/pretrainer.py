"""Pre-training code for a stable initialization
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.0
        Feb. 7, 2021
    """
from __future__ import print_function, absolute_import
from collections import defaultdict
import time
import sys
from copy import deepcopy

import torch
from torch import nn

from reid.utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, criterion, optimizer, data_loader, num_epochs=1, max_steps=2000, num_trials=10):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.num_trials = num_trials
        self.base_model = deepcopy(model.base.state_dict())
        self.best_loss = 1e15
        self.best_acc = -1e15
        self.best_metric = -1e15
        self.best_model = None

    def train(self, result_file, method, sub_method):
        print('Start pre-training.\n')

        for trial in range(self.num_trials):
            model = nn.DataParallel(self.model).cuda()
            criterion = nn.DataParallel(self.criterion).cuda()

            loss, acc = self.single_train(model, criterion, self.optimizer, trial)
            metric = acc * 100 - loss

            if metric > self.best_metric:  # cache
                self.best_loss = loss
                self.best_acc = acc
                self.best_metric = metric
                self.best_model = [deepcopy(model.module.state_dict()), 
                deepcopy(criterion.module.state_dict()), 
                deepcopy(self.optimizer.state_dict())]

            # reset states for the next trial
            if trial < self.num_trials - 1:
                self.model.base.load_state_dict(self.base_model)
                if self.model.neck > 0:
                    self.model.neck_conv.reset_parameters()
                    self.model.neck_bn.reset_running_stats()
                    self.model.neck_bn.reset_parameters()
                self.criterion.reset_running_stats()
                self.criterion.reset_parameters()
                self.optimizer.state = defaultdict(dict)

        print('Pre-training finished. Best metric: %.4f, Best loss: %.3f. Best acc: %.2f%%\n' 
            % (self.best_metric, self.best_loss, self.best_acc * 100))

        with open(result_file, 'a') as f:
            f.write('%s/%s:\n' % (method, sub_method))
            f.write('\tBest metric: %.4f, Best loss: %.3f. Best acc: %.2f%%\n\n' 
                % (self.best_metric, self.best_loss, self.best_acc * 100))

        self.model.load_state_dict(self.best_model[0])
        self.criterion.load_state_dict(self.best_model[1])
        self.optimizer.load_state_dict(self.best_model[2])
        
        return self.model, self.criterion, self.optimizer

    def single_train(self, model, criterion, optimizer, trial):
        model.train()
        criterion.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        iters = 0
        start_time = time.time()
        end = time.time()

        for ep in range(self.num_epochs):
            for i, inputs in enumerate(self.data_loader):
                data_time.update(time.time() - end)

                iters += 1
                inputs, targets = self._parse_data(inputs)
                loss, acc = self._forward(model, criterion, inputs, targets)

                losses.update(loss.item(), targets.size(0))
                precisions.update(acc, targets.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                print('Trial {}: epoch [{}][{}/{}]. '
                    'Time: {:.3f} ({:.3f}). '
                    'Data: {:.3f} ({:.3f}). '
                    'Metric: {:.4f} ({:.4f}). '
                    'Loss: {:.3f} ({:.3f}). '
                    'Prec: {:.2%} ({:.2%}).'
                    .format(trial + 1, ep, i + 1, min(self.max_steps, len(self.data_loader)),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            precisions.val / losses.val, precisions.avg / losses.avg,
                            losses.val, losses.avg,
                            precisions.val, precisions.avg), end='\r', file=sys.stdout.console)

                if iters == self.max_steps - 1:
                    break

            if iters == self.max_steps - 1:
                break

        loss = losses.avg
        acc = precisions.avg

        print(
            '* Trial %d. Metric: %.4f. Loss: %.3f. Acc: %.2f%%. Training time: %.0f seconds.                                                     \n'
            % (trial + 1, acc / loss, loss, acc * 100, time.time() - start_time))
        return loss, acc

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, model, criterion, inputs, targets):
        feature = model(inputs)
        loss, acc = criterion(feature, targets)
        loss = torch.mean(loss)
        acc = torch.mean(acc)
        return loss, acc
