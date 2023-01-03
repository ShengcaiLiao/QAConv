"""Class for the graph sampler proposed in
    Shengcai Liao and Ling Shao, "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification." In arXiv preprint, arXiv:2104.01546, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.2
        Jan 3, 2023
    """

from __future__ import absolute_import
from collections import defaultdict
import time
from random import shuffle
import numpy as np
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .preprocessor import Preprocessor
from reid.evaluators import extract_features, pairwise_distance


class GraphSampler(Sampler):
    def __init__(self, data_source, img_path, transformer, model, matcher, batch_size=64, num_instance=4, 
                gal_batch_size=256, prob_batch_size=256, save_path=None, verbose=False):
        super(GraphSampler, self).__init__(data_source)
        self.data_source = data_source
        self.img_path = img_path
        self.transformer = transformer
        self.model = model
        self.matcher = matcher
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.gal_batch_size = gal_batch_size
        self.prob_batch_size = prob_batch_size
        self.save_path = save_path
        self.verbose = verbose

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)
        for pid in self.pids:
            shuffle(self.index_dic[pid])

        self.sam_index = None
        self.sam_pointer = [0] * self.num_pids

    def make_index(self):
        start = time.time()
        self.graph_index()
        if self.verbose:
            print('\nTotal GS time: %.3f seconds.\n' % (time.time() - start))

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            Preprocessor(dataset, self.img_path, transform=self.transformer),
            batch_size=256, num_workers=8,
            shuffle=False, pin_memory=True)

        if self.verbose:
            print('\t GraphSampler: ', end='\t')
        model = deepcopy(self.model).half().cuda().eval()
        num_features = self.matcher.num_features
        fea_height = self.matcher.height
        fea_width = self.matcher.width
        features = extract_features(model, data_loader, num_features, fea_height, fea_width, self.verbose)

        if self.verbose:
            print('\t GraphSampler: \tCompute distance...', end='\t')
        start = time.time()
        matcher = deepcopy(self.matcher).half().cuda().eval()
        dist = pairwise_distance(matcher, features, features, self.gal_batch_size, self.prob_batch_size, self.verbose)
        
        if self.verbose:
            print('Time: %.3f seconds.' % (time.time() - start))

        return dist

    def graph_index(self):
        sam_index = []
        for pid in self.pids:
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)
        
        dataset = [self.data_source[i] for i in sam_index]
        dist = self.calc_distance(dataset)

        with torch.no_grad():
            dist = dist + torch.eye(self.num_pids, device=dist.device) * 1e15
            topk = self.batch_size // self.num_instance - 1
            _, topk_index = torch.topk(dist.cuda(), topk, largest=False)
            topk_index = topk_index.cpu().numpy()            

        if self.save_path is not None:
            filenames = [fname for fname, _, _, _ in dataset]
            test_file = os.path.join(self.save_path, 'gs%d.npz' % self.epoch)
            np.savez_compressed(test_file, filenames=filenames, dist=dist.cpu().numpy(), topk_index=topk_index)

        sam_index = []
        for i in range(self.num_pids):
            id_index = topk_index[i, :].tolist()
            id_index.append(i)
            index = []
            for j in id_index:
                pid = self.pids[j]
                img_index = self.index_dic[pid]
                len_p = len(img_index)
                index_p = []
                remain = self.num_instance
                while remain > 0:
                    end = self.sam_pointer[j] + remain
                    idx = img_index[self.sam_pointer[j] : end]
                    index_p.extend(idx)
                    remain -= len(idx)
                    self.sam_pointer[j] = end
                    if end >= len_p:
                        shuffle(img_index)
                        self.sam_pointer[j] = 0
                assert(len(index_p) == self.num_instance)
                index.extend(index_p)
            sam_index.extend(index)

        sam_index = np.array(sam_index)
        sam_index = sam_index.reshape((-1, self.batch_size))
        np.random.shuffle(sam_index)
        sam_index = list(sam_index.flatten())
        self.sam_index = sam_index

    def __len__(self):
        if self.sam_index is None:
            return self.num_pids
        else:
            return len(self.sam_index)

    def __iter__(self):
        self.make_index()
        return iter(self.sam_index)
