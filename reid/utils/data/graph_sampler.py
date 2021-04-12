"""Class for the graph sampler proposed in
    Shengcai Liao and Ling Shao, "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification." In arXiv preprint, arXiv:2104.01546, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.0
        April 1, 2021
    """

from __future__ import absolute_import
from collections import defaultdict
import time
from random import shuffle
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .preprocessor import Preprocessor
from reid.evaluators import extract_features, pairwise_distance, reranking


class GraphSampler(Sampler):
    def __init__(self, data_source, img_path, transformer, model, matcher, batch_size=64, num_instance=4, 
                rerank=True, pre_epochs=1, last_epoch=-1, save_path=None, verbose=False):
        super(GraphSampler, self).__init__(data_source)
        self.data_source = data_source
        self.img_path = img_path
        self.transformer = transformer
        self.model = model
        self.matcher = matcher
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.rerank = rerank
        self.pre_epochs = pre_epochs
        self.last_epoch = last_epoch
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
        self.epoch = last_epoch + 1

    def make_index(self):
        if self.epoch < self.pre_epochs:
            self.random_index()
        else:
            start = time.time()
            self.graph_index()
            if self.verbose:
                print('\t GraphSampler: \tTotal GS time for epoch %d: %.3f seconds.\n' % (self.epoch + 1, time.time() - start))
        self.epoch += 1

    def random_index(self):
        sam_index = []
        for pid in self.pids:
            index = self.index_dic[pid].copy()
            batches = len(index) // self.num_instance
            if batches == 0:
                more = np.random.choice(index, size=self.num_instance-len(index), replace=True)
                index.extend(more)
                batches = 1
            shuffle(index)
            if batches > self.batch_size:
                index = index[: self.batch_size * self.num_instance]
            else:
                index = index[: batches * self.num_instance]
            sam_index.extend(index)
        sam_index = np.array(sam_index)
        sam_index = sam_index.reshape((-1, self.num_instance))
        np.random.shuffle(sam_index)
        sam_index = list(sam_index.flatten())
        self.sam_index = sam_index

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            Preprocessor(dataset, self.img_path, transform=self.transformer),
            batch_size=64, num_workers=8,
            shuffle=False, pin_memory=True)

        if self.verbose:
            print('\t GraphSampler: ', end='\t')
        features, _ = extract_features(self.model, data_loader, self.verbose)
        features = torch.cat([features[fname].unsqueeze(0) for fname, _, _, _ in dataset], 0)

        if self.verbose:
            print('\t GraphSampler: \tCompute distance...', end='\t')
        start = time.time()
        dist = pairwise_distance(self.matcher, features, features)        
        if self.verbose:
            print('Time: %.3f seconds.' % (time.time() - start))

        if self.rerank:
            if self.verbose:
                print('\t GraphSampler: \tRerank...', end='\t')
            start = time.time()
            with torch.no_grad():
                dist = torch.cat((dist, dist))
                dist = torch.cat((dist, dist), dim=1)
                dist = reranking(dist, self.num_pids)
                dist = torch.from_numpy(dist).cuda()
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
