from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from .utils import to_torch

from .evaluation_metrics import cmc, mean_ap
from reid.loss.qaconv import QAConv
from .tlift import TLift
from .rerank import re_ranking


def pre_tlift(gallery, query):
    gal_cam_id = np.array([cam for _, _, cam, _ in gallery])
    gal_time = np.array([frame_time for _, _, _, frame_time in gallery])
    prob_cam_id = np.array([cam for _, _, cam, _ in query])
    prob_time = np.array([frame_time for _, _, _, frame_time in query])

    return {'gal_cam_id': gal_cam_id, 'gal_time': gal_time,
            'prob_cam_id': prob_cam_id, 'prob_time': prob_time,
            'num_cams': gal_cam_id.max() + 1}


def extract_cnn_feature(model, inputs):
    model = model.cuda().eval()
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader):
    fea_time = 0
    data_time = 0
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    print('Extract Features...', end='\t')

    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time += time.time() - end
        end = time.time()

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        fea_time += time.time() - end
        end = time.time()

    print('Feature time: {:.3f} seconds. Data time: {:.3f} seconds.'.format(fea_time, data_time))

    return features, labels


def pairwise_distance(gal_fea, prob_fea, qaconv_layer, gal_batch_size=128,
                      prob_batch_size=4096, transpose=False):
    with torch.no_grad():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_gals, num_probs, device=prob_fea.device)
        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            qaconv = torch.nn.DataParallel(QAConv(prob_fea[i: j, :, :, :].cuda(), qaconv_layer)).cuda().eval()
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score[k: k2, i: j] = qaconv(gal_fea[k: k2, :, :, :].cuda())
        if transpose:
            dist = (1. - score.t()).cpu().numpy()  # [p, g]
        else:
            dist = (1. - score).cpu().numpy()  # [g, p]
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, testset, qaconv_layer, gal_batch_size=128,
                 prob_batch_size=4096, tau=100, sigma=100, K=100, alpha=0.1):
        query = testset.query
        gallery = testset.gallery
        prob_fea, _ = extract_features(self.model, query_loader)
        prob_fea = torch.cat([prob_fea[f].unsqueeze(0) for f, _, _, _ in query], 0)
        gal_fea, _ = extract_features(self.model, gallery_loader)
        gal_fea = torch.cat([gal_fea[f].unsqueeze(0) for f, _, _, _ in gallery], 0)

        print('Compute similarity...', end='\t')
        start = time.time()
        dist = pairwise_distance(gal_fea, prob_fea, qaconv_layer, gal_batch_size, prob_batch_size,
                                 transpose=True)  # [p, g]
        print('Time: %.3f seconds.' % (time.time() - start))
        rank1, mAP = evaluate_all(dist, query=query, gallery=gallery)

        print('Compute similarity for rerank...', end='\t')
        start = time.time()
        q_q_dist = pairwise_distance(prob_fea, prob_fea, qaconv_layer, gal_batch_size, prob_batch_size)
        g_g_dist = pairwise_distance(gal_fea, gal_fea, qaconv_layer, gal_batch_size, prob_batch_size)
        dist_rerank = re_ranking(dist, q_q_dist, g_g_dist)
        print('Time: %.3f seconds.' % (time.time() - start))
        rank1_rerank, mAP_rerank = evaluate_all(dist_rerank, query=query, gallery=gallery)
        score_rerank = 1 - dist_rerank

        if testset.has_time_info:
            print('Compute TLift...', end='\t')
            start = time.time()
            pre_tlift_dict = pre_tlift(gallery, query)
            score_tlift = TLift(score_rerank, tau=tau, sigma=sigma, K=K, alpha=alpha,
                                **pre_tlift_dict)
            print('Time: %.3f seconds.' % (time.time() - start))
            dist_tlift = 1 - score_tlift
            rank1_tlift, mAP_tlift = evaluate_all(dist_tlift, query=query, gallery=gallery)
        else:
            pre_tlift_dict = {'gal_time': 0, 'prob_time': 0}
            dist_tlift = 0
            rank1_tlift = rank1_rerank
            mAP_tlift = mAP_rerank

        return rank1, mAP, rank1_rerank, mAP_rerank, rank1_tlift, mAP_tlift, dist, dist_rerank, \
               dist_tlift, pre_tlift_dict
