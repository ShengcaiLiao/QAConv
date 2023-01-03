from __future__ import print_function, absolute_import
import sys
import time

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .tlift import TLift


def pre_tlift(gallery, query):
    gal_cam_id = np.array([cam for _, _, cam, _ in gallery])
    gal_time = np.array([frame_time for _, _, _, frame_time in gallery])
    prob_cam_id = np.array([cam for _, _, cam, _ in query])
    prob_time = np.array([frame_time for _, _, _, frame_time in query])

    return {'gal_cam_id': gal_cam_id, 'gal_time': gal_time,
            'prob_cam_id': prob_cam_id, 'prob_time': prob_time,
            'num_cams': gal_cam_id.max() + 1}


def extract_cnn_feature(model, inputs):
    with torch.inference_mode():
        outputs = model(inputs.half().cuda())
    return outputs


def extract_features(model, data_loader, num_features, fea_height, fea_width, verbose=False):
    fea_time = 0
    data_time = 0
    end = time.time()

    if verbose:
        print('Extract Features...', end='\t')

    shape = [len(data_loader.dataset), num_features, fea_height, fea_width]
    features = torch.zeros(shape, dtype=torch.half, device='cuda:0')

    for i, (imgs, _, _, _) in enumerate(data_loader):
        data_time += time.time() - end
        end = time.time()

        k0 = i * data_loader.batch_size
        k1 = k0 + len(imgs)
        features[k0 : k1, :, :, :] = extract_cnn_feature(model, imgs)

        fea_time += time.time() - end
        end = time.time()

    if verbose:
        print('Feature time: {:.3f} seconds. Data time: {:.3f} seconds.'.format(fea_time, data_time))

    return features


def pairwise_distance(matcher, prob_fea, gal_fea, gal_batch_size=128, prob_batch_size=128, verbose=False):
    with torch.inference_mode():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_probs, num_gals, dtype=torch.half, device=prob_fea.device)

        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            if verbose:
                print('Query batch %d / %d ... \t' % (i // prob_batch_size + 1, num_probs // prob_batch_size + 1), end='\r', file=sys.stdout.console)

            pfea = prob_fea[i: j, :, :, :]
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score[i: j, k: k2] = matcher(pfea, gal_fea[k: k2, :, :, :])
        # # scale matching scores to make them visually more recognizable
        # score = torch.sigmoid(score / 10)
        # dist = 1. - score
        dist = -score
        dist = dist.cpu().float()
    return dist  # [p, g]


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


def reranking(dist, query_num, k1=20, k2=6, lamda_value=0.3, verbose=False):
    original_dist = dist.numpy()
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    if verbose:
        print('starting re_ranking...', end='\t')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class Evaluator(object):
    def __init__(self, model, matcher, test_gal_batch=128, test_prob_batch=128, 
            tau=100, sigma=200, K=10, alpha=0.2):
        super(Evaluator, self).__init__()
        self.num_features = matcher.num_features
        self.fea_height = matcher.height
        self.fea_width = matcher.width
        self.test_gal_batch = test_gal_batch
        self.test_prob_batch = test_prob_batch
        # parameters for TLift:
        self.tau = tau
        self.sigma = sigma
        self.K = K
        self.alpha = alpha
        
        self.model = model.half().cuda().eval()
        self.matcher = matcher.half().cuda().eval()

    def evaluate(self, testset, query_loader, gallery_loader):
        query = testset.query
        gallery = testset.gallery

        prob_fea = extract_features(self.model, query_loader, self.num_features, self.fea_height, self.fea_width)
        gal_fea = extract_features(self.model, gallery_loader, self.num_features, self.fea_height, self.fea_width)

        print('Compute similarity ...', end='\t')
        start = time.time()
        dist = pairwise_distance(self.matcher, prob_fea, gal_fea, self.test_gal_batch, self.test_prob_batch)
        print('Time: %.3f seconds.' % (time.time() - start))
        rank1, mAP = evaluate_all(dist, query=query, gallery=gallery)

        if testset.has_time_info:
            num_prob = len(query)
            num_gal = len(gallery)
            num_all = num_gal + num_prob
            dist_rerank = torch.zeros(num_all, num_all)
            print('Compute similarity for rerank...', end='\t')
            start = time.time()

            with torch.no_grad():
                dist_rerank[:num_prob, num_prob:] = dist
                dist_rerank[num_prob:, :num_prob] = dist.t()
                dist_rerank[:num_prob, :num_prob] = pairwise_distance(self.matcher, prob_fea, prob_fea, self.test_gal_batch,
                                                                    self.test_prob_batch)
                dist_rerank[num_prob:, num_prob:] = pairwise_distance(self.matcher, gal_fea, gal_fea, self.test_gal_batch,
                                                                    self.test_prob_batch)

            dist_rerank = reranking(dist_rerank, num_prob, verbose=True)
            print('Time: %.3f seconds.' % (time.time() - start))
            rank1_rerank, mAP_rerank = evaluate_all(dist_rerank, query=query, gallery=gallery)
            score_rerank = 1 - dist_rerank

            print('Compute TLift...', end='\t')
            start = time.time()
            pre_tlift_dict = pre_tlift(gallery, query)
            score_tlift = TLift(score_rerank, tau=self.tau, sigma=self.sigma, K=self.K, alpha=self.alpha,
                                **pre_tlift_dict)
            print('Time: %.3f seconds.' % (time.time() - start))
            dist_tlift = 1 - score_tlift
            rank1_tlift, mAP_tlift = evaluate_all(dist_tlift, query=query, gallery=gallery)
        else:
            pre_tlift_dict = {'gal_time': 0, 'prob_time': 0}
            dist_tlift = 0
            dist_rerank = 0
            rank1_rerank = 0
            mAP_rerank = 0
            rank1_tlift = 0
            mAP_tlift = 0

        return rank1, mAP, rank1_rerank, mAP_rerank, rank1_tlift, mAP_tlift, dist.numpy(), dist_rerank, \
               dist_tlift, pre_tlift_dict
