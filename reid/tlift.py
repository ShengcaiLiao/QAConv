import numpy as np
import torch
import math


def TLift(in_score, gal_cam_id, gal_time, prob_cam_id, prob_time, num_cams, tau=100, sigma=200, K=10, alpha=0.2):
    """Function for the Temporal Lifting (TLift) method
    TLift is a model-free temporal cooccurrence based score weighting method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Inputs:
        in_score: the similarity score of size [num_probs, num_gals] between the gallery and probe sets.
        gal_cam_id: camera index for samples in the gallery set, starting from 0 and continuously numbered.
        gal_time: time stamps of samples in the gallery set.
        prob_cam_id: camera index for samples in the probe set, starting from 0 and continuously numbered.
        prob_time: time stamps of samples in the probe set.
        num_cams: the number of cameras.
        tau: the interval threshold to define nearby persons. Default: 100.
        sigma: the sensitivity parameter of the time difference. Default: 200.
        K: parameter of the top K retrievals used to define the pivot set P. Default: 10.
        alpha: regularizer for the multiplication fusion. Default: 0.2.
        All the cam_id and time inputs are 1-dim vectors, and they are in the same order corresponding to
        the first axis (probe) or second axis (gallery) of the in_score.
    Outputs:
        out_score: the refined score by TLift, with the same size as the in_score.
    Comments:
        The default alpha value works for the sigmoid or re-ranking matching scores. Otherwise, it is
        suggested that your input scores are distributed in [0, 1], with an average score around 0.01-0.1
        considering many negative matching pairs. To apply TLift directly on QAConv scores, please use
        score = torch.sigmoid(score) instead of the scaled scores in qaconv.py.
    Author:
        Shengcai Liao, reimplemented by Jinchuan Xiao
        scliao@ieee.org
    Version:
        V1.1
        July 12, 2020
    """

    out_score = torch.tensor(np.zeros_like(in_score))
    if torch.cuda.is_available():
        out_score = out_score.cuda()

    if len(prob_time.shape) == 1:
        prob_time = prob_time[np.newaxis, :]
    prob_time_diff = prob_time - np.transpose(prob_time)
    cooccur_mask = (abs(prob_time_diff) < tau)

    g_sam_index = []
    score = []
    gal_time_diff = []

    for g_cam in range(num_cams):
        g_sam_index.append(np.where(gal_cam_id == g_cam)[0])  # camera id starting with 0.
        score.append(in_score[:, g_sam_index[g_cam]])
        frame_id = gal_time[g_sam_index[g_cam]]
        if len(frame_id.shape) == 1:
            frame_id = frame_id[np.newaxis, :]
        gal_time_diff.append(
            torch.tensor(frame_id - np.transpose(frame_id), dtype=out_score.dtype).to(out_score.device))

    for p_cam in range(num_cams):
        p_sam_index = np.where(prob_cam_id == p_cam)[0]
        c_mask = cooccur_mask[p_sam_index][:, p_sam_index]
        num_prob = len(p_sam_index)
        for g_cam in range(num_cams):
            # if p_cam == g_cam:  # in some public datasets they still evaluate negative pairs in the same camera
            #     continue
            prob_score = score[g_cam][p_sam_index, :]
            for i in range(num_prob):
                cooccur_index = np.where(c_mask[:, i] == True)[0]
                cooccur_score = prob_score[cooccur_index, :]
                sorted_score = np.sort(cooccur_score, axis=None)
                if sorted_score.shape[0] > K:
                    thr = sorted_score[-K]
                else:
                    thr = sorted_score[0]
                mask_in_gal = np.where(cooccur_score >= thr)[1]
                dt = gal_time_diff[g_cam][:, mask_in_gal]
                weight = torch.mean(torch.exp(-1 * torch.pow(dt, 2).to(dtype=out_score.dtype) / math.pow(sigma, 2)),
                                    dim=1)
                out_score[p_sam_index[i], g_sam_index[g_cam]] = weight

    out_score = out_score.cpu().numpy()
    out_score = (out_score + alpha) * in_score
    return out_score


if __name__ == '__main__':
    in_score = np.random.randn(50, 100)
    gal_cam_id = np.random.randint(0, 5, (100))
    gal_time = np.random.randint(0, 20, (100))
    prob_cam_id = np.random.randint(0, 5, (50))
    prob_time = np.random.randint(0, 20, (50))
    num_cams = 5
    TLift(in_score, gal_cam_id, gal_time, prob_cam_id, prob_time, num_cams)
