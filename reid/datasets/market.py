from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import re
import numpy as np


class Market(object):

    def __init__(self, root, combine_all=False):
        assert (not combine_all)
        self.images_dir = osp.join(root)
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.has_time_info = True
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append([fname, pid, cam])
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        pattern = re.compile(r's(\d)_([-\d]+)')
        frame_rule = lambda x: map(int, pattern.search(x).groups())
        train_session_id, train_frame_id = map(np.array, zip(*list(map(frame_rule, [f for f, _, _ in self.train]))))
        train_session_id -= 1
        train_cam_id = np.array([cam for _, _, cam in self.train])

        gal_session_id, gal_frame_id = map(np.array, zip(*list(map(frame_rule, [f for f, _, _ in self.gallery]))))
        gal_session_id -= 1
        gal_cam_id = np.array([cam for _, _, cam in self.gallery])

        prob_session_id, prob_frame_id = map(np.array, zip(*list(map(frame_rule, [f for f, _, _ in self.query]))))
        prob_session_id -= 1
        prob_cam_id = np.array([cam for _, _, cam in self.query])

        # For several sessions from each camera, we roughly calculated the overall time frames
        # of each session as offset, and made a cumulative record by assuming the video sessions
        # were continuously recorded.

        offset = np.zeros((6, 6), dtype=gal_frame_id.dtype)
        for c in range(6):
            for s in range(1, 6):
                train_index_max = np.logical_and(train_cam_id == c, train_session_id == s - 1)
                train_frame_max = train_frame_id[train_index_max].max() if train_index_max[
                                                                               train_index_max].size > 0 else 0
                gal_index_max = np.logical_and(gal_cam_id == c, gal_session_id == s - 1)
                gal_frame_max = gal_frame_id[gal_index_max].max() if gal_index_max[gal_index_max].size > 0 else 0
                prob_index_max = np.logical_and(prob_cam_id == c, prob_session_id == s - 1)
                prob_frame_max = prob_frame_id[prob_index_max].max() if prob_index_max[prob_index_max].size > 0 else 0
                offset[c][s] = max(train_frame_max,
                                   gal_frame_max,
                                   prob_frame_max)
                offset[c][s] += offset[c][s - 1]
                if np.logical_and(train_cam_id == c, train_session_id == s).size > 0:
                    train_frame_id[np.logical_and(train_cam_id == c, train_session_id == s)] += offset[c][s]
                if np.logical_and(gal_cam_id == c, gal_session_id == s).size > 0:
                    gal_frame_id[np.logical_and(gal_cam_id == c, gal_session_id == s)] += offset[c][s]
                if np.logical_and(prob_cam_id == c, prob_session_id == s).size > 0:
                    prob_frame_id[np.logical_and(prob_cam_id == c, prob_session_id == s)] += offset[c][s]

        fps = 25.
        train_time = np.true_divide(train_frame_id, fps)
        gal_time = np.true_divide(gal_frame_id, fps)
        prob_time = np.true_divide(prob_frame_id, fps)
        for i in range(len(self.train)):
            self.train[i].append(train_time[i])
        for i in range(len(self.gallery)):
            self.gallery[i].append(gal_time[i])
        for i in range(len(self.query)):
            self.query[i].append(prob_time[i])

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}\n"
              .format(self.num_gallery_ids, len(self.gallery)))
