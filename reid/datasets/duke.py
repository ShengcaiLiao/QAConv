from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import re


class Duke(object):

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
        offset = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
        fps = 59.94

        pattern = re.compile(r'([-\d]+)_c(\d)_f([-\d]+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))

        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam, frame = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            time = (frame + offset[cam]) / fps
            ret.append([fname, pid, cam, time])
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
