from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import re


class ClonedPerson(object):

    def __init__(self, root, combine_all=False):

        self.images_dir = osp.join(root)
        self.train_path = 'train'
        self.gallery_path = 'test/gallery'
        self.query_path = 'test/query'
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.has_time_info = True
        self.load()
        if combine_all:
            raise Exception('combine_all should be False')

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_c([-\d]+)_f([-\d]+)')
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*g')))

        data = []
        all_pids = {}
        camera_offset = [0, 0, 0, 4, 4, 8, 12, 12, 12, 12, 16, 16, 20]
        fps = 24

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            pid, scene, cam, frame = map(int, pattern.search(fname).groups())
            scene = 2
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            if relabel:
                pid = all_pids[pid]
            camid = camera_offset[scene] + cam  # make it starting from 0
            time = frame / fps 
            data.append((fname, pid, camid, time))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path, True)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}\n"
              .format(self.num_gallery_ids, len(self.gallery)))