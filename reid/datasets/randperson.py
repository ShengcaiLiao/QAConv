from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob


class RandPerson(object):

    def __init__(self, root, combine_all=True):

        self.images_dir = osp.join(root)
        self.img_path = 'randperson_subset'
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_ids = 0
        self.has_time_info = False
        self.load()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))

        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            pid = int(fields[0])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = int(fields[2][1:]) - 1  # make it starting from 0
            data.append((fname, pid, camid, 0))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  all    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
