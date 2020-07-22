from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import time
import torch
import numpy as np
import scipy.io as sio

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from reid import datasets
from reid.models import resmap
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint
from reid.evaluators import extract_features
from reid.loss.qaconv_loss import QAConvLoss
from qaconv_match import QAConvMatch


def get_test_data(dataname, data_dir, height, width, test_batch=64):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=False)

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=0,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=0,
        shuffle=False, pin_memory=True)

    return dataset, query_loader, gallery_loader


def main(args):
    cudnn.benchmark = True

    num_classes = 4101  # in training set
    start_prob = 0
    end_prob = 1000

    # Create data loaders
    dataset, query_loader, gallery_loader = get_test_data(args.testset, args.data_dir, args.height, args.width,
                                                          args.test_fea_batch)

    # Create model
    model = resmap.create(args.arch, final_layer=args.final_layer, neck=args.neck).cuda()
    num_features = model.num_features

    # Criterion
    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]
    criterion = QAConvLoss(num_classes, num_features, hei, wid, args.mem_batch_size).cuda()

    print('Loading checkpoint...')
    checkpoint = load_checkpoint(osp.join(args.exp_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    criterion.load_state_dict(checkpoint['criterion'])

    model = nn.DataParallel(model).cuda()

    # Final test
    print('Evaluate the learned model:')
    t0 = time.time()

    feature, _ = extract_features(model, query_loader)
    feature = torch.cat([feature[f].unsqueeze(0) for f, _, _, _ in dataset.query], 0)

    num_probs = end_prob - start_prob
    feature = feature[start_prob: end_prob]
    feature = feature.cuda()

    with torch.no_grad():
        score = torch.zeros(num_probs, num_probs)
        prob_score = torch.zeros(num_probs, num_probs, hei, wid)
        index_in_gal = torch.zeros(num_probs, num_probs, hei, wid)
        gal_score = torch.zeros(num_probs, num_probs, hei, wid)
        index_in_prob = torch.zeros(num_probs, num_probs, hei, wid)
        qaconv = torch.nn.DataParallel(QAConvMatch(feature, criterion).cuda()).cuda().eval()
        batch_size = args.test_prob_batch

        for i in range(0, num_probs, batch_size):
            end = min(num_probs, i + batch_size)
            s, ps, ig, gs, ip = qaconv(feature[i: end])
            score[i: end, :] = s
            prob_score[i: end, :, :, :] = ps
            index_in_gal[i: end, :, :, :] = ig
            gal_score[i: end, :, :, :] = gs
            index_in_prob[i: end, :, :, :] = ip
            if ((i + 1) // batch_size) % 100 == 0:
                print('Compute similarity: [{}/{}]. Min score: {}. Max score: {}. Avg score: {}.'
                      .format(i, num_probs, s.min(), s.max(), s.mean()))

    test_prob_list = np.array([fname for fname, _, _, _ in dataset.query], dtype=np.object)
    test_prob_ids = [pid for _, pid, _, _ in dataset.query]
    test_prob_cams = [cam for _, _, cam, _ in dataset.query]
    test_prob_list = test_prob_list[start_prob: end_prob]
    test_prob_ids = test_prob_ids[start_prob: end_prob]
    test_prob_cams = test_prob_cams[start_prob: end_prob]
    test_score_file = osp.join(args.exp_dir, '%s_query_score_%d-%d.mat' % (args.testset, start_prob, end_prob))
    weight = criterion.fc.weight.view(2, hei, wid).detach().cpu()
    sio.savemat(test_score_file, {'fc': weight.numpy(),
                                  'score': score.numpy(),
                                  'prob_score': prob_score.numpy(),
                                  'index_in_gal': index_in_gal.numpy(),
                                  'gal_score': gal_score.numpy(),
                                  'index_in_prob': index_in_prob.numpy(),
                                  'prob_list': test_prob_list,
                                  'prob_ids': test_prob_ids,
                                  'prob_cams': test_prob_cams},
                oned_as='column',
                do_compression=True)

    test_time = time.time() - t0
    print("Total testing time: %.3f sec.\n" % test_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv")
    # data
    parser.add_argument('--testset', type=str, default='market')
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 384")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=resmap.names())
    parser.add_argument('--final_layer', type=str, default='layer3')
    parser.add_argument('--neck', type=int, default=128,
                        help="number of bottle neck channels, default: 128")
    # test configs
    parser.add_argument('--mem_batch_size', type=int, default=16,
                        help='Batch size for the convolution with the class memory in QAConvLoss. '
                             'Reduce this if you encounter a gpu memory overflow.')
    parser.add_argument('--test_fea_batch', type=int, default=64, help="Feature extraction batch size during testing. "
                                                                        "Reduce this if you encounter a gpu memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=4,
                        help="QAConv probe batch size (as kernel) during testing. Reduce this "
                             "if you encounter a gpu memory overflow.")
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--exp-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'Exp'))

    main(parser.parse_args())
