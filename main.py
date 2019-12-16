from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import sys
import string
import time

import torch
from torch.backends import cudnn
import numpy as np
import random
import scipy.io as sio

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from reid import datasets
from reid.models import resmap
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.loss.qaconv_loss import QAConvLoss


def get_data(dataname, data_dir, height, width, batch_size, combine_all=False,
                 min_size=0., max_size=0.8, workers=8, test_batch=2048):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=combine_all)

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((height, width), interpolation=3),
        T.RandomBlock(min_size, max_size),
        T.ToTensor(),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir, dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader


def get_test_data(dataname, data_dir, height, width, test_batch=2048):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=False)

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    return dataset, query_loader, gallery_loader

def set_seed(seed):
    if seed < 0:
        seed = random.randint(0, 10000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return seed


def main(args):
    exp_database_dir = osp.join(args.exp_dir, string.capwords(args.dataset))
    output_dir = osp.join(exp_database_dir, args.method, args.sub_method)
    log_file = osp.join(output_dir, 'log.txt')
    # Redirect print to both console and log file
    sys.stdout = Logger(log_file)

    seed = set_seed(args.seed)
    print('Random seed of this run: %d\n' % seed)

    # Create data loaders
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.combine_all,
                 args.min_size, args.max_size, args.workers, args.test_fea_batch)

    # Create model
    model = resmap.create(args.arch, final_layer=args.final_layer, neck=args.neck).cuda()
    num_features = model.num_features
    # print(model)
    print('\n')

    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')

    # Criterion

    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]
    criterion = QAConvLoss(num_classes, num_features, hei, wid, args.mem_batch_size).cuda()

    # Optimizer
    base_param_ids = set(map(id, model.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': model.base.parameters(), 'lr': 0.1 * args.lr},
        {'params': new_params, 'lr': args.lr},
        {'params': criterion.parameters(), 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Decay LR by a factor of 0.1 every step_size epochs
    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Load from checkpoint
    start_epoch = 0

    if args.resume or args.evaluate:
        print('Loading checkpoint...')
        if args.resume and (args.resume != 'ori'):
            checkpoint = load_checkpoint(args.resume)
        else:
            checkpoint = load_checkpoint(osp.join(output_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model'])
        criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))

    model = nn.DataParallel(model).cuda()
    criterion = nn.DataParallel(criterion).cuda()

    if not args.evaluate:
        # Trainer
        trainer = Trainer(model, criterion)

        t0 = time.time()
        # Start training
        for epoch in range(start_epoch, args.epochs):
            loss, acc = trainer.train(epoch, train_loader, optimizer, args.print_freq)

            lr = list(map(lambda group: group['lr'], optimizer.param_groups))
            lr_scheduler.step(epoch + 1)
            train_time = time.time() - t0

            print(
                '* Finished epoch %d at lr=[%g, %g, %g]. Loss: %.3f. Acc: %.2f%%. Training time: %.0f seconds.\n'
                % (epoch + 1, lr[0], lr[1], lr[2], loss, acc * 100, train_time))

            save_checkpoint({
                'model': model.module.state_dict(),
                'criterion': criterion.module.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(output_dir, 'checkpoint.pth.tar'))

    # Final test
    cudnn.benchmark = True
    print('Evaluate the learned model:')
    t0 = time.time()

    # Evaluator
    evaluator = Evaluator(model)

    test_names = args.testset.strip().split(',')
    for test_name in test_names:
        if test_name not in datasets.names():
            print('Unknown dataset: {test_name}.')
            continue

        testset, test_query_loader, test_gallery_loader = \
            get_test_data(test_name, args.data_dir, args.height, args.width, args.test_fea_batch)

        test_rank1, test_mAP, test_rank1_rerank, test_mAP_rerank, test_rank1_tlift, test_mAP_tlift, test_dist, \
            test_dist_rerank, test_dist_tlift, pre_tlift_dict = \
            evaluator.evaluate(test_query_loader, test_gallery_loader, testset, criterion.module, args.test_gal_batch,
                               args.test_prob_batch)

        print('  %s: rank1=%.1f, mAP=%.1f, rank1_rerank=%.1f, mAP_rerank=%.1f,'
              ' rank1_rerank_tlift=%.1f, mAP_rerank_tlift=%.1f.\n'
              % (test_name, test_rank1 * 100, test_mAP * 100, test_rank1_rerank * 100, test_mAP_rerank * 100,
                 test_rank1_tlift * 100, test_mAP_tlift * 100))

        result_file = osp.join(exp_database_dir, args.method, test_name + '_results.txt')
        with open(result_file, 'a') as f:
            f.write('%s/%s:\n' % (args.method, args.sub_method))
            f.write('\t%s: rank1=%.1f, mAP=%.1f, rank1_rerank=%.1f, mAP_rerank=%.1f rank1_rerank_tlift=%.1f, '
                    'mAP_rerank_tlift=%.1f.\n\n'
                    % (test_name, test_rank1 * 100, test_mAP * 100, test_rank1_rerank * 100, test_mAP_rerank * 100,
                       test_rank1_tlift * 100, test_mAP_tlift * 100))

        if args.save_score:
            test_gal_list = np.array([fname for fname, _, _, _ in testset.gallery], dtype=np.object)
            test_prob_list = np.array([fname for fname, _, _, _ in testset.query], dtype=np.object)
            test_gal_ids = [pid for _, pid, _, _ in testset.gallery]
            test_prob_ids = [pid for _, pid, _, _ in testset.query]
            test_gal_cams = [c for _, _, c, _ in testset.gallery]
            test_prob_cams = [c for _, _, c, _ in testset.query]
            test_score_file = osp.join(exp_database_dir, args.method, args.sub_method, '%s_score.mat' % test_name)
            sio.savemat(test_score_file, {'score': 1. - test_dist,
                                          'score_rerank': 1. - test_dist_rerank,
                                          'score_tlift': 1. - test_dist_tlift,
                                          'gal_time': pre_tlift_dict['gal_time'], 'prob_time': pre_tlift_dict['prob_time'],
                                          'gal_list': test_gal_list, 'prob_list': test_prob_list,
                                          'gal_ids': test_gal_ids, 'prob_ids': test_prob_ids,
                                          'gal_cams': test_gal_cams, 'prob_cams': test_prob_cams},
                        oned_as='column',
                        do_compression=True)

    test_time = time.time() - t0
    if not args.evaluate:
        print('Finished training at epoch %d, loss %.3f, acc %.2f%%.\n'
              % (epoch + 1, loss, acc * 100))
        print("Total training time: %.3f sec. Average training time per epoch: %.3f sec." % (
            train_time, train_time / (args.epochs - start_epoch + 1)))
    print("Total testing time: %.3f sec.\n" % test_time)

    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv")
    parser.add_argument('-s', '--seed', type=int, default=-1, help="random seed for training, default: -1 (automatic)")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market', choices=datasets.names())
    parser.add_argument('--combine_all', action='store_true', default=False, help="combine all data for training")
    parser.add_argument('--testset', type=str, default='duke,market')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 384")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet152',
                        choices=resmap.names())
    parser.add_argument('--final_layer', type=str, default='layer3')
    parser.add_argument('--neck', type=int, default=128,
                        help="number of bottle neck channels, default: 128")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters. For pretrained "
                             "parameters it is 10 times smaller than this.")
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help="path for resume training. "
                                                                               "Choices: '' (new start), 'ori' (original"
                                                                               "path), or a real path")
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--mem_batch_size', type=int, default=2048, help='Batch size for the convolution with the class memory in QAConvLoss. '
                                                                         'Reduce this if you encounter a gpu memory overflow.')
    parser.add_argument('--test_fea_batch', type=int, default=2048, help="Feature extraction batch size during testing. "
                                                                     "Reduce this if you encounter a gpu memory overflow.")
    parser.add_argument('--test_gal_batch', type=int, default=16, help="QAConv gallery batch size during testing. Reduce this "
                                                                     "if you encounter a gpu memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=4096, help="QAConv probe batch size (as kernel) during testing. Reduce this "
                                                                     "if you encounter a gpu memory overflow.")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--print-freq', type=int, default=1)
    # random block
    parser.add_argument('--min_size', type=float, default=0)
    parser.add_argument('--max_size', type=float, default=0.8)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--exp-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'Exp'))
    parser.add_argument('--method', type=str, default='QAConv')
    parser.add_argument('--sub_method', type=str, default='res152_layer3')
    parser.add_argument('--save_score', default=False, action='store_true', help="save the matching score or not")

    main(parser.parse_args())
