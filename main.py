from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import string
import time

import torch
from torch.backends import cudnn
import numpy as np
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
             min_size=0., max_size=0.8, workers=8, test_batch=64):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=combine_all)

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((height, width), interpolation=3),
        T.RandomOcclusion(min_size, max_size),
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
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=32,
        shuffle=False, pin_memory=True)

    return dataset, query_loader, gallery_loader


def main(args):
    cudnn.deterministic = False
    cudnn.benchmark = True

    exp_database_dir = osp.join(args.exp_dir, string.capwords(args.dataset))
    output_dir = osp.join(exp_database_dir, args.method, args.sub_method)
    log_file = osp.join(output_dir, 'log.txt')
    # Redirect print to both console and log file
    sys.stdout = Logger(log_file)

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
            loss, acc = trainer.train(epoch, train_loader, optimizer)

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
    print('Evaluate the learned model:')
    t0 = time.time()

    # Evaluator
    evaluator = Evaluator(model)

    test_names = args.testset.strip().split(',')
    for test_name in test_names:
        if test_name not in datasets.names():
            print('Unknown dataset: %s.' % test_name)
            continue

        testset, test_query_loader, test_gallery_loader = \
            get_test_data(test_name, args.data_dir, args.height, args.width, args.test_fea_batch)

        test_rank1, test_mAP, test_rank1_rerank, test_mAP_rerank, test_rank1_tlift, test_mAP_tlift, test_dist, \
        test_dist_rerank, test_dist_tlift, pre_tlift_dict = \
            evaluator.evaluate(test_query_loader, test_gallery_loader, testset, criterion.module,
                               args.test_gal_batch, args.test_prob_batch,
                               args.tau, args.sigma, args.K, args.alpha)

        print('  %s: rank1=%.1f, mAP=%.1f, rank1_rerank=%.1f, mAP_rerank=%.1f,'
              ' rank1_rerank_tlift=%.1f, mAP_rerank_tlift=%.1f.\n'
              % (test_name, test_rank1 * 100, test_mAP * 100, test_rank1_rerank * 100, test_mAP_rerank * 100,
                 test_rank1_tlift * 100, test_mAP_tlift * 100))

        result_file = osp.join(exp_database_dir, args.method, test_name + '_results.txt')
        with open(result_file, 'a') as f:
            f.write('%s/%s:\n' % (args.method, args.sub_method))
            f.write('\t%s: rank1=%.1f, mAP=%.1f, rank1_rerank=%.1f, mAP_rerank=%.1f, rank1_rerank_tlift=%.1f, '
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
                                          'gal_time': pre_tlift_dict['gal_time'],
                                          'prob_time': pre_tlift_dict['prob_time'],
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
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market', choices=datasets.names(),
                        help="the training dataset")
    parser.add_argument('--combine_all', action='store_true', default=False,
                        help="combine all data for training, default: False")
    parser.add_argument('--testset', type=str, default='duke,market', help="the test datasets")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="the batch size, default: 32")
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help="the number of workers for the dataloader, default: 8")
    parser.add_argument('--height', type=int, default=384, help="height of the input image, default: 384")
    parser.add_argument('--width', type=int, default=128, help="width of the input image, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=resmap.names(),
                        help="the backbone network, default: resnet50")
    parser.add_argument('--final_layer', type=str, default='layer3', choices=['layer2', 'layer3', 'layer4'],
                        help="the final layer, default: layer3")
    parser.add_argument('--neck', type=int, default=128,
                        help="number of channels for the final neck layer, default: 128")
    # TLift
    parser.add_argument('--tau', type=float, default=100,
                        help="the interval threshold to define nearby persons in TLift, default: 100")
    parser.add_argument('--sigma', type=float, default=200,
                        help="the sensitivity parameter of the time difference in TLift, default: 200")
    parser.add_argument('--K', type=int, default=10,
                        help="parameter of the top K retrievals used to define the pivot set P in TLift, "
                             "default: 10")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="regularizer for the multiplication fusion in TLift, default: 0.2")

    # random occlusion
    parser.add_argument('--min_size', type=float, default=0, help="minimal size for the random occlusion, default: 0")
    parser.add_argument('--max_size', type=float, default=0.8,
                        help="maximal size for the ramdom occlusion. default: 0.8")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate of the new parameters. For pretrained "
                             "parameters it is 10 times smaller than this. Default: 0.01.")
    # training configurations
    parser.add_argument('--epochs', type=int, default=60, help="the number of training epochs, default: 60")
    parser.add_argument('--step_size', type=int, default=40, help="step size for the learning rate decay, default: 40")
    parser.add_argument('--mem_batch_size', type=int, default=16,
                        help="Batch size for the convolution with the class memory in QAConvLoss. Default: 16."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help="Path for resuming training. Choices: '' (new start, default), "
                             "'ori' (original path), or a real path")
    # test configurations
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only, default: False")
    parser.add_argument('--test_fea_batch', type=int, default=64,
                        help="Feature extraction batch size during testing. Default: 64."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_gal_batch', type=int, default=128,
                        help="QAConv gallery batch size during testing. Default: 128."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=4096,
                        help="QAConv probe batch size (as kernel) during testing. Default: 4096."
                             "Reduce this if you encounter a GPU memory overflow.")
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'),
                        help="the path to the image data")
    parser.add_argument('--exp-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'Exp'),
                        help="the path to the output directory")
    parser.add_argument('--method', type=str, default='QAConv', help="method name for the output directory")
    parser.add_argument('--sub_method', type=str, default='res50_layer3',
                        help="sub method name for the output directory")
    parser.add_argument('--save_score', default=False, action='store_true',
                        help="save the matching score or not, default: False")

    main(parser.parse_args())
