from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import string
import time
import json

import torch
from torch.backends import cudnn
import numpy as np
import scipy.io as sio

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from reid import datasets
from reid.models import resmap
from reid.models.qaconv import QAConv
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils.data.graph_sampler import GraphSampler
from reid.loss.triplet_loss import TripletLoss


def get_data(dataname, data_dir, model, matcher, save_path, args):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=args.combine_all)

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5), 
        T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.RandomOcclusion(args.min_size, args.max_size),
        T.ToTensor(),
    ])

    test_transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    train_path = osp.join(dataset.images_dir, dataset.train_path)
    train_loader = DataLoader(
        Preprocessor(dataset.train, root=train_path, transform=train_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        sampler=GraphSampler(dataset.train, train_path, test_transformer, model, matcher, args.batch_size, args.num_instance,
                    args.test_gal_batch, args.test_prob_batch, save_path, args.gs_verbose),
        pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=args.test_fea_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=args.test_fea_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader


def get_test_data(dataname, data_dir, height, width, workers=8, test_batch=64):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=False)

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
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

    # Create model
    ibn_type = args.ibn
    if ibn_type == 'none':
        ibn_type = None
    model = resmap.create(args.arch, ibn_type=ibn_type, final_layer=args.final_layer, neck=args.neck).cuda()
    num_features = model.num_features
    # print(model)
    # print('\n')

    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]
    matcher = QAConv(num_features, hei, wid).cuda()

    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')

    # Criterion
    criterion = TripletLoss(matcher, args.margin).cuda()

    # Optimizer
    base_param_ids = set(map(id, model.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': model.base.parameters(), 'lr': 0.1 * args.lr},
        {'params': new_params, 'lr': args.lr},
        {'params': matcher.parameters(), 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Load from checkpoint
    start_epoch = 0
    base_loss = None
    final_epochs = args.max_epochs
    lr_stepped = False

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
        base_loss = checkpoint['base_loss']
        final_epochs = checkpoint['final_epochs']
        lr_stepped = checkpoint['lr_stepped']

        if lr_stepped:
            print('Decay the learning rate by a factor of 0.1.')
            for group in optimizer.param_groups:
                group['lr'] *= 0.1

        print("=> Start epoch {} ".format(start_epoch))

    model = nn.DataParallel(model).cuda()

    # Create data loaders
    save_path = None
    if args.gs_save:
        save_path = output_dir
    dataset, num_classes, train_loader, _, _ = get_data(args.dataset, args.data_dir, model, matcher, save_path, args)

    if not args.evaluate:
        # Trainer
        trainer = Trainer(model, criterion, args.clip_value)
        t0 = time.time()

        # Start training
        for epoch in range(start_epoch, args.max_epochs):
            loss, acc = trainer.train(epoch, train_loader, optimizer)

            if epoch == 1:
                base_loss = loss

            lr = list(map(lambda group: group['lr'], optimizer.param_groups))

            train_time = time.time() - t0
            epoch1 = epoch + 1

            print(
                '* Finished epoch %d at lr=[%g, %g, %g]. Loss: %.3f. Acc: %.2f%%. Training time: %.0f seconds.                  \n'
                % (epoch1, lr[0], lr[1], lr[2], loss, acc * 100, train_time))

            if not lr_stepped:
                if args.step_factor >= 1:  # for fixed schedule
                    if epoch1 == args.step_factor:
                        lr_stepped = True
                elif (base_loss is not None) and (loss < base_loss * args.step_factor):  # for dynamic schedule
                    lr_stepped = True
                    final_epochs = min(args.max_epochs, epoch1 + epoch1 // 2)
                if lr_stepped:
                    print('Decay the learning rate by a factor of 0.1. Final epochs: %d.\n' % final_epochs)
                    for group in optimizer.param_groups:
                        group['lr'] *= 0.1

            save_checkpoint({
                'model': model.module.state_dict(),
                'criterion': criterion.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch1,
                'final_epochs': final_epochs,
                'base_loss': base_loss,
                'lr_stepped': lr_stepped,
            }, fpath=osp.join(output_dir, 'checkpoint.pth.tar'))
            
            if epoch1 == final_epochs:
                print('The learning converges at epoch %d.\n' % epoch1)
                break

    json_file = osp.join(output_dir, 'results.json')
    
    if not args.evaluate:
        arg_dict = {'train_dataset': args.dataset, 'exp_dir': args.exp_dir, 'method': args.method, 'sub_method': args.sub_method}
        with open(json_file, 'a') as f:
            json.dump(arg_dict, f)
            f.write('\n')
        train_dict = {'train_dataset': args.dataset, 'loss': loss, 'acc': acc, 'epochs': epoch1, 'train_time': train_time}
        with open(json_file, 'a') as f:
            json.dump(train_dict, f)
            f.write('\n')

    # Final test
    print('\nEvaluate the learned model:')
    t0 = time.time()

    # Evaluator
    evaluator = Evaluator(model, matcher, args.test_gal_batch, args.test_prob_batch, 
            args.tau, args.sigma, args.K, args.alpha)

    test_names = args.testset.strip().split(',')
    for test_name in test_names:
        if test_name not in datasets.names():
            print('Unknown dataset: %s.' % test_name)
            continue

        t1 = time.time()
        testset, test_query_loader, test_gallery_loader = \
            get_test_data(test_name, args.data_dir, args.height, args.width, args.workers, args.test_fea_batch)

        if not args.do_tlift:
            testset.has_time_info = False

        test_rank1, test_mAP, test_rank1_rerank, test_mAP_rerank, test_rank1_tlift, test_mAP_tlift, test_dist, \
        test_dist_rerank, test_dist_tlift, pre_tlift_dict = \
            evaluator.evaluate(testset, test_query_loader, test_gallery_loader)

        test_time = time.time() - t1

        if testset.has_time_info:
            test_dict = {'test_dataset': test_name, 'rank1': test_rank1, 'mAP': test_mAP, 'rank1_rerank': test_rank1_rerank, 
                    'mAP_rerank': test_mAP_rerank, 'rank1_tlift': test_rank1_tlift, 'mAP_tlift': test_mAP_tlift, 'test_time': test_time}
            print('  %s: rank1=%.1f, mAP=%.1f, rank1_rerank=%.1f, mAP_rerank=%.1f,'
                ' rank1_rerank_tlift=%.1f, mAP_rerank_tlift=%.1f.\n'
                % (test_name, test_rank1 * 100, test_mAP * 100, test_rank1_rerank * 100, test_mAP_rerank * 100,
                    test_rank1_tlift * 100, test_mAP_tlift * 100))
        else:
            test_dict = {'test_dataset': test_name, 'rank1': test_rank1, 'mAP': test_mAP, 'test_time': test_time}
            print('  %s: rank1=%.1f, mAP=%.1f.\n' % (test_name, test_rank1 * 100, test_mAP * 100))

        with open(json_file, 'a') as f:
            json.dump(test_dict, f)
            f.write('\n')
            
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
        print('Finished training at epoch %d, loss = %.3f, acc = %.2f%%.\n'
              % (epoch1, loss, acc * 100))
        print("Total training time: %.3f sec. Average training time per epoch: %.3f sec." % (
            train_time, train_time / (epoch1 - start_epoch)))
    print("Total testing time: %.3f sec.\n" % test_time)

    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv_GS")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market', choices=datasets.names(),
                        help="the training dataset")
    parser.add_argument('--combine_all', action='store_true', default=False,
                        help="combine all data for training, default: False")
    parser.add_argument('--testset', type=str, default='cuhk03_np_detected,msmt', help="the test datasets")
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="the batch size, default: 64")
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
    parser.add_argument('--ibn', type=str, choices={'a', 'b', 'none'}, default='b', help="IBN type. Choose from 'a' or 'b'. Default: 'b'")
    # TLift
    parser.add_argument('--do_tlift', action='store_true', default=False, help="apply TLift, default: False")
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
    parser.add_argument('--max_size', type=float, default=0.8, help="maximal size for the ramdom occlusion. default: 0.8")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.005,
                        help="Learning rate of the new parameters. For pretrained "
                             "parameters it is 10 times smaller than this. Default: 0.005.")
    # training configurations
    parser.add_argument('--step_factor', type=float, default=0.7, help="loss descent factor to reduce the learning rate")
    parser.add_argument('--max_epochs', type=int, default=60, help="the maximal number of training epochs, default: 60")
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help="Path for resuming training. Choices: '' (new start, default), "
                             "'ori' (original path), or a real path")
    parser.add_argument('--clip_value', type=float, default=8, help="the gradient clip value, default: 8")
    parser.add_argument('--margin', type=float, default=16, help="margin of the triplet loss, default: 16")
    # graph sampler
    parser.add_argument('--num_instance', type=int, default=2, help="the number of instance per class in a batch, default: 2")
    parser.add_argument('--gs_save', action='store_true', default=False, help="save the graph distance and top-k indices, default: False")
    parser.add_argument('--gs_verbose', action='store_true', default=False, help="verbose for the graph sampler, default: False")
    
    # test configurations
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only, default: False")
    parser.add_argument('--test_fea_batch', type=int, default=256,
                        help="Feature extraction batch size during testing. Default: 256."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_gal_batch', type=int, default=128,
                        help="QAConv gallery batch size during testing. Default: 128."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=128,
                        help="QAConv probe batch size (as kernel) during testing. Default: 128."
                             "Reduce this if you encounter a GPU memory overflow.")
    
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'),
                        help="the path to the image data")
    parser.add_argument('--exp-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'Exp'),
                        help="the path to the output directory")
    parser.add_argument('--method', type=str, default='QAConv_GS', help="method name for the output directory")
    parser.add_argument('--sub_method', type=str, default='res50-ibnb-layer3',
                        help="sub method name for the output directory")
    parser.add_argument('--save_score', default=False, action='store_true',
                        help="save the matching score or not, default: False")

    main(parser.parse_args())
