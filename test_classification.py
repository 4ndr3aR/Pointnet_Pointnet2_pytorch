#!/usr/bin/env python3
"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

from pathlib import Path

from torchinfo import summary

from data_utils.mnist_dataset import MNIST3D, create_3dmnist_dataloaders, show_3d_image, get_random_sample
from data_utils.curveml_dataset import CurveML, create_curveml_dataloaders, show_one_batch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_classes', default=40, type=int, choices=[8, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--gt_column', default='none',  type=str, help='ground truth column name in the DataFrame')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--mnist_dataset', action='store_true', default=False, help='use the 3D MNIST dataset')
    parser.add_argument('--curveml_dataset', action='store_true', default=False, help='use the CurveML dataset')
    parser.add_argument('--show_one_batch', action='store_true', default=False, help='show one batch before start training')
    parser.add_argument('--show_predictions', action='store_true', default=False, help='show predictions during testing')
    return parser.parse_args()


def test(model, loader, num_classes=40, vote_num=1, debug=False):
	mean_correct = []
	classifier = model.eval()
	class_acc = np.zeros((num_classes, 3))

	for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
		if not args.use_cpu:
			points, target = points.cuda(), target.cuda()

		points = points.transpose(2, 1)
		vote_pool = torch.zeros(target.size()[0], num_classes).cuda()

		for _ in range(vote_num):
			pred, _ = classifier(points)
			vote_pool += pred
		pred = vote_pool / vote_num
		pred_choice = pred.data.max(1)[1]

		for cat in np.unique(target.cpu()):
			classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
			class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
			class_acc[cat, 1] += 1
		correct = pred_choice.eq(target.long().data).cpu().sum()
		mean_correct.append(correct.item() / float(points.size()[0]))

		if debug:
			bs = points.size()[0]
			for i in range(bs):
				print(f'{points.shape = } - {target.shape = } - {pred.shape = }')
				print(f'{points[i].permute(1, 0).cpu().shape = } - {target[i].cpu().shape = } - {pred_choice[i].cpu().shape = }')
				#print(f'{points[i].permute(1, 0).cpu() = } - {target[i].cpu() = } - {pred_choice[i].cpu() = }')

				labels       = loader.dataset.labels
				target_label = labels[target[i].cpu()]
				pred_label   = labels[pred_choice[i].cpu()]

				#show_3d_image(points[i].permute(1, 0).cpu(), f'GT: {target[i].cpu()}/Pred: {pred_choice[i].cpu()}')
				show_3d_image(points[i].permute(1, 0).cpu(), f'GT: {target_label}/Pred: {pred_label}')
			break

	class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
	class_acc = np.mean(class_acc[:, 2])
	instance_acc = np.mean(mean_correct)
	return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    trainDataLoader, valDataLoader, testDataLoader = None, None, None
    if args.mnist_dataset:
        log_string('Loading the 3D MNIST dataset...')
        _, _, testDataLoader = create_3dmnist_dataloaders(bs=args.batch_size)
    elif args.curveml_dataset:
        log_string('Loading the CurveML dataset...')
        curveml_path = Path('./data/CurveML')
        gt_column = args.gt_column if args.gt_column is not None and args.gt_column != 'none' else 'label'
        print(f'Using column: {gt_column} as ground truth...')
        _, _, testDataLoader = create_curveml_dataloaders(curveml_path, gt_column=gt_column, bs=args.batch_size, only_test_set=True)

    print(f'testDataLoader size: {len(testDataLoader)}')

    '''MODEL LOADING'''
    num_classes = args.num_classes
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_classes, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    # take a look at what you're testing...
    one_batch = next(iter(testDataLoader))
    print(f'one_batch: {len(one_batch)} - {one_batch[0].shape}')
    one_batch_data  = one_batch[0]
    one_batch_label = one_batch[1]
    summary(classifier, input_data=torch.transpose(one_batch_data, 1, 2).cuda())
    if args.show_one_batch:
        show_one_batch([one_batch_data, one_batch_label])

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_classes=num_classes, debug=args.show_predictions)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
