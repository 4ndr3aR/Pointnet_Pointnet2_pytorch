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

'''
cmdlines:

./test_regression.py --curveml_dataset --gt_column n_petals --y_range_min 0. --y_range_max 8. --num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-n_petals-bs480/2023-11-22_16-27
./test_regression.py --curveml_dataset --gt_column angle --y_range_min 0. --y_range_max 360. --num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-angle-bs480/2023-11-23_18-30

'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

logger = logging.getLogger("Model")

def log_string(str):
    logger.info(str)
    print(str)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_classes', default=40, type=int, choices=[1, 8, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--y_range_min', default=-1.,  type=float, help='min value to pass to SigmoidRange class')
    parser.add_argument('--y_range_max', default=-1.,  type=float, help='max value to pass to SigmoidRange class')
    parser.add_argument('--gt_column', default='none',  type=str, help='max value to pass to SigmoidRange class')
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


def test_regression(model, loader, num_classes=1, debug=False):
	mse_total = torch.zeros(len(loader))
	regressor = model.eval()

	if debug:
		log_string(f'type(loader): {type(loader)}')
		log_string(f'len(loader): {len(loader)}')
		log_string(f'bs: {loader.batch_size}')
		log_string(f'mse_total: {mse_total.shape}')

	sample_counter = 0

	for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

		if not args.use_cpu:
			points, target = points.cuda(), target.cuda()

		points  = points.transpose(2, 1)
		pred, _ = regressor(points)
		target  = target.float()

		if debug:
			log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')
			log_string(f'[{j}] pred   : {pred} - target   : {target}')
			log_string(f'[{j}] pred[0]: {pred[0]} - target[0]: {target[0]}')
		pred = pred.squeeze(1)
		if debug:
			log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')

		assert(pred.shape == target.shape)

		mse_tensor = (pred - target) ** 2
		if debug:
			log_string(f'[{j}] mse_tensor: {mse_tensor.shape}')
			log_string(f'[{j}] mse_tensor: {mse_tensor}')
			log_string(f'[{j}] mse_total : {mse_total.shape}')
		mse_total[j] = mse_tensor.sum()
		if debug:
			log_string(f'[{j}] mse_total : {mse_total}')

		sample_counter += loader.batch_size

		if debug:
			bs = points.size()[0]
			for i in range(bs):
				print(f'{points.shape = } - {target.shape = } - {pred.shape = }')

				tgt = target[i].cpu()
				prd = pred[i].cpu()

				print(f'{points[i].permute(1, 0).cpu().shape = } - {tgt.shape = } - {prd.shape = }')

				show_3d_image(points[i].permute(1, 0).cpu(), f'GT: {tgt}/Pred: {prd}')
			break

	mse_mean = mse_total.mean()
	mse_sum  = mse_total.sum()
	mse = 1. * mse_total.sum() / sample_counter
	if debug:
		log_string(f'Returning mse_mean: {mse_mean} - mse_sum: {mse_sum} - mse: {mse}')
	return mse_mean, mse_sum, mse


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/regression/' + args.log_dir

    '''LOG'''
    args = parse_args()
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

    y_range = [args.y_range_min, args.y_range_max] if args.y_range_min != -1. and args.y_range_max != -1. else None
    if y_range is not None:
        log_string(f'Received y_range: {y_range} with type: {type(y_range[0])} - {type(y_range[1])}')
    regressor = model.get_model(num_classes, normal_channel=args.use_normals, y_range=y_range)
    if not args.use_cpu:
        regressor = regressor.cuda()

    one_batch = next(iter(testDataLoader))
    print(f'one_batch: {len(one_batch)} - {one_batch[0].shape}')
    one_batch_data  = one_batch[0]
    one_batch_label = one_batch[1]
    summary(regressor, input_data=torch.transpose(one_batch_data, 1, 2).cuda())
    if args.show_one_batch:
        show_one_batch([one_batch_data, one_batch_label])

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    regressor.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        #instance_acc, class_acc = test(regressor.eval(), testDataLoader, vote_num=args.num_votes, num_classes=num_classes, debug=args.show_predictions)
        mse_mean, mse_sum, mse = test_regression(regressor.eval(), testDataLoader, num_classes=num_classes, debug=args.show_predictions)
        log_string(f'Test MSE Loss: {mse} - Test mean MSE Loss: {mse_mean} - Test sum MSE Loss: {mse_sum}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
