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

from data_utils.mnist_dataset    import MNIST3D,  create_3dmnist_dataloaders,  show_3d_image, get_random_sample
from data_utils.curveml_dataset  import CurveML,  create_curveml_dataloaders,  show_one_batch
from data_utils.symmetry_dataset import Symmetry, create_symmetry_dataloaders, show_one_batch

'''
cmdlines:

./test_regression.py --curveml_dataset --gt_columns n_petals	--y_range_min 0.	--y_range_max 8.	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-n_petals-bs480/2023-11-22_16-27
./test_regression.py --curveml_dataset --gt_columns angle	--y_range_min 0.	--y_range_max 360.	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-angle-bs480/2023-11-23_18-30

./test_regression.py --curveml_dataset --gt_columns a		--y_range_min 0.	--y_range_max 2.83	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-a-bs480/2023-11-24_18-41
./test_regression.py --curveml_dataset --gt_columns b		--y_range_min 0.	--y_range_max 1.05	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-b-bs480/2023-11-25_08-24

./test_regression.py --curveml_dataset --gt_columns trans_x	--y_range_min -0.8	--y_range_max 0.8	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-trans_x-bs480/2023-11-25_22-43
./test_regression.py --curveml_dataset --gt_columns trans_y	--y_range_min -0.8	--y_range_max 0.8	--num_classes 1 --batch_size 480 --log_dir pointnet-nonormal-curveml-regression-trans_y-bs480/2023-11-26_23-59

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
    parser.add_argument('--model', default='pointnet_regr', help='model name [default: pointnet_regr]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[1, 8, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--y_range_min', default=-1.,  type=float, help='min value to pass to SigmoidRange class')
    parser.add_argument('--y_range_max', default=-1.,  type=float, help='max value to pass to SigmoidRange class')
    parser.add_argument('--gt_columns', default='none', nargs='+', type=str, help='ground truth column name in the DataFrame')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--mnist_dataset', action='store_true', default=False, help='use the 3D MNIST dataset')
    parser.add_argument('--curveml_dataset', action='store_true', default=False, help='use the CurveML dataset')
    parser.add_argument('--symmetry_dataset', action='store_true', default=False, help='use the Symmetry dataset')
    parser.add_argument('--show_one_batch', action='store_true', default=False, help='show one batch before start training')
    parser.add_argument('--show_predictions', action='store_true', default=False, help='show predictions during testing')
    return parser.parse_args()



def print_list_of_tensors(list_of_tensors, name, strtype):
	if isinstance(list_of_tensors, torch.Tensor):
		print(f'{name.title()} is a tensor with shape: {list_of_tensors.shape}')
	elif (isinstance(list_of_tensors, list) or isinstance(list_of_tensors, tuple)):
		print(f'{name.title()} is a {strtype} with length: {len(list_of_tensors)}')
		for idx,pr in enumerate(list_of_tensors):
			if isinstance(pr, torch.Tensor):
				print(f'{name}[{idx}] shape: {pr.shape}')
			elif (isinstance(pr, list) or isinstance(pr, tuple)):
				print(f'{name}[{idx}] len: {len(pr)}')
				for jdx,pr_itm in enumerate(pr):
					print(f'{name}[{idx}][{jdx}]: {pr_itm.shape}')


def print_pred_target(pred, target):
	print(f'{pred = }')
	print(f'{target = }')
	pr_type  = str(type(pred)).replace("<class '", "").replace("'>", "")
	tgt_type = str(type(target)).replace("<class '", "").replace("'>", "")
	print_list_of_tensors(pred  , 'pred'  , pr_type)
	print_list_of_tensors(target, 'target', tgt_type)
	'''
	if isinstance(pred, torch.Tensor):
		print(f'Pred is a tensor with shape: {pred.shape}')
	elif (isinstance(pred, list) or isinstance(pred, tuple)):
		print(f'Pred is a {pr_type} with length: {len(pred)}')
		for idx,pr in enumerate(pred):
			if isinstance(pr, torch.Tensor):
				print(f'pred[{idx}] shape: {pr.shape}')
			elif (isinstance(pr, list) or isinstance(pr, tuple)):
				print(f'pred[{idx}] len: {len(pr)}')
				for jdx,pr_itm in enumerate(pr):
					print(f'pred[{idx}][{jdx}]: {pr_itm.shape}')
	if isinstance(target, torch.Tensor):
		print(f'Target is a tensor with shape: {target.shape}')
	elif(isinstance(target, list) or isinstance(target, tuple)):
		print(f'Target is a {tgt_type} with length: {len(target)}')
		for idx,tgt in enumerate(target):
			if isinstance(tgt, torch.Tensor):
				print(f'target[{idx}] shape: {tgt.shape}')
			elif (isinstance(tgt, list) or isinstance(tgt, tuple)):
				print(f'target[{idx}] len: {len(tgt)}')
				for jdx,tgt_itm in enumerate(tgt):
					print(f'target[{idx}][{jdx}]: {tgt_itm.shape}')
	'''


def test_regression(model, regressor, loader, num_classes=1, debug=False):
	mse_total = torch.zeros(len(loader))
	regressor = regressor.eval()

	if debug:
		log_string(f'type(loader): {type(loader)}')
		log_string(f'len(loader): {len(loader)}')
		log_string(f'bs: {loader.batch_size}')
		log_string(f'mse_total: {mse_total.shape}')

	sample_counter = 0

	for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

		#print(f'{type(points) = } - {type(target) = }')
		if isinstance(points, list):
			points = points[0]					# TODO: remove me or I'll cause bugs!

		if isinstance(target, torch.Tensor):
			target  = target.float()
			if not args.use_cpu:
				target = target.cuda()
		elif isinstance(target, list):
			regressor.list_target_to_cuda_float_tensor(target, cuda=(not args.use_cpu))             # because now target is a list of lists/np.arrays

		if not args.use_cpu:
			#points, target = points.cuda(), target.cuda()
			points = points.cuda()

		points  = points.transpose(2, 1)
		pred, _ = regressor(points)
		#target  = target.float()

		if debug:
			if isinstance(target, torch.Tensor):
				log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')
			elif isinstance(target, list):
				log_string(f'[{j}] pred   : {len(pred)} - target   : {len(target)}')
			log_string(f'[{j}] pred   : {pred} - target   : {target}')
			log_string(f'[{j}] pred[0]: {pred[0]} - target[0]: {target[0]}')
		'''
		pred = pred.squeeze(1)
		if debug:
			log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')
		'''
		if isinstance(pred, torch.Tensor):
			pred = pred.squeeze(1)
		elif (isinstance(pred, list) or isinstance(pred, tuple)) and (isinstance(target, list) or isinstance(target, tuple)):
			pass
		else:
			log_string(f'Unhandled pred/target types: {type(pred)} - {type(target)}')
		if debug:
			if isinstance(pred, torch.Tensor):
				log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')
			elif isinstance(target, list):
				log_string(f'[{j}] pred   : {len(pred)} - target   : {len(target)}')


		'''
		assert(pred.shape == target.shape)

		mse_tensor = (pred - target) ** 2
		if debug:
			log_string(f'[{j}] mse_tensor: {mse_tensor.shape}')
			log_string(f'[{j}] mse_tensor: {mse_tensor}')
			log_string(f'[{j}] mse_total : {mse_total.shape}')
		mse_total[j] = mse_tensor.sum()
		if debug:
			log_string(f'[{j}] mse_total : {mse_total}')
		'''
		mse_loss_lst = None
		if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
			assert(pred.shape == target.shape)
			mse_loss_lst = [(pred - target) ** 2]
		elif (isinstance(pred, list) or isinstance(pred, tuple)) and (isinstance(target, list) or isinstance(target, tuple)):
			'''
			mse_tensor_lst = []
			for idx,pr in enumerate(pred):
				log_string(f'[{j}] pred[{idx}]: {pr.shape} - target[{idx}]: {target[idx].shape}')
				tgt = target[idx].reshape(pr.shape)
				assert(pr.shape == tgt.shape)
				mse_tensor_itm = (pr - tgt) ** 2
				mse_tensor_lst.append(mse_tensor_itm.sum())
			mse_tensor = torch.stack(mse_tensor_lst)
			'''
			loss = model.get_loss.list_target_loss_impl(pred, target)
			mse_loss_lst = [loss]
		else:
			log_string(f'Unhandled pred/target types: {type(pred)} - {type(target)}')

		mse_total[j] = mse_loss_lst.sum() if isinstance(mse_loss_lst, torch.Tensor) else sum(mse_loss_lst)
		if debug:
			log_string(f'[{j}] mse_total : {mse_total}')

		sample_counter += loader.batch_size

		print_pred_target(pred, target)

		if debug:
			bs = points.size()[0]
			for i in range(bs):
				if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
					print(f'{points.shape = } - {target.shape = } - {pred.shape = }')
				elif (isinstance(pred, list) or isinstance(pred, tuple)) and (isinstance(target, list) or isinstance(target, tuple)):
					print(f'{len(points) = } - {len(target) = } - {len(pred) = }')

				print(f'pred[{i}]: {pred[i] = }')
				print(f'target[{i}]: {target[i] = }')

				if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
					prd = pred[i].cpu()
					print(f'{points[i].permute(1, 0).cpu().shape = } - {tgt.shape = } - {prd.shape = }')
				else:
					prd = pred[i]
				tgt = target[i].cpu()

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
        gt_columns = args.gt_columns if args.gt_columns is not None and args.gt_columns != 'none' else 'label'
        print(f'Using column: {gt_columns} as ground truth...')
        _, _, testDataLoader = create_curveml_dataloaders(curveml_path, gt_columns=gt_columns, bs=args.batch_size, only_test_set=True)
    elif args.symmetry_dataset:
        log_string('Loading the Symmetry dataset...')
        symmetry_path = Path('./data/Symmetry')
        #symmetry_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/gz/symmetries-dataset-astroid-geom_petal-100k')
        #symmetry_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/gz/symmetries-dataset-astroid-geom_petal-10k')
        gt_columns = args.gt_columns if args.gt_columns is not None and args.gt_columns != 'none' else 'label'
        print(f'Using column: {gt_columns} as ground truth...')
        _, _, testDataLoader = create_symmetry_dataloaders(symmetry_path, gt_columns=gt_columns, bs=args.batch_size, only_test_set=True, extension='.xz')

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

    best_model_fn = str(experiment_dir) + '/checkpoints/best_model.pth'
    print(f'Loading model: {best_model_fn}')
    checkpoint = torch.load(best_model_fn)
    regressor.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        mse_mean, mse_sum, mse = test_regression(model, regressor.eval(), testDataLoader, num_classes=num_classes, debug=args.show_predictions)
        log_string(f'Test MSE Loss: {mse} - Test mean MSE Loss: {mse_mean} - Test sum MSE Loss: {mse_sum}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
