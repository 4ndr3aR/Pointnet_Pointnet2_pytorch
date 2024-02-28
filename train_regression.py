#!/usr/bin/env python3
"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

#from torchsummary import summary
from torchinfo import summary

from data_utils.ModelNetDataLoader import ModelNetDataLoader

from data_utils.mnist_dataset    import MNIST3D,  create_3dmnist_dataloaders,  show_3d_image, get_random_sample
from data_utils.curveml_dataset  import CurveML,  create_curveml_dataloaders,  show_one_batch
from data_utils.symmetry_dataset import Symmetry, create_symmetry_dataloaders, show_one_batch

'''
cmdlines:

./train_regression.py --curveml_dataset --batch_size 480 --gt_column n_petals --y_range_min 0 --y_range_max 8 --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-n_petals-bs480 &> pointnet1-nonormals-curveml-regression-n_petals-bs480-train-`currdate`-`currtime`.txt
./train_regression.py --curveml_dataset --batch_size 480 --gt_column angle --y_range_min 0. --y_range_max 360. --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-angle-bs480 &> pointnet1-nonormals-curveml-regression-angle-bs480-train-`currdate`-`currtime`.txt

./train_regression.py --curveml_dataset --batch_size 480 --gt_column a --y_range_min 0. --y_range_max 2.83 --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-a-bs480 &> pointnet1-nonormals-curveml-regression-a-bs480-train-`currdate`-`currtime`.txt
./train_regression.py --curveml_dataset --batch_size 480 --gt_column b --y_range_min 0. --y_range_max 1.05 --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-b-bs480 &> pointnet1-nonormals-curveml-regression-b-bs480-train-`currdate`-`currtime`.txt

./train_regression.py --curveml_dataset --batch_size 480 --gt_column trans_x --y_range_min -0.8 --y_range_max 0.8 --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-trans_x-bs480 &> pointnet1-nonormals-curveml-regression-trans_x-bs480-train-`currdate`-`currtime`.txt
./train_regression.py --curveml_dataset --batch_size 480 --gt_column trans_y --y_range_min -0.8 --y_range_max 0.8 --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-curveml-regression-trans_y-bs480 &> pointnet1-nonormals-curveml-regression-trans_y-bs480-train-`currdate`-`currtime`.txt



./train_regression.py --symmetry_dataset --batch_size 4 --gt_column cls type popx popy popz nx ny nz rot --num_classes 1 --model pointnet_cls --log_dir pointnet-nonormal-symmetry-bs128 --only_test_set

./train_regression.py --symmetry_dataset --batch_size 128 --learning_rate 0.01 --gt_column cls type popx popy popz nx ny nz rot --num_classes 1 --model pointnet_regr --log_dir pointnet-nonormal-symmetry-bs128

quick run:

./train_regression.py --symmetry_dataset --batch_size 5 --learning_rate 0.05 --gt_column cls type popx popy popz nx ny nz rot --num_classes 1 --model pointnet_regr --log_dir pointnet-nonormal-symmetry-bs5 --only_test_set


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
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[1, 8, 10, 40],  help='training on ModelNet10/40')
    #parser.add_argument('--y_range_min', default=-1.,  type=float, help='min value to pass to SigmoidRange class')
    #parser.add_argument('--y_range_max', default=-1.,  type=float, help='max value to pass to SigmoidRange class')
    parser.add_argument('--y_range_min', default=-1., nargs='+', type=float, help='min value to pass to SigmoidRange class (can be a list of floats)')
    parser.add_argument('--y_range_max', default=-1., nargs='+', type=float, help='max value to pass to SigmoidRange class (can be a list of floats)')
    parser.add_argument('--gt_columns', default='none', nargs='+', type=str, help='column to use as ground truth (can be a list of strings only with the Symmetry dataset)')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--mnist_dataset', action='store_true', default=False, help='use the 3D MNIST dataset')
    parser.add_argument('--curveml_dataset', action='store_true', default=False, help='use the CurveML dataset')
    parser.add_argument('--symmetry_dataset', action='store_true', default=False, help='use the Symmetry dataset')
    parser.add_argument('--show_one_batch', action='store_true', default=False, help='show one batch before start training')
    parser.add_argument('--only_test_set', action='store_true', default=False, help='only use test set for a very quick run (perfect to see if the model is learning)')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    regressor = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), torch.tensor(target).cuda()

        points = points.transpose(2, 1)
        pred, _ = regressor(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def test_regression(model, regressor, loader, num_class=1, debug=False):
	mse_total = torch.zeros(len(loader))
	regressor = regressor.eval()

	if debug:
		log_string(f'type(loader): {type(loader)}')
		log_string(f'len(loader): {len(loader)}')
		log_string(f'bs: {loader.batch_size}')
		log_string(f'mse_total: {mse_total.shape}')

	sample_counter = 0

	for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

		if not args.use_cpu:
			#points, target = points.cuda(), target.cuda()
			points = points.cuda()

		points  = points.transpose(2, 1)
		pred, _ = regressor(points)
		if isinstance(target, torch.Tensor):
			target  = target.float()
			if not args.use_cpu:
				target = target.cuda()
		elif isinstance(target, list):
			regressor.list_target_to_cuda_float_tensor(target, cuda=(not args.use_cpu))		# because now target is a list of lists/np.arrays
			'''
			for idx,tgt in enumerate(target):
				target[idx] = tgt.float()
			'''

		if debug:
			if isinstance(pred, torch.Tensor):
				log_string(f'[{j}] pred   : {pred.shape} - target   : {target.shape}')
			elif isinstance(target, list):
				log_string(f'[{j}] pred   : {len(pred)} - target   : {len(target)}')
			log_string(f'[{j}] pred   : {pred} - target   : {target}')
			log_string(f'[{j}] pred[0]: {pred[0]} - target[0]: {target[0]}')
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

		'''
		if debug:
			log_string(f'[{j}] mse_tensor: {mse_tensor.shape}')
			log_string(f'[{j}] mse_tensor: {mse_tensor}')
			log_string(f'[{j}] mse_total : {mse_total.shape}')
		'''
		mse_total[j] = mse_loss_lst.sum() if isinstance(mse_loss_lst, torch.Tensor) else sum(mse_loss_lst)
		if debug:
			log_string(f'[{j}] mse_total : {mse_total}')

		sample_counter += loader.batch_size

	mse_mean = mse_total.mean()
	mse_sum  = mse_total.sum()
	mse = 1. * mse_total.sum() / sample_counter
	if debug:
		log_string(f'Returning mse_mean: {mse_mean} - mse_sum: {mse_sum} - mse: {mse}')
	return mse_mean, mse_sum, mse


def save_model(best_epoch, regressor, optimizer, checkpoints_dir, instance_acc=0., class_acc=0., mse=0., mse_mean=0., mse_sum=0.):
	logger.info('Saving model...')
	savepath = str(checkpoints_dir) + '/best_model.pth'
	log_string('Saving at %s' % savepath)
	state = {
		'epoch': best_epoch,
		'instance_acc': instance_acc,
		'class_acc': class_acc,
		'mse': mse,
		'mse_mean': mse_mean,
		'mse_sum': mse_sum,
		'model_state_dict': regressor.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}
	torch.save(state, savepath)


def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
        exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    trainDataLoader, valDataLoader, testDataLoader = None, None, None
    if args.mnist_dataset:
        log_string('Loading the 3D MNIST dataset...')
        trainDataLoader, valDataLoader, testDataLoader = create_3dmnist_dataloaders(bs=args.batch_size)
    elif args.curveml_dataset:
        log_string('Loading the CurveML dataset...')
        curveml_path = Path('./data/CurveML')
        gt_columns = args.gt_columns if args.gt_columns is not None else 'label'
        log_string(f'Using column {gt_columns} as ground truth')
        trainDataLoader, valDataLoader, testDataLoader = create_curveml_dataloaders(curveml_path, gt_columns=gt_columns, bs=args.batch_size, only_test_set=args.only_test_set)
    elif args.symmetry_dataset:
        log_string('Loading the Symmetry dataset...')
        #symmetry_path = Path('./data/Symmetry')
        symmetry_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/gz/symmetries-dataset-astroid-geom_petal-100k')
        #symmetry_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/gz/symmetries-dataset-astroid-geom_petal-10k')
        gt_columns = args.gt_columns if args.gt_columns is not None else 'label'
        log_string(f'Using column {gt_columns} as ground truth')
        trainDataLoader, valDataLoader, testDataLoader = create_symmetry_dataloaders(symmetry_path, gt_columns=gt_columns, bs=args.batch_size, only_test_set=args.only_test_set, extension='.gz')

    log_string(f'trainDataLoader size (in batches): {len(trainDataLoader)}, valDataLoader size: {len(valDataLoader)}, testDataLoader size: {len(testDataLoader)}')

    '''MODEL LOADING'''
    num_class = args.num_classes
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    if args.mnist_dataset:
        shutil.copy('data_utils/mnist_dataset.py', str(exp_dir))
    if args.curveml_dataset:
        shutil.copy('data_utils/curveml_dataset.py', str(exp_dir))
    if args.symmetry_dataset:
        shutil.copy('data_utils/symmetry_dataset.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))

    if isinstance(args.y_range_min, list):
        args.y_range_min = float(args.y_range_min[0]) if len(args.y_range_min) == 1. else args.y_range_min		# with nargs, we always receive a list
    if isinstance(args.y_range_max, list):
        args.y_range_max = float(args.y_range_max[0]) if len(args.y_range_max) == 1. else args.y_range_max		# check if it's a list or not and convert
    y_range = [args.y_range_min, args.y_range_max] if args.y_range_min != -1. and args.y_range_max != -1. else None
    if y_range is not None:
        log_string(f'Received y_range: {y_range} with type: {type(y_range[0])} - {type(y_range[1])}')
    regressor = model.get_model(num_class, normal_channel=args.use_normals, y_range=y_range)

    dataset = 'symmetry' if args.symmetry_dataset else 'curveml' if args.curveml_dataset else 'mnist'
    criterion = model.get_loss(dataset=dataset, y_range=y_range)
    if args.y_range_min == -1. and args.y_range_max == -1.:
        regressor.apply(inplace_relu)

    if not args.use_cpu:
        regressor = regressor.cuda()
        criterion = criterion.cuda()

    # take a look at what you're training...
    one_batch = next(iter(trainDataLoader))
    log_string(f'one_batch: {len(one_batch)} - {one_batch[0].shape}')
    one_batch_data  = one_batch[0]
    one_batch_label = one_batch[1]
    input_data = torch.transpose(one_batch_data, 1, 2)
    summary(regressor, input_data=input_data.cuda() if not args.use_cpu else input_data)
    if args.show_one_batch:
        show_one_batch([one_batch_data, one_batch_label])

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        regressor.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            regressor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch      = 0
    global_step       = 0

    best_instance_acc = 0.0
    best_class_acc    = 0.0
    best_mse_mean     = 1.e12
    best_mse_sum      = 1.e12
    best_mse          = 1.e12

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        regressor = regressor.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            if not args.symmetry_dataset and not args.curveml_dataset:
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.symmetry_dataset:
                if not args.use_cpu:
                    points, target = points.cuda(), torch.tensor(target).cuda()

                pred, trans_feat = regressor(points)
                if args.y_range_min == -1. and args.y_range_max == -1.:
                    loss = criterion(pred, target.long(), trans_feat)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(points.size()[0]))
                else:
                    loss = criterion(pred, target.float(), trans_feat)
            else:
                regressor.list_target_to_cuda_float_tensor(target, cuda=(not args.use_cpu))		# because now target is a list of lists/np.arrays
                if not args.use_cpu:
                    points = points.cuda()
                '''
                    for idx,tgt in enumerate(target):		# because now target is a list of lists/np.arrays
                        print(f'cpu  target[{idx}]: {type(tgt)} -  {tgt}')
                        target[idx] = torch.tensor(tgt).cuda()
                        print(f'cuda target[{idx}]: {type(tgt)} -  {tgt}')
                else:
                    for idx,tgt in enumerate(target):		# because now target is a list of lists/np.arrays
                        print(f'cpu  target[{idx}]: {type(tgt)} -  {tgt}')
                        target[idx] = torch.tensor(tgt)
                '''

                pred, trans_feat = regressor(points)
                '''
                if args.y_range_min == -1. and args.y_range_max == -1.:
                    loss = criterion(pred, target, trans_feat)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(points.size()[0]))
                else:
                    loss = criterion(pred, target.float(), trans_feat)
                '''
                #print(f'pred  : {pred} - {type(pred)}')
                #print(f'target: {target} - {type(target)}')
                loss = criterion(pred, target, trans_feat)

            loss.backward()
            optimizer.step()
            global_step += 1

        if args.y_range_min == -1. and args.y_range_max == -1.:
            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)
        else:
            log_string(f'Train MSE Loss: {loss.item()}')


        with torch.no_grad():
            if y_range is not None or args.symmetry_dataset or args.curveml_dataset:
                mse_mean, mse_sum, mse = test_regression(model, regressor.eval(), valDataLoader, num_class=num_class)

                if (mse < best_mse):
                    best_epoch    = epoch + 1
                    best_mse      = mse
                    save_model(best_epoch, regressor, optimizer, checkpoints_dir, mse=mse, mse_mean=mse_mean, mse_sum=mse_sum)

                if (mse_mean < best_mse_mean):
                    best_mse_mean = mse_mean

                if (mse_sum  < best_mse_sum):
                    best_mse_sum  = mse_sum

                log_string(f'Valid MSE Loss: {mse} - Valid mean MSE Loss: {mse_mean} - Valid sum MSE Loss: {mse_sum}')

            else:
                instance_acc, class_acc = test(regressor.eval(), valDataLoader, num_class=num_class)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch        = epoch + 1
                    save_model(best_epoch, regressor, optimizer, checkpoints_dir, instance_acc=instance_acc, class_acc=class_acc)
    
                if (class_acc >= best_class_acc):
                    best_class_acc    = class_acc

                log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
