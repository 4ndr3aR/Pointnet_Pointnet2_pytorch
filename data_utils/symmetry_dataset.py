#!/usr/bin/env python3

import numpy as np
import torch
import torchvision

#import cv2
import sys

import lzma
import pickle

from pathlib import Path
from numpy import genfromtxt

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.precision = 3

'''
from .mnist_dataset import show_3d_image as mnist_show_3d_image
from .mnist_dataset import transform_img2pc, get_random_sample, show_number_of_points_histogram
'''

from mnist_dataset import show_3d_image as mnist_show_3d_image
from mnist_dataset import transform_img2pc, get_random_sample, show_number_of_points_histogram

def show_3d_image(points, label):
	return mnist_show_3d_image(points, label)

def load_dataset(path, fname, debug=False):
	data = None
	if Path(Path(path) / fname).is_file() and Path(fname).suffix == '.xz':
		if debug:
			print(f'Reading LZMA compressed dataset...')
		with lzma.open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			#if debug:
			#	print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	else:
		if debug:
			print(f'Reading uncompressed dataset...')
		with open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			#if debug:
			#	print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	if debug:
		print(f'Loaded {len(data)} samples into {type(data)} structure...')
		print(f'Dataframe length : {len(data)}')
		print(f'Dataframe columns: {list(data[0].keys())}')
		print(f'Dataframe content: {data}')

	return data

class Symmetry(Dataset):
	"""Symmetry dataset."""

	'''
     1	astroid
     2	geom-petal
	'''

	NUM_CLASSIFICATION_CLASSES = 2
	MAX_POINTS = 10000

	#LABELS = ['cassinian-oval', 'cissoid', 'citrus', 'egg', 'geom-petal', 'hypocycloid', 'mouth', 'spiral']
	LABELS = ['astroid', 'geometric_petal']

	def __init__(self, path, partition, gt_column=None, max_points=MAX_POINTS, labels=LABELS, add_noise=False):
		self.path       = path
		self.labels     = labels
		#self.vocab      = [[], labels]			# because of: ```if is_listy(self.vocab): self.vocab = self.vocab[-1]```
		self.add_noise  = add_noise
		self.dataset    = load_dataset(path, partition + '.xz')
		self.gt_column  = gt_column			# e.g. 'label', 'angle', 'trans_x', 'trans_y', ...
		self.max_points = max_points

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx, debug=False):
		debug = True
		# Dataframe columns: ['angle', 'trans_x', 'trans_y', 'a', 'b', 'n_petals', 'label', 'fpath', 'points']
		#row = self.dataset.iloc[idx]
		row = self.dataset[idx]
		points,label,split = row['points'],row['label'],row['split']
		lbl = torch.tensor(self.labels.index(label))
		#angle,trans_x,trans_y,a,b,n_petals = row['angle'],row['trans_x'],row['trans_y'],row['a'],row['b'],row['n_petals']
		#angle,trans_x,trans_y,a,b,n_petals = row['angle'],row['trans_x'],row['trans_y'],row['a'],row['b'],row['n_petals']
		label, split, points, gt = row['label'], row['split'], row['points'][0], row['gt']

		if debug:
			print(f'__getitem__() idx: {idx} - {type(row) = } - {len(row) = } - {points.shape = } - {label = } - {split = } - {lbl = }')
		points = np.hstack((points, np.zeros((points.shape[0], 1))))
		if debug:
			print(f'__getitem__() idx: {idx} - {type(row) = } - {len(row) = } - {points.shape = } - {label = } - {split = } - {lbl = }')
			for gt_row in gt.iterrows():
				idx     = gt_row[0]
				row_arr = gt_row[1].values
			#print(f'__getitem__() idx: {idx} - {angle = } - {trans_x = } - {trans_y = } - {a = } - {b = } - {n_petals = }')
			print(f'__getitem__() idx: {idx} - {label = } - {split = } - {points.shape = } - {gt.shape = } - {row_arr = }')

		if self.max_points - points.shape[0] > 0:
			# Duplicate points
			sampling_indices = np.random.choice(points.shape[0], self.max_points - points.shape[0])
			if debug:
				print(f'__getitem__() idx: {idx} - {len(sampling_indices) = } - {sampling_indices = }')
			new_points = points[sampling_indices, : ]
			points = np.concatenate((points, new_points), axis=0)
		else:
			# sample points
			sampling_indices = np.random.choice(points.shape[0], self.max_points)
			if debug:
				print(f'__getitem__() idx: {idx} - {len(sampling_indices) = } - {sampling_indices = }')
			points = points[sampling_indices, :]

		points = points.astype(np.float32)
		points = torch.tensor(points)
		if self.add_noise:
			#points = points + (0.01**0.5)*torch.randn(points.shape[0], points.shape[1])		# high
			points = points + (0.001**0.5)*torch.randn(points.shape[0], points.shape[1])		# perfect
		if self.gt_column:
			if self.gt_column == 'label':
				gt = lbl
			else:
				gt = row[self.gt_column]
		else:
			gt = row['label']
		return points, gt

	def __getitems__(self, idxs, debug=False):
		item_list = []
		for idx in idxs:
			item_list.append(self.__getitem__(idx, debug=debug))
		return item_list

	def items(self, idxs, debug=False):
		return self.__getitems__(idxs, debug=debug)

def show_one_batch(one_batch, debug=False):
	if debug:
		print(f'one_batch: {type(one_batch)}')
		print(f'one_batch: {len(one_batch)}')

	if type(one_batch) == list:
		if debug:
			print(f'one_batch: {type(one_batch[0])}')
			print(f'one_batch: {one_batch[0].shape}')
			print(f'one_batch: {type(one_batch[1])}')
			print(f'one_batch: {one_batch[1]}')
		for idx in range(len(one_batch[0])):
			points = one_batch[0][idx]
			label  = one_batch[1][idx]
			if debug:
				print(f'Points shape: {points.shape} - label: {label}')
			show_3d_image(points, label)
	elif type(one_batch) == torch.Tensor:
		if debug:
			print(f'one_batch: {one_batch.shape}')
		for idx in range(len(one_batch)):
			points = one_batch[idx]
			label  = one_batch[idx]
			if debug:
				print(f'Points shape: {points.shape} - label: {label}')
			show_3d_image(points, label)
	else:
		print(f'Unknown type: {type(one_batch)}')



def load_and_describe_npz(file_path):
    """Loads an .npz file, extracts variable names and types, and returns them in a dictionary.

    Args:
        file_path (str): The path to the .npz file.

    Returns:
        dict: A dictionary where keys are variable names and values are their data types.
    """

    with np.load(file_path) as data:
        variables = {key: data[key] for key in data.files}
    return variables

def load_npz(fn):
    pts  = load_and_describe_npz(fn)['points']
    return pts

def read_gt_file(gt_file, debug=False):
    '''
    Read a -sym.txt file with the format:
        plane 0.0 0.0 0.0 -0.7071067811865476 0.7071067811865476 0.0
        axis 0.0 0.0 0.0 0.0 0.0 1.0 1.5707963267948966
    and return a dataframe containing in each row the type of symmetry (planar or circular)
    and the six floats representing the plane or seven if representing an axis (skiping the first row)
    '''
    
    df = pd.read_csv(gt_file, header=None, skiprows=1)
    df = df[0].str.split(' ', expand=True)
    
    if debug:
        print(f'Dataframe BEFORE float conversion: {df}')
    for col in range(7):
        df[col+1] = df[col+1].astype(float)
    if debug:
        print(f'Dataframe AFTER  float conversion: {df}')
    
    df.columns = ['type', 'popx', 'popy', 'popz', 'nx', 'ny', 'nz', 'rot'] # normals and points on plane (for planar symmetry) and rot for axial symmetry
    return df

def read_symmetry_dataset(path, debug=False):
	#dataset = pd.DataFrame(columns=['angle, trans_x, trans_y, a, b, n_petals', 'label', 'fpath', 'points'])
	#dataset = pd.DataFrame()
	dataset = []

	counter = 0

	if debug:
		print(f'read_symmetry_dataset() - Received path: {path}')
	csv_files = Path(path).rglob('*.npz')
	if debug:
		print(f'read_symmetry_dataset() - {len(list(csv_files))} files found')

	csv_files = Path(path).rglob('*.npz')
	for idx, file in enumerate(list(csv_files)):
		if debug:
			print(f'read_symmetry_dataset() - reading file: {file}')
			print(f'read_symmetry_dataset() - in path     : {file.parents[0]}')
		if file.is_file():
			#points = genfromtxt(file, delimiter=',')
			points = load_npz(file)
			label  = file.parent.stem			# geom_petal
			split  = file.parent.parent.stem		# test
			gtfn   = file.parents[0] / (file.name[:-4] + '-sym.txt')
			#tmpdf  = pd.DataFrame(columns=['label', 'fpath', 'points', 'gt'])
			tmpdf  = dict()
			tmpdf['label']  = label
			tmpdf['split']  = split 
			tmpdf['points'] = [points]
			tmpdf['gt']     = read_gt_file(gtfn)

			if counter % 1000 == 0:
				print(f'read_symmetry_dataset() - {counter+1} files processed so far ({split} - {label} - {points.shape})')
				#print(f'read_symmetry_dataset() - {idx} - {file} - {parmfn} - {tmpdf.shape} - {tmpdf}')
				print(f'read_symmetry_dataset() - {idx} - {len(dataset)} - {dataset}')

			dataset.append(tmpdf)

			counter += 1

	print(f'read_symmetry_dataset() - {counter} files read')

	if debug:
		for i in range(3):
			print(f'read_symmetry_dataset() - {i} - {dataset[i]}')

	return dataset

def save_dataset(dataset, path, fname):
	print(f'Writing uncompressed dataset...')
	with open(Path(path) / Path(str(fname) + '.pickle'), 'wb') as fhandle:
		pickle.dump(dataset, fhandle)
	print(f'Writing compressed dataset...')
	with lzma.open(Path(path) / Path(str(fname) + '.xz'), 'wb') as fhandle:
		pickle.dump(dataset, fhandle)

def read_and_save_dataset_partitions(dataset_path):
	test_dataset     = read_symmetry_dataset(dataset_path / 'test')
	save_dataset(test_dataset, './', 'test')
	valid_dataset    = read_symmetry_dataset(dataset_path / 'valid')
	save_dataset(valid_dataset, './', 'valid')
	train_dataset    = read_symmetry_dataset(dataset_path / 'train')
	save_dataset(train_dataset, './', 'train')
	return train_dataset, valid_dataset, test_dataset

def create_symmetry_dataloaders(symmetry_path, bs, gt_column=None, only_test_set=False, validation_and_test_sets=False):
	train_dataset,    val_dataset,    test_dataset    = None, None, None
	train_dataloader, val_dataloader, test_dataloader = None, None, None

	print('.', end='', flush=True)
	test_dataset = Symmetry(path=symmetry_path, gt_column=gt_column, partition='test')

	if not only_test_set:
		print('.', end='', flush=True)
		val_dataset = Symmetry(path=symmetry_path, gt_column=gt_column, partition='validation')
		print('.', end='', flush=True)
		train_dataset = Symmetry(path=symmetry_path, gt_column=gt_column, partition='training')		# keep this one as the last because it's pretty slow
	else:
		print(f'Warning: using only test set for dataloaders...')

	print('. DONE!', flush=True)

	if not only_test_set:
		train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
		val_dataloader   = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
	test_dataloader = DataLoader(test_dataset,  batch_size=bs, shuffle=only_test_set)

	if only_test_set:
		train_dataloader = test_dataloader
		val_dataloader   = test_dataloader

	if validation_and_test_sets:
		print('.', end='', flush=True)
		val_dataset    = Symmetry(path=symmetry_path, gt_column=gt_column, partition='validation')
		val_dataloader = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
		print(f'Warning: using only validation and test set for dataloaders...')

	return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

	test_read      = False
	test_write     = False
	test_load      = False
	test_show      = False
	test_one_batch = True

	if test_read:
		print(f'Testing the read_symmetry_dataset() function...')
		dataset_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/symmetries-dataset-astroid-geom_petal-100')
		test_data = read_symmetry_dataset(dataset_path)
		print(f'read_symmetry_dataset() complete')
		sys.exit()

	if test_write:
		print(f'Testing the read_and_save_dataset_partitions() function...')
		dataset_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/symmetries-dataset-astroid-geom_petal-100')
		read_and_save_dataset_partitions(dataset_path)
		print(f'read_and_save_dataset_partitions() complete')
		sys.exit()

	if test_load:
		print(f'Testing the load_dataset() function...')
		test_data = load_dataset('./', 'test.xz', debug=True)
		print(f'Test dataset length: {len(test_data)}')
		print(f'load_dataset() complete')
		sys.exit()

	if test_show:
		dset   = load_dataset('./', 'train.xz', debug=True)
		idx    = 0
		points = dset[idx]['points'][0]
		show_3d_image(points, dset[idx]['label'])
		gt     = dset[idx]['gt']
		print(f'gt: {gt}')

		idx    = 11
		points = dset[idx]['points'][0]
		show_3d_image(points, dset[idx]['label'])
		gt     = dset[idx]['gt']
		print(f'gt: {gt}')

	if test_one_batch:
		#symmetry_path = Path('../data/Symmetry')
		symmetry_path = Path('./')
		trainDataLoader, valDataLoader, testDataLoader = create_symmetry_dataloaders(symmetry_path, gt_column='label', bs=16, only_test_set=True)

		one_batch = next(iter(testDataLoader))
		show_one_batch(one_batch)



