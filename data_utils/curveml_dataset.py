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

from .mnist_dataset import transform_img2pc, show_3d_image, get_random_sample, show_number_of_points_histogram

def load_dataset(path, fname, debug=False):
	data = None
	if Path(Path(path) / fname).is_file() and Path(fname).suffix == '.xz':
		if debug:
			print(f'Reading LZMA compressed dataset...')
		with lzma.open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			if debug:
				print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	else:
		if debug:
			print(f'Reading uncompressed dataset...')
		with open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			if debug:
				print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	return data

class CurveML(Dataset):
	"""CurveML dataset."""

	'''
     1	cassinian-oval
     2	cissoid
     3	citrus
     4	egg
     5	geom-petal
     6	hypocycloid
     7	mouth
     8	spiral
	'''

	NUM_CLASSIFICATION_CLASSES = 8
	#POINT_DIMENSION = 3
	MAX_POINTS = 400

	LABELS = ['cassinian-oval', 'cissoid', 'citrus', 'egg', 'geom-petal', 'hypocycloid', 'mouth', 'spiral']

	def __init__(self, path, partition, max_points=MAX_POINTS, labels=LABELS, add_noise=False):
		self.path       = path
		self.labels     = labels
		self.add_noise  = add_noise
		self.max_points = max_points
		self.dataset    = load_dataset(path, partition + '.xz')

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx, debug=False):
		points,label,fpath = self.dataset[idx]
		lbl = torch.tensor(self.labels.index(label))

		if debug:
			print(f'__getitem__() idx: {idx} - {type(self.dataset[idx]) = } - {len(self.dataset[idx]) = } - {points.shape = } - {label = } - {fpath = } - {lbl = }')
		points = np.hstack((points, np.zeros((points.shape[0], 1))))
		if debug:
			print(f'__getitem__() idx: {idx} - {type(self.dataset[idx]) = } - {len(self.dataset[idx]) = } - {points.shape = } - {label = } - {fpath = } - {lbl = }')


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

		return points, lbl

def show_one_batch(one_batch):
	print(f'one_batch: {type(one_batch)}')
	print(f'one_batch: {len(one_batch)}')

	if type(one_batch) == list:
		print(f'one_batch: {type(one_batch[0])}')
		print(f'one_batch: {one_batch[0].shape}')
		print(f'one_batch: {type(one_batch[1])}')
		print(f'one_batch: {one_batch[1]}')
		for idx in range(len(one_batch[0])):
			points = one_batch[0][idx]
			label  = one_batch[1][idx]
			print(f'Points shape: {points.shape} - label: {label}')
			show_3d_image(points, label)
	elif type(one_batch) == torch.Tensor:
		print(f'one_batch: {one_batch.shape}')
		for idx in range(len(one_batch)):
			points = one_batch[idx]
			label  = one_batch[idx]
			print(f'Points shape: {points.shape} - label: {label}')
			show_3d_image(points, label)
	else:
		print(f'Unknown type: {type(one_batch)}')

def read_curveml_dataset(path):
	dataset = []

	counter = 0

	csv_files = Path(path).rglob('point_cloud*.csv')
	print(f'read_curveml_dataset() - {len(list(csv_files))} files found')

	csv_files = Path(path).rglob('point_cloud*.csv')
	for idx, file in enumerate(list(csv_files)):
		if file.is_file():
			points = genfromtxt(file, delimiter=',')
			fpath  = file.parent.stem			# 028676
			label  = file.parent.parent.stem		# cassinian-oval
			counter += 1
			dataset.append((points, label, fpath))
			if counter % 1000 == 0:
				print(f'read_curveml_dataset() - {counter} files processed ({fpath} - {label} - {points.shape})')	
	print(f'read_curveml_dataset() - {counter} files read')

	return dataset

def save_dataset(dataset, path, fname):
	print(f'Writing uncompressed dataset...')
	with open(Path(path) / Path(str(fname) + '.pickle'), 'wb') as fhandle:
		pickle.dump(dataset, fhandle)
	print(f'Writing compressed dataset...')
	with lzma.open(Path(path) / Path(str(fname) + '.xz'), 'wb') as fhandle:
		pickle.dump(dataset, fhandle)

def save_dataset_partitions(dataset_path):
	test_list    = read_curveml_dataset(dataset_path / 'test')
	save_dataset(test_list, './', 'test')
	valid_list    = read_curveml_dataset(dataset_path / 'validation')
	save_dataset(valid_list, './', 'validation')
	train_list    = read_curveml_dataset(dataset_path / 'training')
	save_dataset(train_list, './', 'training')
	return train_list, valid_list, test_list

def create_curveml_dataloaders(curveml_path, bs, only_test_set=False):
	train_dataloader, val_dataloader, test_dataloader = None, None, None

	print('.', end='', flush=True)
	test_dataset  = CurveML(path=curveml_path, partition='test')

	if not only_test_set:
		print('.', end='', flush=True)
		valid_dataset = CurveML(path=curveml_path, partition='validation')
		print('.', end='', flush=True)
		train_dataset = CurveML(path=curveml_path, partition='training')		# keep this one as the last because it's pretty slow
	else:
		print(f'Warning: using only test set for dataloaders...')

	print('. DONE!', flush=True)

	if not only_test_set:
		train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
		val_dataloader   = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
	test_dataloader  = DataLoader(test_dataset,  batch_size=bs, shuffle=only_test_set)

	if only_test_set:
		train_dataloader = test_dataloader
		val_dataloader   = test_dataloader

	return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

	test_load      = False
	test_show      = False
	test_one_batch = True

	if test_load:
		dataset_path = Path('/tmp/geometric-primitives-classification/geometric-primitives-dataset-v1.0-wo-splines')
		#save_dataset_partitions(dataset_path)
		test_data = load_dataset('./', 'test.xz')
		sys.exit()

	if test_show:
		srcdir = Path('curveml-train-geom-petal-036142')
		points = genfromtxt(srcdir / 'point_cloud_clean.csv', delimiter=',')
		show_3d_image(points, 'geom-petal')
		points = genfromtxt(srcdir / 'point_cloud.csv', delimiter=',')
		show_3d_image(points, 'geom-petal')

		srcdir = Path('curveml-test-cassinian-oval-027194')
		points = genfromtxt(srcdir / 'point_cloud_clean.csv', delimiter=',')
		show_3d_image(points, 'cassinian-oval')
		points = genfromtxt(srcdir / 'point_cloud.csv', delimiter=',')
		show_3d_image(points, 'cassinian-oval')

	if test_one_batch:
		#curveml_path = Path('/tmp/geometric-primitives-classification/geometric-primitives-dataset-v1.0-wo-splines')
		curveml_path = Path('./')
		trainDataLoader, valDataLoader, testDataLoader = create_curveml_dataloaders(curveml_path, bs=128)

		one_batch = next(iter(testDataLoader))
		show_one_batch(one_batch)



