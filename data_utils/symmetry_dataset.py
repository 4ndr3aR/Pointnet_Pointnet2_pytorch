#!/usr/bin/env python3

import sys

import numpy as np

import torch
import torchvision

#import cv2

import lzma
import pickle

from pathlib import Path
from numpy import genfromtxt

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from collections import Counter			# just to subtract lists of strings

import pandas as pd
pd.options.display.precision = 3

'''
import re
import builtins
from functools import wraps

original_print = builtins.print

def fprint(print_func, precision=2):
    @wraps(print_func)
    def wrapper(*args, **kwargs):
        def repl(match):
            float_number = float(match.group(1))
            #ret = "{" + str(float_number) + ":." + str(precision) + "f}"
            precision_term = "{:." + str(precision) + "f}"
            ret = ''.join(precision_term.format(float_number).rjust(precision+3))
            #original_print(ret)
            return f"{ret}"
        
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = re.sub(r'([-+]?\d*\.\d+([eE][-+]?\d+)?)', repl, arg)
            new_args.append(arg)
        
        return print_func(*new_args, **kwargs)
    return wrapper

#builtins.print = fprint(print)
'''

def to_precision(lst, precision=2):
	print(f'to_precision() received type: {type(lst)} - lst: {lst}')
	lst_len = len(lst) if isinstance(lst, list) else (lst.shape[0] if isinstance(lst, torch.Tensor) and len(lst.shape) != 0 else 0)
	print(f'to_precision() received type: {type(lst)} - {lst_len = } - lst: {lst}')

	str_lst = []
	precision_term = "{:." + str(precision) + "f}"

	if isinstance(lst, torch.Tensor) and len(lst.shape) == 0:				# e.g. torch.Tensor(7)
		#str_data = ', '.join(precision_term.format(float(lst)).rjust(precision+3))
		str_data = precision_term.format(float(lst)).rjust(precision+3)
		print(f'to_precision() returning {str_data = }')
		return str_data

	if (isinstance(lst, torch.Tensor) and len(lst.shape) != 0) or (isinstance(lst, list) and len(lst) > 0):
		if isinstance(lst[0], list) or len(lst[0].shape) != 0:
			print('passo di qui')
			return [to_precision(x) for x in lst]


	if (isinstance(lst, torch.Tensor) and len(lst.shape) != 0) or (isinstance(lst, list) and len(lst) > 0):
		print('passo di qua')
		#str_lst.append(', '.join(precision_term.format(lst).rjust(precision+3)))
		str_lst.append(', '.join(precision_term.format(float(x)).rjust(precision+3) for x in lst))
		'''
		for elem in lst:
			str_lst.append(', '.join(precision_term.format(float(elem)).rjust(precision+3)))
		'''
	else:
		str_lst.append(', '.join(precision_term.format(float(x)).rjust(precision+3) for x in lst))
	print(f'{str_lst = }')
	'''
	for sublist in lst:
		for x in list(sublist):
			print(f'{type(x) = } - {x = }')
			print(f'{precision_term.format(x).rjust(precision+3) = }')
		[print(type(x)) for x in sublist]
		str_lst.append(', '.join([precision_term.format(float(x)).rjust(precision+3) for x in sublist]))
	'''
	return str_lst
'''
def to_precision2(data, precision=2):
	print(data)
def to_precision(data, precision=2):
	if isinstance(data, float):
		return "{:.{}f}".format(data, precision) + ", "
	elif isinstance(data, int):
		return str(data) + ", "
	elif isinstance(data, str):
		return data + ", "
	elif isinstance(data, (list, tuple)):
		result = ""
		for element in data:
			result += to_precision(element, precision)
		return '[' + result.rstrip(", ") + '], \n'
	elif isinstance(data, (np.ndarray, np.matrix)):
		result = ""
		for row in data:
			result += to_precision(row, precision)
		return '((' + result.rstrip(", ") + ')), \n'
	elif isinstance(data, dict):
		result = ""
		for key, value in data.items():
			result += "{}: {} ".format(key, to_precision(value, precision))
		return '{' + result.rstrip(", ") + '}, \n'
	else:
		return str(data)
'''





if __name__ == '__main__':
	from mnist_dataset import show_3d_image as mnist_show_3d_image
	from mnist_dataset import transform_img2pc, get_random_sample, show_number_of_points_histogram
else:
	from .mnist_dataset import show_3d_image as mnist_show_3d_image
	from .mnist_dataset import transform_img2pc, get_random_sample, show_number_of_points_histogram


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
	MAX_POINTS = 1000

	#LABELS = ['cassinian-oval', 'cissoid', 'citrus', 'egg', 'geom-petal', 'hypocycloid', 'mouth', 'spiral']
	LABELS = ['astroid', 'geometric_petal']

	def __init__(self, path, partition, gt_columns=None, max_points=MAX_POINTS, labels=LABELS, add_noise=False):
		self.path       = path
		self.labels     = labels
		#self.vocab      = [[], labels]			# because of: ```if is_listy(self.vocab): self.vocab = self.vocab[-1]```
		self.add_noise  = add_noise
		self.dataset    = load_dataset(path, partition + '.xz')
		self.gt_columns = gt_columns			# e.g. 'type', 'popx', 'popy', 'popz', 'nx', 'ny', 'nz', 'rot' (only for the first row in the gt dataframe)
		self.max_points = max_points

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx, debug=False):
		debug = True
		# Dataframe columns: ['angle', 'trans_x', 'trans_y', 'a', 'b', 'n_petals', 'label', 'fpath', 'points']
		#row = self.dataset.iloc[idx]
		gt_columns = self.gt_columns
		row = self.dataset[idx]
		points,label,split = row['points'],row['label'],row['split']
		lbl = torch.tensor(self.labels.index(label))
		#angle,trans_x,trans_y,a,b,n_petals = row['angle'],row['trans_x'],row['trans_y'],row['a'],row['b'],row['n_petals']
		#angle,trans_x,trans_y,a,b,n_petals = row['angle'],row['trans_x'],row['trans_y'],row['a'],row['b'],row['n_petals']
		label, split, points, gt = row['label'], row['split'], row['points'][0], row['gt']

		'''
		if debug:
			print(f'1. __getitem__() idx: {idx} - {type(row) = } - {len(row) = } - {points.shape = } - {label = } - {split = } - {lbl = }')
		points = np.hstack((points, np.zeros((points.shape[0], 1))))
		'''
		# The symmetry dataset already contains real 3D point clouds, so no need for the np.hstack as in the CurveML dataset...
		if debug:
			print(f'1. __getitem__() idx: {idx} - {type(row) = } - {len(row) = } - {points.shape = } - {label = } - {split = } - {lbl = }')
			for gt_row in gt.iterrows():
				idx     = gt_row[0]
				row_arr = gt_row[1].values
			#print(f'__getitem__() idx: {idx} - {angle = } - {trans_x = } - {trans_y = } - {a = } - {b = } - {n_petals = }')
			print(f'2. __getitem__() idx: {idx} - {label = } - {split = } - {points.shape = } - {gt.shape = } - {row_arr = }')

		if self.max_points - points.shape[0] > 0:
			# Duplicate points
			sampling_indices = np.random.choice(points.shape[0], self.max_points - points.shape[0])
			if debug:
				print(f'3. __getitem__() idx: {idx} - {len(sampling_indices) = } - {sampling_indices = }')
			new_points = points[sampling_indices, : ]
			points = np.concatenate((points, new_points), axis=0)
		else:
			# sample points
			sampling_indices = np.random.choice(points.shape[0], self.max_points)
			if debug:
				print(f'4. __getitem__() idx: {idx} - {len(sampling_indices) = } - {sampling_indices = }')
			points = points[sampling_indices, :]

		points = points.astype(np.float32)
		points = torch.tensor(points)
		if self.add_noise:
			#points = points + (0.01**0.5)*torch.randn(points.shape[0], points.shape[1])		# high
			points = points + (0.001**0.5)*torch.randn(points.shape[0], points.shape[1])		# perfect

		if gt_columns:
			if isinstance(gt_columns, list) and len(gt_columns) == 1:
				gt_columns = gt_columns[0]
			if gt_columns == 'label':
				gt = lbl
			else:
				if isinstance(gt_columns, list):
					if 'cls' in gt_columns or 'class' in gt_columns:			# clean up gt_columns so it can be used to directly address gt_df
						gt_columns = list((Counter(gt_columns) - Counter(['cls', 'class'])).elements())		# this is a very pythonic list subtraction
				if debug:
					print(f'5. __getitem__() gt_columns: {gt_columns} - {row = }')
				gt_df     = row['gt']
				gt_df_col = gt_df[gt_columns]
				if debug:
					print(f'6. __getitem__() gt_columns: {gt_columns} - gt_df_col: {gt_df_col}')
				if isinstance(gt_columns, str):
					if gt_columns in ['popx', 'popy', 'popz']:
						gt = gt_df_col.unique()[0]				# there is always some difference between axis and plane points but it's ~1e-6
					else:
						print(f'WARNING. Returning only the first row of the GT column for regression testing purposes!')
						gt = gt_df_col[0]					# this is only for test purposes and should have some warning
					if debug:
						print(f'7. __getitem__() gt_columns: {gt_columns} - gt_df_col.values:\n{gt_df_col.values}')
				elif isinstance(gt_columns, list):
					gt_cls = None			# class, categorical, one for each figure
					gt_arr = []			# popx, popy, popz, just three float for each figure
					gt_mat = []			# nx, ny, nz, three float for each symmetry
					gt_cat = []			# type, rot, categorical*, one for each symmetry
									# *we observe that, after a .fillna(-1), there are only 6 possible values for rot
									# in the dataset: [-1.0, 0.628319, 0.785398, 1.047198, 1.570796, 3.141593] ==
									# == [-1, π/5, π/4, π/3, π/2, π] so we can encode them just as [0, 5, 4, 3, 2, 1]
					#gt_mat_tmp = []
					for idx,col in enumerate(gt_columns):
						gt_df_col_vals = gt_df[col].values
						print(f'6.{idx}. __getitem__() gt_columns: {gt_columns} - gt_df[{col}].values: {gt_df_col_vals}')
						if 'pop' in col:
							gt_arr.append(gt_df[col].unique()[0])
						elif 'rot' in col:
							gt_cat.append(list(gt_df_col_vals))
						elif 'type' in col:
							gt_cat.append([0 if val == 'plane' else 1 for val in list(gt_df_col_vals)])
						else:
							#gt_mat_tmp.append(list(gt_df_col_vals))
							gt_mat.append(list(gt_df_col_vals))
					#gt_mat = gt_mat_tmp

					if 'cls' in self.gt_columns or 'class' in self.gt_columns:			# use the original here!
						gt_cls = lbl

					if debug:
						print(f'7. __getitem__() gt_columns: {gt_columns}\ngt_arr: {gt_arr}\ngt_mat: {gt_mat}\ngt_cat: {gt_cat}\ngt_cls: {gt_cls}')
					gt = [torch.tensor(gt_arr), torch.tensor(gt_mat), torch.tensor(gt_cat), torch.tensor(gt_cls)]
				if debug:
					print(f'9. __getitem__() gt_columns: {gt_columns} - gt_df_col.values[0]: {gt_df_col.values[0]}')
		else:
			gt = lbl #row['label']
		print(f'10. __getitem__() gt_columns: {gt_columns} - returning GT with shape: {[list(itm.shape) for itm in gt]} - GT: \n{to_precision(gt)}')
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

def save_dataset(dataset, path, fname, also_pickle=False):
	if also_pickle:
		print(f'Writing uncompressed dataset...')
		with open(Path(path) / Path(str(fname) + '.pickle'), 'wb') as fhandle:
			pickle.dump(dataset, fhandle)
	print(f'Writing compressed dataset...')
	with lzma.open(Path(path) / Path(str(fname) + '.xz'), 'wb') as fhandle:
		pickle.dump(dataset, fhandle)

def read_and_save_dataset_partitions(dataset_path):
	test_dataset     = read_symmetry_dataset(dataset_path / 'test')
	save_dataset(test_dataset, './', 'test'  , also_pickle=False)
	valid_dataset    = read_symmetry_dataset(dataset_path / 'valid')
	save_dataset(valid_dataset, './', 'valid', also_pickle=False)
	train_dataset    = read_symmetry_dataset(dataset_path / 'train')
	save_dataset(train_dataset, './', 'train', also_pickle=False)
	return train_dataset, valid_dataset, test_dataset

def create_symmetry_dataloaders(symmetry_path, bs, gt_columns=None, only_test_set=False, valid_and_test_sets=False):
	train_dataset,    val_dataset,    test_dataset    = None, None, None
	train_dataloader, val_dataloader, test_dataloader = None, None, None

	print('.', end='', flush=True)
	test_dataset = Symmetry(path=symmetry_path, gt_columns=gt_columns, partition='test')

	if not only_test_set:
		print('.', end='', flush=True)
		val_dataset = Symmetry(path=symmetry_path, gt_columns=gt_columns, partition='valid')
		print('.', end='', flush=True)
		train_dataset = Symmetry(path=symmetry_path, gt_columns=gt_columns, partition='train')		# keep this one as the last because it's pretty slow
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

	if valid_and_test_sets:
		print('.', end='', flush=True)
		val_dataset    = Symmetry(path=symmetry_path, gt_columns=gt_columns, partition='valid')
		val_dataloader = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
		print(f'Warning: using only valid and test set for dataloaders...')

	return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

	test_read      = False
	test_write     = True
	test_load      = False
	test_show      = False
	test_one_batch = False

	if test_read:
		print(f'Testing the read_symmetry_dataset() function...')
		dataset_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/symmetries-dataset-astroid-geom_petal-10k')
		test_data = read_symmetry_dataset(dataset_path)
		print(f'read_symmetry_dataset() complete')
		sys.exit()

	if test_write:
		print(f'Testing the read_and_save_dataset_partitions() function...')
		dataset_path = Path('/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/symmetries-dataset-astroid-geom_petal-10k')
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
		trainDataLoader, valDataLoader, testDataLoader = create_symmetry_dataloaders(symmetry_path, gt_columns='label', bs=16, only_test_set=True)

		one_batch = next(iter(testDataLoader))
		show_one_batch(one_batch)



