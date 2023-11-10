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

from mnist_dataset import transform_img2pc, show_3d_image, get_random_sample, show_number_of_points_histogram

def load_dataset(path, fname):
	data = None
	if Path(Path(path) / fname).is_file() and Path(fname).suffix == '.xz':
		print(f'Reading LZMA compressed dataset...')
		with lzma.open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	else:
		print(f'Reading uncompressed dataset...')
		with open(Path(path) / Path(str(fname)), 'rb') as fhandle:
			data = pickle.load(fhandle)
			print(f'Read {len(data)} samples - {type(data[0]) = } - {len(data[0]) = } - {data[0][0].shape = } - {data[0][1] = } - {data[0][2] = }')
	return data

'''
def transform_img2pc(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    return indices.astype(np.float32)

def show_3d_image(img, label):
	#pc  = train_dataset[5][0].numpy()
	#lbl = train_dataset[5][1]
	pc  = img.numpy()
	lbl = label
	fig = plt.figure(figsize=[7,7])
	ax  = plt.axes(projection='3d')
	sc  = ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=pc[:,0] ,s=80, marker='o', cmap="viridis", alpha=0.7)
	ax.set_zlim3d(-1, 1)
	plt.title(f'Label: {lbl}')
	plt.show()

def show_number_of_points_histogram():
	dataset = MNIST(root='./data', train=True, download=True)
	len_points = []
	# loop over samples
	for idx in range(len(dataset)):
		img,label = dataset[idx]
		pc = transform_img2pc(img)
		len_points.append(len(pc))
    
	h = plt.hist(len_points)
	plt.title('Histogram of amount of points per number')
	plt.show()								# this shows us that the number of points should be 200

def create_3dmnist_dataloaders(bs):
	train_dataset = MNIST(root='./data/MNIST', download=True, train=True)
	test_dataset  = MNIST(root='./data/MNIST', download=True, train=False)
	dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

	number_of_points = 200

	dataset_3d = MNIST3D(dataset, number_of_points)
	l_data = len(dataset_3d)
	train_dataset, val_dataset, test_dataset = random_split(dataset_3d,
                                          [round(0.8*l_data), round(0.1*l_data), round(0.1*l_data)],
                                          generator=torch.Generator().manual_seed(1))

	train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
	val_dataloader   = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
	test_dataloader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False)

	return train_dataloader, val_dataloader, test_dataloader

def get_random_sample(dataset):
	random_index  = int(np.random.random() * len(dataset))
	random_sample = dataset[random_index]
	img, label    = random_sample[0], random_sample[1]
	print(f'img shape: {img.shape} - label: {label}')
	return img, label

class MNIST3D(Dataset):
    """3D MNIST dataset."""
    
    NUM_CLASSIFICATION_CLASSES = 10
    POINT_DIMENSION = 3

    def __init__(self, dataset, num_points):
        self.dataset = dataset
        self.number_of_points = num_points

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img,label = self.dataset[idx]
        pc = transform_img2pc(img)
        
        if self.number_of_points-pc.shape[0]>0:
            # Duplicate points
            sampling_indices = np.random.choice(pc.shape[0], self.number_of_points-pc.shape[0])
            new_points = pc[sampling_indices, :]
            pc = np.concatenate((pc, new_points),axis=0)
        else:
            # sample points
            sampling_indices = np.random.choice(pc.shape[0], self.number_of_points)
            pc = pc[sampling_indices, :]
            
        pc = pc.astype(np.float32)
        # add z
        noise = np.random.normal(0,0.05,len(pc))
        noise = np.expand_dims(noise, 1)
        pc = np.hstack([pc, noise]).astype(np.float32)
        pc = torch.tensor(pc)
        
        return pc, label
'''

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

	def __init__(self, path, partition, max_points=MAX_POINTS):
		self.path = path
		self.max_points = max_points
		#self.dataset = None
		self.dataset = load_dataset(path, partition + '.xz')

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx, debug=False):
		points,label,fpath = self.dataset[idx]
		if debug:
			print(f'__getitem__() idx: {idx} - {type(self.dataset[idx]) = } - {len(self.dataset[idx]) = } - {points.shape = } - {label = } - {fpath = }')
		points = np.hstack((points, np.zeros((points.shape[0], 1))))
		if debug:
			print(f'__getitem__() idx: {idx} - {type(self.dataset[idx]) = } - {len(self.dataset[idx]) = } - {points.shape = } - {label = } - {fpath = }')


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

		'''
		pc = pc.astype(np.float32)
		# add z
		noise = np.random.normal(0,0.05,len(pc))
		noise = np.expand_dims(noise, 1)
		pc = np.hstack([pc, noise]).astype(np.float32)
		pc = torch.tensor(pc)
		'''

		return points, label

def read_curveml_dataset(path):
	dataset = []

	counter = 0

	csv_files = Path(path).rglob('point_cloud*.csv')
	print(f'read_curveml_dataset() - {len(list(csv_files))} files found')

	csv_files = Path(path).rglob('point_cloud*.csv')
	for idx, file in enumerate(list(csv_files)):
		#print(f'read_curveml_dataset() - {idx} - {file}')
		if file.is_file():
			points = genfromtxt(file, delimiter=',')
			fpath  = file.parent.stem			# 028676
			label  = file.parent.parent.stem		# cassinian-oval
			counter += 1
			dataset.append((points, label, fpath))
			if counter % 1000 == 0:
				print(f'read_curveml_dataset() - {counter} files processed ({fpath} - {label} - {points.shape})')	
			#if counter % 5000 == 0:
			#	break
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

def create_curveml_dataloaders(curveml_path, bs):
	#train_dataset = CurveML(path=curveml_path, partition='training')
	#valid_dataset = CurveML(path=curveml_path, partition='validation')
	test_dataset  = CurveML(path=curveml_path, partition='test')
	'''
	dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

	number_of_points = 200

	dataset_3d = MNIST3D(dataset, number_of_points)
	l_data = len(dataset_3d)
	train_dataset, val_dataset, test_dataset = random_split(dataset_3d,
                                          [round(0.8*l_data), round(0.1*l_data), round(0.1*l_data)],
                                          generator=torch.Generator().manual_seed(1))
	'''

	#train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
	#val_dataloader   = DataLoader(val_dataset,   batch_size=bs, shuffle=True)
	test_dataloader  = DataLoader(test_dataset,  batch_size=bs, shuffle=True)
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
		print(f'one_batch: {type(one_batch)}')
		print(f'one_batch: {len(one_batch)}')
		print(f'one_batch: {type(one_batch[0])}')
		print(f'one_batch: {one_batch[0].shape}')
		print(f'one_batch: {type(one_batch[1])}')
		print(f'one_batch: {one_batch[1]}')
		for idx in range(len(one_batch[0])):
			points = one_batch[0][idx]
			label  = one_batch[1][idx]
			print(f'Points shape: {points.shape} - label: {label}')
			show_3d_image(points, label)

	'''
	#show_number_of_points_histogram()
	train_dataloader, val_dataloader, test_dataloader = create_3dmnist_dataloaders()
	#img, label = next(iter(train_dataloader))[0], next(iter(train_dataloader))[1]

	#for (image, label) in list(enumerate(train_loader))[:1000]:
	dataset = train_dataloader.dataset

	#img, label = train_dataloader[2][0], train_dataloader[2][1]
	print(f'img shape: {img.shape} - label: {label}')
	show_3d_image(img, label)
	'''







