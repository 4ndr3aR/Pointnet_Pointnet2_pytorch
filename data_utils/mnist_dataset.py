#!/usr/bin/env python3

import numpy as np
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def transform_img2pc(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    return indices.astype(np.float32)

def show_3d_image(img, label):
	pc  = img
	print(f'Showing image: {pc.shape}')
	if type(pc) != np.ndarray:
		pc  = img.numpy()
	if pc.shape[1] == 2:
		pc = np.hstack((pc, np.zeros((pc.shape[0], 1))))

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


if __name__ == '__main__':
	#show_number_of_points_histogram()
	train_dataloader, val_dataloader, test_dataloader = create_3dmnist_dataloaders()
	#img, label = next(iter(train_dataloader))[0], next(iter(train_dataloader))[1]

	#for (image, label) in list(enumerate(train_loader))[:1000]:
	dataset = train_dataloader.dataset

	#img, label = train_dataloader[2][0], train_dataloader[2][1]
	print(f'img shape: {img.shape} - label: {label}')
	show_3d_image(img, label)

