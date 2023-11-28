#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from pathlib import Path

'''
if len(sys.argv) < 2:
	print(f'\n\tUsage: {sys.argv[0]} <stem>\n')
	sys.exit()

stem = Path(sys.argv[1]).stem
param = stem.split('-')[0]
'''

limits = {
		'a': (0, 0.050),
		'b': (0, 0.015),
		'x': (0, 0.010),
		'y': (0, 0.006),
}



# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

# Create figure and axis objects
fig, ax = plt.subplots()

#labels = ['Training Loss', 'Validation Loss']
labels = ['a', 'b', 'y']


# Set plot title
plt.title(f'PointNet Regression Loss (MSE)')

# Set x-axis label
plt.xlabel('Thousands of Steps')

# Set y-axis label
plt.ylabel('Loss')

# Set x and y axis limits
plt.xlim(0, 200)
#plt.ylim(limits[param][0], limits[param][1])  # Update with your desired y-axis limits
plt.ylim(0, 0.01)

fn_prefix = 'wandb_export_'
fn_middle = '-250px-'
fn_suffix = '-loss-2023-11-28.csv'

for run in ['train', 'valid']:
	for param in ['a', 'b', 'y']:
		fn = fn_prefix + param + fn_middle + run + fn_suffix
		print(f'Processing {fn} for parameter {param}-{run}')
	
		# Read data from text file
		#data = np.genfromtxt(stem + '.txt', skip_header=1)
		data = pd.read_csv(fn, header=None)
	
		print(f'{data.shape[0]} rows')
		print(f'{data}')
		print(f'{data[0][1:].values = }')
		print(f'{data[1][1:].values = }')
		steps  = data[0][1:].values
		losses = data[1][1:].values
	
		for j in range(len(losses)):
			steps[j]  = round(float(steps[j])/1000)
			losses[j] = float(losses[j])
	
		print(f'{steps  = }')
		print(f'{losses = }')
	
		# Plot each loss column
		#for i in range(losses.shape[1]):
		#	ax.plot(data[0], data[1], label=labels[i])
		ax.plot(steps, losses, label=f'{str(run).title()} Loss for parameter {param}')

# Add legend
plt.legend()

fig.set_size_inches(19.2, 10.8)
fig.savefig('aby-valid-loss.png', dpi=100)

# Show the plot
plt.show()

# Extract epoch values from the first column
#epochs = data[:, 0]

# Extract loss values from the second to nth columns
#losses = data[:, 1:]





