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
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler		# share the same color cycle
#ax1.plot(x, y1, 'g-')
#ax2.plot(x, y2, 'b-')

#ax1.set_xlabel('X data')
#ax1.set_ylabel('Y1 data', color='g')
#ax2.set_ylabel('Y2 data', color='b')



#labels = ['Training Loss', 'Validation Loss']
labels = ['a', 'b', 'y']


# Set plot title
plt.title(f'PointNet Classification Loss')

# Set x-axis label
#plt.xlabel('Thousands of Steps')
ax1.set_xlabel('Epochs')

# Set y-axis label
#plt.ylabel('Loss')
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Validation Loss')

# Set x and y axis limits
#plt.xlim(-0.5, 200)
#plt.ylim(limits[param][0], limits[param][1])  # Update with your desired y-axis limits
#plt.ylim(0, 0.01)
ax1.set_xlim(-0.1, 6.1)
ax2.set_xlim(-0.1, 6.1)
ax1.set_ylim(-0.025, 2.5)
ax2.set_ylim(-0.00001, 0.01)



fn_prefix = 'wandb_export_'
fn_middle = '-250px-'
fn_suffix = '-loss-2023-11-28.csv'

lines = []

for run in ['train', 'valid']:
	curr_ax = ax1 if run == 'train' else ax2
	for param in ['cls']:
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
			steps[j]  = float(steps[j])/14000. 	# 14k step = 1 epoch
			losses[j] = float(losses[j])
	
		print(f'{steps  = }')
		print(f'{losses = }')
	
		# Plot each loss column
		#for i in range(losses.shape[1]):
		#	ax.plot(data[0], data[1], label=labels[i])
		line = curr_ax.plot(steps, losses, label=f'{str(run).title()} Loss for classification')
		lines.append(line[0])


for l in lines:
	print(f'{l = }')
	print(f'{l.get_label() = }')
labs = [line.get_label() for line in lines]
leg = ax1.legend(lines, labs, loc=0, frameon=True)
leg.get_frame().set_edgecolor('k')
#leg.get_frame().set_linewidth(0.5)


# Add legend
#ax1.legend()
#ax2.legend(loc=0)

fig.set_size_inches(19.2, 10.8)
fig.savefig('cls-train-valid-loss.png', dpi=100)

# Show the plot
plt.show()

# Extract epoch values from the first column
#epochs = data[:, 0]

# Extract loss values from the second to nth columns
#losses = data[:, 1:]





