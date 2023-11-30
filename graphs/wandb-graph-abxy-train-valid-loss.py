#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

if len(sys.argv) < 2:
	print(f'Usage: {sys.argv[0]} <run>')
	exit(1)

run = sys.argv[1]

if run != "train" and run != "valid":
	print(f'Usage: {sys.argv[0]} <run>')
	exit(1)

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

labels = ['Training Loss', 'Validation Loss']

# Who knows what's this for...
ax.tick_params(axis='x', which='major', bottom=True)

fn_prefix = 'wandb-csv/wandb_export_'
fn_middle = '-250px-'
fn_suffix = '-loss-2023-11-28.csv'

#fn_suffix = '-train-valid-loss.txt'
paramlist = ['a', 'b', 'x', 'y']
#paramlist = ['a', 'b']
#paramlist = ['x', 'y']
max_val   = -1
max_epoch = -1

for param in paramlist:
	fn =  f'{fn_prefix}{param}{fn_middle}{run}{fn_suffix}'
	print(f'Processing {fn} for parameter {param}')

	'''
	# Read data from text file
	data = np.genfromtxt(fn, skip_header=1)

	# Extract epoch values from the first column
	epochs = data[:, 0]

	# Extract loss values from the second to nth columns
	losses = data[:, 1:]
	'''

	data = pd.read_csv(fn)

	print(f'{data.shape[0]} rows')
	print(f'{data}')
	print(f'{data.iloc[:,0].values = }')
	print(f'{data.iloc[:,1].values = }')
	steps  = data.iloc[:,0].values
	losses = data.iloc[:,1].values

	steps = steps.astype(float)
	for j in range(len(losses)):
		steps[j]  = 1.*float(steps[j])/14000.      # 14k step = 1 epoch
		losses[j] = float(losses[j])

	epochs = steps

	print(f'{steps  = }')
	print(f'{losses = }')

	print(f'{losses.shape = }')

	#for i in range(losses.shape[1]):
	if True:
		max_val   = np.max(losses) if max_val   < np.max(losses) else max_val
		max_epoch = np.max(epochs) if max_epoch < np.max(epochs) else max_epoch
		#label = [f'"{param}" {run.title()} Loss' for run in ['Training', 'Validation']][i]
		label = f'"{param}" {run.title()} Loss'
		color = select_color(param, selector=1)
		#img   = ax.plot(epochs, losses[:, i], label=label, color=color)
		img   = ax.plot(epochs, losses, label=label, color=color)

		# set imshow outline
		for spine in img[0].axes.spines.values():
			spine.set_edgecolor(hex_to_rgba(colors['axis']))    


apply_wandb_graph_style(ax, plt, title=f'ResNet-101 {run.title()} Losses')

# Set x and y axis limits
plt.xlim(-0.001*max_epoch,  max_epoch)
plt.ylim(-0.004*max_val,    max_val + 0.01*max_val)


plt.subplots_adjust(bottom=0.15)
if run == "valid":
	plt.subplots_adjust(left=0.11)

# Save figure
fig.set_size_inches(19.2, 10.8)
#fig.savefig(f'{"".join(paramlist)}-{run}-loss.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))
fig.savefig(f'wandb-{"".join(paramlist)}-{run}-loss.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))





#plt.tight_layout()

# Show the plot
plt.show()


