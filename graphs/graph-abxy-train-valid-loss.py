#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style


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


fn_suffix = '-train-valid-loss.txt'
#paramlist = ['a', 'b', 'x', 'y']
#paramlist = ['a', 'b']
paramlist = ['x', 'y']
max_val   = -1

for param in paramlist:
	fn =  f'{param}{fn_suffix}'
	print(f'Processing {fn} for parameter {param}')

	# Read data from text file
	data = np.genfromtxt(fn, skip_header=1)

	# Extract epoch values from the first column
	epochs = data[:, 0]

	# Extract loss values from the second to nth columns
	losses = data[:, 1:]

	for i in range(losses.shape[1]):
		max_val = np.max(losses) if max_val < np.max(losses) else max_val
		label = [f'"{param}" {run.title()} Loss' for run in ['Training', 'Validation']][i]
		#label = label if not 'x' in param or not 'y' in param else label.replace('x', 'trans_x').replace('y', 'trans_y')
		color = select_color(param, i)
		img   = ax.plot(epochs, losses[:, i], label=label, color=color)

		# set imshow outline
		for spine in img[0].axes.spines.values():
			spine.set_edgecolor(hex_to_rgba(colors['axis']))    


apply_wandb_graph_style(ax, plt, title='PointNet MSE Regression Losses')

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0., max_val + 0.01*max_val)

plt.subplots_adjust(bottom=0.15)
if 'x' in paramlist:
	plt.subplots_adjust(left=0.11)
else:
	plt.subplots_adjust(left=0.10)

# Save figure
fig.set_size_inches(19.2, 10.8)
fig.savefig(f'{"".join(paramlist)}-train-valid-loss.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))





#plt.tight_layout()

# Show the plot
plt.show()


