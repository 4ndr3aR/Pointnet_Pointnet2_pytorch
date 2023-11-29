#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from colors import colors, hex_to_rgba, select_color

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
		color = select_color(param, i)
		img   = ax.plot(epochs, losses[:, i], label=label, color=color)

		# set imshow outline
		for spine in img[0].axes.spines.values():
			spine.set_edgecolor(hex_to_rgba(colors['axis']))    



# Set plot border visibility
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Set ticks and labels params
ax.tick_params(axis='x', colors=hex_to_rgba(colors['grid']), size=15, width=2, pad=15, direction='inout', labelsize=32)		# size = tick length
ax.tick_params(axis='y', colors=hex_to_rgba(colors['grid']), size=0 , width=2, pad=15, direction='inout', labelsize=32)

# Set plot ticks
for ticklabel in plt.gca().get_xticklabels():
    ticklabel.set_color(hex_to_rgba(colors['labels']))
for ticklabel in plt.gca().get_yticklabels():
    ticklabel.set_color(hex_to_rgba(colors['labels']))

# Set plot grid
plt.grid(axis='x', which='major', color=hex_to_rgba(colors['background']), linestyle=':', linewidth=0.5)
plt.grid(axis='y', which='major', color=hex_to_rgba(colors['grid']), linewidth=1.5)

# Change font
titlefont = {'fontname': 'Source Sans Pro'}

# Set plot title
#plt.title(f'PointNet Regression Loss (MSE) for parameter "{param}"', color=hex_to_rgba(colors['title']), fontsize=32, **titlefont)
plt.title(f'Train/Validation Losses', color=hex_to_rgba(colors['title']), fontsize=48, fontweight='normal', **titlefont)

# Set x-axis label
plt.xlabel('Epoch', color=hex_to_rgba(colors['title']))

# Set y-axis label
plt.ylabel('Loss', color=hex_to_rgba(colors['title']))

# Set plot background color
ax.patch.set_facecolor(hex_to_rgba(colors['background']))

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0., max_val + 0.01*max_val)

# Add legend
leg = plt.legend(labelcolor='linecolor', facecolor=hex_to_rgba(colors['background']), prop={'size': 32}, loc='upper right', ncol=2)

# Save figure
fig.set_size_inches(19.2, 10.8)
fig.savefig(f'{"".join(paramlist)}-train-valid-loss.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))





#plt.tight_layout()

# Show the plot
plt.show()


