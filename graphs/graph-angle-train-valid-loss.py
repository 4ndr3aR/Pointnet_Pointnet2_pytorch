#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

param = 'angle'
stem = f'{param}-train-valid-loss'

# Read data from text file
data = np.genfromtxt(stem + '.txt', skip_header=1)

# Extract epoch values from the first column
epochs = data[:, 0]

# Extract loss values from the second to nth columns
losses = data[:, 1:]

# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

# Create figure and axis objects
fig, ax = plt.subplots()

labels = ['Training Loss', 'Validation Loss']

# Plot each loss column
for i in range(losses.shape[1]):
	label = [f'"angle" {run.title()} Loss' for run in ['Training', 'Validation']][i]
	ax.plot(epochs, losses[:, i], label=label, color=select_color(param, i))

# Set plot title
#plt.title(f'PointNet MSE Regression Losses')

# Set x-axis label
#plt.xlabel('Epoch')

# Set y-axis label
#plt.ylabel('Loss')

apply_wandb_graph_style(ax, plt, title='PointNet MSE Regression Losses')

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0., 2050.)  # Update with your desired y-axis limits

plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.11)

# Add legend
#plt.legend()

fig.set_size_inches(19.2, 10.8)
fig.savefig(stem + '.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))

# Show the plot
plt.show()


