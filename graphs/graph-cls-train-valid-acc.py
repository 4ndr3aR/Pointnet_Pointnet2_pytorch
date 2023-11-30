#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

# Read data from text file
param = 'cls'
data = np.genfromtxt(f'{param}-train-valid-acc.txt', skip_header=1)

# Extract epoch values from the first column
epochs = data[:, 0]

# Extract loss values from the second to nth columns
losses = data[:, 1:]

# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

# Create figure and axis objects
fig, ax = plt.subplots()

labels = ['Training Acc.', 'Validation Acc.']

# Plot each loss column
for i in range(losses.shape[1]):
    ax.plot(epochs, losses[:, i], label=labels[i], color=select_color(param, i))

# Set plot title
#plt.title('PointNet Classification Accuracy')

# Add legend
#plt.legend(loc='lower right')

apply_wandb_graph_style(ax, plt, loc='lower right', title='PointNet Classification Accuracies')

# Set x-axis label
plt.xlabel('Epoch')

# Set y-axis label
plt.ylabel('Accuracy')

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0.8, 1.01)  # Update with your desired y-axis limits

plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.10)

fig.set_size_inches(19.2, 10.8)
fig.savefig(f'{param}-train-valid-acc.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))

# Show the plot
plt.show()


