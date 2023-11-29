#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from colors import colors, hex_to_rgba, select_color, apply_wandb_graph_style

param = 'n_petals'
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
    ax.plot(epochs, losses[:, i], label=labels[i], color=select_color(param, i))

apply_wandb_graph_style(ax, plt)

# Set plot title
plt.title('PointNet Regression Loss (MSE) for parameter "n"')

# Set x-axis label
plt.xlabel('Epoch')

# Set y-axis label
plt.ylabel('Loss')

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0., 2.56)  # Update with your desired y-axis limits

# Add legend
#plt.legend()

fig.set_size_inches(19.2, 10.8)
fig.savefig(stem + '.png', dpi=100)

# Show the plot
plt.show()


