#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Read data from text file
data = np.genfromtxt('cls-train-valid-acc.txt', skip_header=1)

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
    ax.plot(epochs, losses[:, i], label=labels[i])

# Set plot title
plt.title('PointNet Classification Accuracy')

# Set x-axis label
plt.xlabel('Epoch')

# Set y-axis label
plt.ylabel('Accuracy')

# Set x and y axis limits
plt.xlim(0, 200)
plt.ylim(0.8, 1.01)  # Update with your desired y-axis limits

# Add legend
plt.legend()

fig.set_size_inches(19.2, 10.8)
fig.savefig('cls-train-valid-acc.png', dpi=100)

# Show the plot
plt.show()


