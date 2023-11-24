#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Read data from text file
data = np.loadtxt('cls-train-loss.txt', skiprows=1)

# Extract epochs and loss values
epochs = data[:, 0]
loss_values = data[:, 1:]

# Set up plot
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()

# Plot each loss value
for i in range(loss_values.shape[1]):
    ax.plot(epochs, loss_values[:, i], label=f'Loss {i+1}')

# Set plot title and labels
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set x and y limits
plt.xlim(0, 200)
plt.ylim(0, 1)

# Add legend
plt.legend()

# Show plot
plt.show()
