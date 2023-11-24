#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Load data from text file
data = np.loadtxt('cls-train-loss.txt', delimiter='\t', skiprows=1) 

# Extract column names from header
column_names = data[0,:]
epoch_col = column_names[0]

print(f'Column names: {column_names}')
print(f'Epoch column: {epoch_col}')

# Extract data (remove header row)
data = data[1:,:] 

# Set plot title, xlimits and ylimits
plt.title('Training and Validation Losses')
plt.xlim(0, 200) 
plt.ylim(0, 1)

# Plot each loss metric as a separate line
for i in range(1,data.shape[1]):
  plt.plot(data[:,0], data[:,i], label=column_names[i])

# Add gridlines and legend
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left') 

# Save figure 
plt.savefig('losses.png', bbox_inches='tight')
