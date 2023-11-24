#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your data file path here
file_path = 'cls-train-loss.txt'

# Read the data file and set the header as the index
data = pd.read_csv(file_path, sep='\s+', index_col=0)

# Define your preferred title, x and y limits
title = "Cls Training Loss"
xlabel = "Epoch"
ylabel = "Loss"
xlim = (0, 200)
ylim = (0, 1.0)

print(data)

# Plot the data using seaborn lineplot
sns.lineplot(data=data, x=xlabel, y=data.columns[1:], legend=False)

# Configure the plot's style
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xticks(rotation=45)
plt.show()
