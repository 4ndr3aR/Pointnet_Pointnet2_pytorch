#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the styling
sns.set()
sns.set_style("whitegrid")

# Replace this with the path to your .txt file
file_path = "cls-train-loss.txt"

# Read the data into a pandas DataFrame
df = pd.read_csv(file_path, sep="\s+", header=0, index_col=0, usecols=[0] + list(range(1, len(df.columns))))

# Rename columns according to the header
# Replace this line with the actual header names in your file
header_dict = {"epoch": "Epoch", "cls training loss": "Classification Training Loss", "cls validation loss": "Classification Validation Loss", "angle training loss": "Angle Training Loss", "angle validation loss": "Angle Validation Loss"}
df.rename(columns=header_dict, inplace=True)

# Plot the data
sns.lineplot(data=df, x="Epoch", y=list(df.columns[1:]), legend=False)

# Set plot title, x and y limits, and save
plt.title("Title of your plot")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(left=0, right=200)
plt.ylim(bottom=0, top=1)
plt.tight_layout()
plt.savefig("output.png", dpi=300)
