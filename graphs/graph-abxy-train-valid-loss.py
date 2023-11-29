#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from colors import colors, hex_to_rgba, select_color

'''
if len(sys.argv) < 2:
	print(f'\n\tUsage: {sys.argv[0]} <stem>\n')
	sys.exit()

stem = Path(sys.argv[1]).stem
param = stem.split('-')[0]
'''

limits = {
		'a': (0, 0.050),
		'b': (0, 0.015),
		'x': (0, 0.010),
		'y': (0, 0.006),
}

'''
print(f'Processing {stem}.txt for parameter {param}')

#stem = 'n_petals-train-valid-loss'

# Read data from text file
data = np.genfromtxt(stem + '.txt', skip_header=1)

# Extract epoch values from the first column
epochs = data[:, 0]

# Extract loss values from the second to nth columns
losses = data[:, 1:]
'''

# Set fivethirtyeight style
plt.style.use('fivethirtyeight')

# Create figure and axis objects
fig, ax = plt.subplots()

labels = ['Training Loss', 'Validation Loss']

# Plot each loss column
'''
for i in range(losses.shape[1]):
    ax.plot(epochs, losses[:, i], label=labels[i], color=select_color(param, i))
'''

fg_color = 'red'
bg_color = 'blue'

ax.tick_params(color=fg_color, labelcolor=hex_to_rgba(colors['ticks']))
ax.tick_params(axis='x', which='major', bottom=True)


fn_suffix = '-train-valid-loss.txt'
paramlist = ['a', 'b', 'x', 'y']
#paramlist = ['a', 'b']
#paramlist = ['x', 'y']
max_val   = -1

color_list = []

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

		color_list.append(color)



#kfunc = lambda x, pos: f'{x} K - {pos}'
'''
kfunc = lambda x, _: f'{x} K'
kformatter = matplotlib.ticker.FuncFormatter(kfunc)
ax.xaxis.set_major_formatter(kformatter)
'''

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.tick_params(axis='x', colors=hex_to_rgba(colors['grid']), size=15, width=2, pad=15, direction='inout', labelsize=32)		# size = tick length
ax.tick_params(axis='y', colors=hex_to_rgba(colors['grid']), size=0 , width=2, pad=15, direction='inout', labelsize=32)

for ticklabel in plt.gca().get_xticklabels():
    ticklabel.set_color(hex_to_rgba(colors['labels']))
for ticklabel in plt.gca().get_yticklabels():
    ticklabel.set_color(hex_to_rgba(colors['labels']))

#ticks_loc = ax.get_xticks().tolist()
#ax.set_xticklabels([x for x in ticks_loc])

#ax.set_xticklabels(hex_to_rgba(colors['labels']), rotation=270)

'''
# Currently, there are no minor ticks,
#   so trying to make them visible would have no effect
ax.xaxis.get_ticklocs(minor=True)     # []
ax.yaxis.get_ticklocs(minor=True)     # []

# Initialize minor ticks
ax.minorticks_on()

# Now minor ticks exist and are turned on for both axes

# Turn off x-axis minor ticks
#ax.xaxis.set_tick_params(which='minor', bottom=False)
ax.xaxis.set_tick_params(which='both', bottom=False)
#ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.yaxis.set_tick_params(which='both', bottom=False)
'''




plt.grid(axis='x', which='major', color=hex_to_rgba(colors['background']), linestyle=':', linewidth=0.5)
plt.grid(axis='y', which='major', color=hex_to_rgba(colors['grid']), linewidth=1.5)

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
#plt.ylim(limits[param][0], limits[param][1])  # Update with your desired y-axis limits
#plt.ylim(0., 0.05)  # Update with your desired y-axis limits
plt.ylim(0., max_val + 0.01*max_val)

# Add legend
leg = plt.legend(labelcolor='linecolor', facecolor=hex_to_rgba(colors['background']), prop={'size': 32}, loc='upper right', ncol=2)

'''
for idx, text in enumerate(leg.get_texts()):
    text.set_color(color_list[idx])
'''

fig.set_size_inches(19.2, 10.8)
fig.savefig(f'{"".join(paramlist)}-train-valid-loss.png', dpi=100, bbox_inches='tight', facecolor=hex_to_rgba(colors['background']))






'''
cb = plt.colorbar(im)



# set tick and ticklabel color
im.axes.tick_params(color=fg_color, labelcolor=fg_color)

# set imshow outline
for spine in im.axes.spines.values():
    spine.set_edgecolor(fg_color)    

# COLORBAR
# set colorbar label plus label color
cb.set_label('colorbar label', color=fg_color)

# set colorbar tick color
cb.ax.yaxis.set_tick_params(color=fg_color)

# set colorbar edgecolor 
cb.outline.set_edgecolor(fg_color)

# set colorbar ticklabels
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

fig.patch.set_facecolor(bg_color)    

'''
















#plt.tight_layout()

# Show the plot
plt.show()


