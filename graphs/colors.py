#!/usr/bin/env python3

colors = {

		'cls'        : 'a12864',
		'angle'      : '0da3ee',
		'n_petals'   : '229487',
		'a'          : 'ca8b21',
		'b'          : '5387dd',
		'x'          : 'da4c4c',
		'y'          : '0ecf22',
		'title'      : '363a3d',
		'grid'       : 'f2f2f4',
		'labels'     : '6b6b76',
		'axis'       : 'e6e6e9',
		'ticks'      : 'e6e6e9',
		'background' : 'ffffff',
}

def hex_to_rgba2(hex_string):							# Model B: claude-1
    hex_string = hex_string.strip('#')
    r, g, b = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
    a = 255
    return r, g, b, a

def hex_to_rgba(hex_str):							# Model B: claude-2.1
    """Convert hex string to RGBA tuple for matplotlib"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,)		# ok, this one works if you multiply each element of the tuple by 256... but it sucks
										# EDIT: `ValueError: RGBA values should be within 0-1 range`... Ha! Claude-2.1 wins again!

def test_hex_to_rgba2():
	#r, g, b, a = hex_to_rgba2(colors['cls'])				# 10561636 == 161, 40, 100
	r, g, b, a = hex_to_rgba2(colors['angle'])				# 893934 == 13, 163, 238
	print(f'{r}, {g}, {b}, {a}')						# 'aabbcc' == (0.6666.., 0.733, 0.8, 1)

def test_hex_to_rgba():
	hex_color = colors['angle']
	rgba = hex_to_rgba(hex_color) 
	print(f'{rgba = } - {rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]}')	# 'aabbcc' == (0.6666.., 0.733, 0.8, 1)

def select_color(param, selector, debug=False):
	if debug:
		print(f'Received param {param}, selector {selector}')
	rgba = hex_to_rgba(colors[param])
	rgba_darker = tuple([c * 0.5 if c < 1. else c for c in rgba])
	rgba_select = [rgba if selector == 1 else rgba_darker][0]
	if debug:
		print(f'{rgba = }')
		print(f'{rgba_darker = }')
		print(f'{rgba_select = }')
	return rgba_select

def apply_wandb_graph_style(ax, plt, loc='upper right'):
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
	
	# Add legend
	leg = plt.legend(labelcolor='linecolor', facecolor=hex_to_rgba(colors['background']), prop={'size': 32}, loc=loc, ncol=2)

if __name__ == '__main__':
	test_hex_to_rgba2()
	test_hex_to_rgba()
