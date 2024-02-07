#!/usr/bin/env python3

import pickle
import lzma

import pandas as pd

pd.options.display.float_format = "{:,.20f}".format

with lzma.open('train.xz', 'rb') as fhandle:
	data = pickle.load(fhandle)

for idx,dat in enumerate(data):
	found = False
	gt_val = dat['gt'].round(6)				# with .round(4) there are no duplicates in this limited dataset (1k) samples
	#print(idx, gt_val)
	#for col in ['type', 'popx', 'popy', 'popz', 'nx', 'ny', 'nz', 'rot']:
	for col in ['popx', 'popy', 'popz']:
		uniq = gt_val[col].unique()
		#print(col, gt_val[col])
		print(f'{idx} {col} (unique): {len(uniq)} - {uniq}')
		if len(uniq) > 1:
			print(f'---------------------------------')
			print(f'---------------------------------')
			print(f'---------------------------------')
			print(f'{col} (diff): {uniq[1] - uniq[0]}')
			print(f'---------------------------------')
			print(f'---------------------------------')
			print(f'---------------------------------')
			found = True
	if found:
		#break
		pass
