#!/usr/bin/env python3

import pickle
import lzma

import pandas as pd

pd.options.display.float_format = "{:,.20f}".format

with lzma.open('train.xz', 'rb') as fhandle:
	data = pickle.load(fhandle)

uniques = []
for idx,dat in enumerate(data):
	found = False
	gt_val = dat['gt'].fillna(-1).round(6)				# with .round(4) there are no duplicates in this limited dataset (1k) samples
	#print(idx, gt_val)
	#for col in ['type', 'popx', 'popy', 'popz', 'nx', 'ny', 'nz', 'rot']:
	#for col in ['popx', 'popy', 'popz']:
	for col in ['rot']:
		uniq = gt_val[col].unique()

		for u in uniq:
			if u not in uniques:
				uniques.append(u)

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

print(f'Found {len(uniques)} unique values: {uniques}')
