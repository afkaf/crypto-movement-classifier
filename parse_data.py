import os
import numpy as np

files = os.listdir('parse')

headers = ['unix','open','high','low','close','volume']
datasets = {}
for file in files:
	with open('parse/'+file,'r') as f:
		dataset = [line[:-1].split(',') for line in f.readlines()[2:]]

	for row in dataset:
		row.pop(1)
		row.pop(1)
		if int(row[0]) < 1*10**10:
			row[0] = f"{int(row[0])*1000}"
		for i, e in enumerate(row):
			if float(e) == 0:
				row[i] = '0.1'
	dataset = [headers]+dataset

	with open('parsed/'+file.split('_')[1]+'.csv','w+') as f:
		f.writelines([','.join(line)+'\n' for line in dataset])

