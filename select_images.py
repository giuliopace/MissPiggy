import numpy as np
import os

movie = "movie2"

dir = ['dataset/movie1/images',
			]	
for directory in dir:
	print(directory)
	for filename in os.listdir(directory):
			if filename.endswith(".jpg"): 
				#print(os.path.join(directory, filename))
				number = filename.split('-')
				number = number[1].split('.')
				number = number[0]
				if (int(number) % 5) != 0:
					path = os.path.join(directory, filename)
					os.remove(path)
			else:
				continue
