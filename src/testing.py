from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

path = 'data/movie3/'

arrs = []
for i in tqdm(range(28)):
    arr = np.load(path + 'negative_batch_' + str(i+1) + '.npy')
    arrs.append(arr)

print("done")