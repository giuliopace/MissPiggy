from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

path = 'data/movie3/No pigs/'

images = []
i = 0
n = 1
for filename in tqdm(os.listdir(path)):
    img = Image.open(path + filename)
    img_arr = np.array(img)
    images.append(img_arr)
    i += 1
    if i > 1000:
        images = np.asarray(images)
        np.save('data/movie3/negative_batch_' + str(n) + '.npy', images)
        n += 1
        i = 0
        images = []

images = np.asarray(images)
print(images.shape)
np.save('data/movie3/negative_batch_' + str(n) + '.npy', images)

exit()

img = Image.open("data/movie1-0000707.jpg")
img_arr = np.array(img)
print(img_arr)
print(img_arr.shape)
plt.figure()
plt.imshow(np.transpose(img_arr, (0, 1, 2)))
plt.show()
