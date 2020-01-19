import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from glob import glob
from tqdm import tqdm

def create_spectrogram(filename, name, mode):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    name = 'dataset/audio_features/' + mode + '/' + name
    plt.savefig(name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,clip,sample_rate,fig,ax,S




modes = ['train/pigs', 'train/no_pigs', 'test/pigs', 'test/no_pigs']
#modes = ['train/pigs']

for mode in modes:
    print(mode)
    Data_dir = np.array(glob('dataset/audio/' + mode + '/*'))
    print(Data_dir.shape)
    i = 0
    batch_size = 2000
    limit = Data_dir.shape[0]
    while (i < limit):
        n = 0
        for file in tqdm(Data_dir[i:i+batch_size]):
            filename = file.split('\\')[1].split('.')[0] + '.jpg'
            create_spectrogram(file,filename,mode)
            n += 1
        gc.collect()
        i += n


"""
i=0
for file in Data_dir[i:i+2000]:
    #Define the filename as is, "name" refers to the JPG, and is split off into the number itself.
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,'train/pigs')
gc.collect()

i=2000
for file in Data_dir[i:i+2000]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()
i=4000
for file in Data_dir[i:]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()
"""

# y, sr = librosa.load(filename)
# trim silent edges
# whale_song, _ = librosa.effects.trim(y)
# librosa.display.waveplot(whale_song, sr=sr)

# plt.figure()
# librosa.display.waveplot(y, sr=sr)
# plt.title('Monophonic')
# plt.show()
# plt.close()
