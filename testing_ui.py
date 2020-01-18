import joblib
from keras.preprocessing.image import load_img
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from glob import glob
from tqdm import tqdm
import gc
from keras.preprocessing.image import ImageDataGenerator


FILEPATH = 'filepath/'


def load_models():
    print('Loading DL Models...')
    image_net = joblib.load('models/inception_model')
    audio_net = joblib.load('models/cnn_audio_model')
    print('...done!\n')
    return image_net, audio_net


def create_spectrogram(filename, name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    name = FILEPATH + 'audio_features/'+ name
    plt.savefig(name, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,clip,sample_rate,fig,ax,S


def preprocess_audio():
    print('Preprocessing audio files...')
    Data_dir = np.array(glob(FILEPATH + 'audio/*'))
    i = 0
    batch = 1
    batch_size = 2000
    limit = Data_dir.shape[0]
    while (i < limit):
        n = 0
        for file in tqdm(Data_dir[i:i + batch_size], desc='Batch ' + str(batch)):
            filename = file.split('\\')[1].split('.')[0] + '.jpg'
            create_spectrogram(file, filename)
            n += 1
        gc.collect()
        i += n
        batch += 1
    print('...done!\n')


def predicting_targets(image_net, audio_net):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_img = datagen.flow_from_directory(FILEPATH + 'images/', class_mode='categorical', batch_size=64, target_size=(299, 299),
                                           seed=10)
    test_audio = datagen.flow_from_directory(FILEPATH + 'audio_features/', class_mode='categorical', batch_size=64, target_size=(64, 64),
                                             seed=10)
    print('Finding pigs...')
    y_pred_img = image_net.predict_generator(test_img, steps=len(test_img), verbose=0)
    y_pred_audio = audio_net.predict_generator(test_audio, steps=len(test_audio), verbose=0)

    y_img = []
    y_audio = []

    for elem in y_pred_img:
        if elem[0] > elem[1]:
            y_img.append(0)
        else:
            y_img.append(1)
    for elem in y_pred_audio:
        if elem[0] > elem[1]:
            y_audio.append(0)
        else:
            y_audio.append(1)

    print('...done!')
    return y_img, y_audio


if __name__ == '__main__':
    image_net, audio_net = load_models()
    preprocess_audio()
    y_img, y_audio = predicting_targets(image_net, audio_net)
