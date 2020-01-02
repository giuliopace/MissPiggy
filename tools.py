import os
import random
import librosa
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class ROCCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def audio_data_generator(movie, mode, batchsize=64):
    # train set
    pigs = os.listdir('dataset/' + movie + '/audio/' + mode + '/pigs/')
    no_pigs = os.listdir('dataset/' + movie + '/audio/' + mode + '/no_pigs/')

    filenames = []
    for p in pigs:
        filenames.append('dataset/' + movie + '/audio/' + mode + '/pigs/' + p)
    for p in no_pigs:
        filenames.append('dataset/' + movie + '/audio/' + mode + '/no_pigs/' + p)

    random.seed(42)
    random.shuffle(filenames)

    i = 0
    for file in filenames:
        if i == 0:
            data = []
            targets = []

        y, sr = librosa.load(file)
        d = librosa.stft(y)
        d_harmonic, _ = librosa.decompose.hpss(d)
        d_harmonic = d_harmonic[:,:8]
        data.append(d_harmonic)

        if file.split('/')[4] == 'pigs':
            targets.append(1)
        else:
            targets.append(0)

        i += 1
        if i % batchsize == 0:
            i = 0
            yield (data, targets)

'''
datagen = audio_data_generator('movie1', 'train', 5)
x, y = next(datagen)
print(len(x), len(y))
x, y = next(datagen)
print(len(x), len(y))

datagen = audio_data_generator('movie1', 'test', 5)
x, y = next(datagen)
print(len(x), len(y))
x, y = next(datagen)
print(len(x), len(y))
print(x[0])
'''
