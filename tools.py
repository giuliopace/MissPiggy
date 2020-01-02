import os
import random
import librosa


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
