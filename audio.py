import numpy as np
import os
import shutil

movie = 'movie1'

with open('dataset/' + movie + '/audio_labelling.txt') as file:
    data = file.read()

data = data.split("\n")

timestamps = []

for stamp in data:
    stamp = stamp.split('-')
    starts = stamp[0].split(':')
    ends = stamp[1].split(':')
    start = (int(starts[2]) * 100) + (int(starts[1]) * 1000) + (int(starts[0]) * 60 * 1000)
    end = (int(ends[2]) * 100) + (int(ends[1]) * 1000) + (int(ends[0]) * 60 * 1000)
    start = start / 200
    end = end / 200
    print(starts, start, ends, end)
    timestamps.append([start, end])

print(timestamps)

filenames = []

for name in timestamps:
    for i in range(int(name[1])-int(name[0])):
        name_help = str(name[0]+i).split('.')[0]
        if len(name_help) == 4:
            name_new = movie + '_' + name_help + '.mp3'
        elif len(name_help) == 3:
            name_new = movie + '_0' + name_help + '.mp3'
        elif len(name_help) == 2:
            name_new = movie + '_00' + name_help + '.mp3'
        else:
            name_new = movie + '_000' + name_help + '.mp3'
        filenames.append(name_new)

print(len(filenames))
print(filenames)

for name in filenames:
    os.rename("dataset/" + movie + "/audio/" + name, "dataset/" + movie + "/audio/train/pigs/" + name)
