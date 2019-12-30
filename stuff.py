from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

y, sr = librosa.load('dataset/movie3/movie33831.mp3')

D = librosa.stft(y)

D_harmonic, D_percussive = librosa.decompose.hpss(D)

# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))
print(librosa.amplitude_to_db(D, ref=rp).shape)

plt.figure(figsize=(12, 8))

plt.subplot()
librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')
plt.show()
plt.close()

plt.subplot()
librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Harmonic spectrogram')
plt.show()
plt.close()

plt.subplot()
librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')
plt.tight_layout()
plt.show()

plt.close()
