import joblib
from keras.preprocessing.image import load_img as load_img
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from glob import glob
from tqdm import tqdm
import gc
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tensorflow.python.util import deprecation
import os
import logging


FILEPATH = 'out/'



def load_models():
	print('Loading DL Models...')
	print("be patient, this might take a couple of minutes")
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
	name = FILEPATH + 'audio_features/audio_features/'+ name
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
			filename = file.split('/')[2].split('.')[0] + '.png'
			try:
				create_spectrogram(file, filename)
			except Exception:
				pass 
			n += 1
		gc.collect()
		i += n
		batch += 1
	print('...done!\n')


def predicting_targets(image_net, audio_net):
	y_img = []
	y_audio = []

	datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	file_path_image = FILEPATH + 'images/'
	file_path_audio = FILEPATH + 'audio_features/'
	test_img = datagen.flow_from_directory(file_path_image, class_mode='categorical', batch_size=64, target_size=(299, 299),
										   shuffle=False)
	test_audio = datagen.flow_from_directory(file_path_audio, class_mode='categorical', batch_size=64, target_size=(64, 64),
										   shuffle=False)
	print('Finding pigs')
	y_pred_img = image_net.predict_generator(test_img, steps=len(test_img), verbose=0)
	y_pred_audio = audio_net.predict_generator(test_audio, steps=len(test_audio), verbose=0)


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
	
#	dir_images = np.array(glob(FILEPATH + 'images/*'))
#	dir_audio = np.array(glob(FILEPATH + 'audio_features/*'))
#
#	y_img = []
#	y_audio = []
#
#	for file in tqdm(dir_images, desc='Images'):
#		y_pred = image_net.predict(np.array(load_img(file, target_size=(299, 299))).reshape(1, 299, 299, 3))
#		if y_pred[0][0] > y_pred[0][1]:
#			y_img.append(0)
#		else:
#			y_img.append(1)
#
#	for file in tqdm(dir_audio, desc='Audio'):
#		y_pred = audio_net.predict(np.array(load_img(file, target_size=(64, 64))).reshape(1, 64, 64, 3))
#		if y_pred[0][0] > y_pred[0][1]:
#			y_audio.append(0)
#		else:
#			y_audio.append(1)

	print('...done!')
	return y_img, y_audio

def watermark(input_image_path,
			   output_image_path,
			   watermark_image_path,
			   text, pos, color, size):
	photo = Image.open(input_image_path)
	watermark = Image.open(watermark_image_path)
	watermark = watermark.resize(size)

	# make the image editable
	drawing = ImageDraw.Draw(photo)
	photo.paste(watermark, pos)
	font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
	drawing.text(pos, text, fill=color, font=font)
	photo.save(output_image_path)

def transform_to_timestamp(i):
	tot = i * 0.2
	s = tot % 60
	m = tot / 60
	return str(m) + ":" + str(s)

def output_manager(y_img, y_audio):
	found_some_pics = False
	for i in range(len(y_img)):
		if i+1<10:
			img_name = "image-00000" + str(i+1) + ".jpg"
		elif i+1<100:
			img_name = "image-0000" + str(i+1) + ".jpg"
		elif i+1<1000:
			img_name = "image-000" + str(i+1) + ".jpg"
		elif i+1<10000:
			img_name = "image-00" + str(i+1) + ".jpg"
		elif i+1<100000:
			img_name = "image-0" + str(i+1) + ".jpg"
		else:
			img_name = "image-" + str(i+1) + ".jpg"
		
		if y_img[i]==1:
			found_some_pics = True
			watermark("out/images/images/" + img_name, "out/labelled_images/" + img_name, "watermark/background.jpg", text="Pig detected", pos=(0,0), color=(23, 155, 115), size=(300,50))
		else:
			watermark("out/images/images/" + img_name, "out/labelled_images/" + img_name, "watermark/background.jpg", text="no pigs", pos=(0,0), color=(192, 192, 192), size=(200,50))
		
	if found_some_pics==False:
		print("no pigs found in images")
	else:
		print("labelled images are now available in the labelled_images folder")
		
	#timestamps for audio
	found_some_audio = False
	curr_start = -1
	streak = False
	for i in range(len(y_audio)):
		if y_audio[i]==1:
			found_some_audio = True
			if streak == False:
				streak = True
				curr_start = i
		else:
			if streak == True:
				print("pig audio found from", transform_to_timestamp(curr_start), " to ", transform_to_timestamp(i-1))
				streak = False
				curr_start = -1

	if curr_start != -1:
		print("pig audio found from", transform_to_timestamp(curr_start), " to ", transform_to_timestamp(len(y_audio)))
	if found_some_audio == False:
		print("no pigs found in audio")
	#print(y_img)
	#print(y_audio)



if __name__ == '__main__':
	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	image_net, audio_net = load_models()
	preprocess_audio()
	y_img, y_audio = predicting_targets(image_net, audio_net)
	output_manager(y_img, y_audio)
	