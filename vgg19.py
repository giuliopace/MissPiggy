# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
import joblib
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping


# define cnn model
def define_model():
	model = VGG16(weights=None, classes=2)
	"""
	model = Sequential()
	model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	# fully connected layers
	model.add(Flatten())
	model.add(Dense(units=4096, activation="relu"))
	model.add(Dense(units=4096, activation="relu"))
	model.add(Dense(units=1, activation="sigmoid"))
	"""
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	# opt = Adam(lr=0.001)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['acc'], color='blue', label='train')
	pyplot.plot(history.history['val_acc'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset/movie3/small_dataset/train/',
		class_mode='categorical', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('dataset/movie3/small_dataset/test/',
		class_mode='categorical', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=5, verbose=0)

	joblib.dump(model, 'vgg16_model_all')

	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()