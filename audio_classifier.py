# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from tools import audio_data_generator
import joblib


# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv1D(1, 5, activation='tanh', input_shape=(8200,1)))
	model.add(MaxPooling1D(2))
	model.add(Conv1D(1, 5, activation='tanh'))
	model.add(MaxPooling1D(2))
	model.add(Conv1D(1, 5, activation='tanh'))
	model.add(MaxPooling1D(2))
	model.add(Flatten())
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(200, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.001)
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
	# create data generator and prepare iterators
	train_it = audio_data_generator('movie1', 'train')
	test_it = audio_data_generator('movie1', 'test')
	print('Hello')
	train, test = next(train_it)
	print(train.shape)
	print(len(list(test_it)))
	print('train')
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(list(train_it)),
		validation_data=test_it, validation_steps=len(list(test_it)), epochs=1, verbose=0)

	joblib.dump(model, 'cnn_audio_model')

	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(list(test_it)), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()