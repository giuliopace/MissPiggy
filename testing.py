import joblib
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0/255.0)
test_it = datagen.flow_from_directory('dataset/movie3/small_dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200))

model = joblib.load('cnn_model')

# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))