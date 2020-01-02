import joblib
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt

datagen = ImageDataGenerator(rescale=1.0/255.0)
test_it = datagen.flow_from_directory('dataset/movie3/small_dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200), seed=10)

y = []
i = 0
while i < len(test_it):
    i += 1
    data = next(test_it)
    for elem in data[1]:
        y.append(elem)

y = np.array(y)

model = joblib.load('cnn_model')
y_pred = model.predict_generator(test_it, steps=len(test_it), verbose=0)

y_hat = []
for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_hat.append([int(y[i]), 1])
    else:
        y_hat.append([int(y[i]), 0])

print(y_hat)

fpr, tpr, thresholds = roc_curve(y, y_pred)
auc_score = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='CNN (area = {:.3f})'.format(auc_score))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


