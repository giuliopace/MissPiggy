import joblib
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt


datagen = ImageDataGenerator(rescale=1.0/255.0)
#test_it_cnn = datagen.flow_from_directory('dataset/movie3/small_dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200), seed=10)
#test_it_vgg = datagen.flow_from_directory('dataset/movie3/small_dataset/test/', class_mode='categorical', batch_size=64, target_size=(224, 224), seed=10)
#test_it_incep = datagen.flow_from_directory('dataset/movie3/small_dataset/test/', class_mode='categorical', batch_size=64, target_size=(299, 299), seed=10)
test_it_binary = datagen.flow_from_directory('dataset/audio_features/test/', class_mode='binary', batch_size=64, target_size=(64, 64), seed=10)
test_it = datagen.flow_from_directory('dataset/audio_features/test/', class_mode='categorical', batch_size=64, target_size=(64, 64), seed=10)

y = []
i = 0
while i < len(test_it_binary):
    i += 1
    data = next(test_it_binary)
    for elem in data[1]:
        y.append(elem)

print(y[:300])

print("Loading models...")
#cnn = joblib.load('models/cnn_model')
#print("CNN done")
#vgg = joblib.load('models/vgg16_model')
#print("VGG done")
cnn_audio = joblib.load('cnn_audio_model')
print("CNN done")

print("Predicting targets...")
#y_pred_cnn = cnn.predict_generator(test_it_cnn, steps=len(test_it_cnn), verbose=0)
#print("CNN done")
#y_pred_vgg = vgg.predict_generator(test_it_vgg, steps=len(test_it_vgg), verbose=0)
#print("VGG done")
y_pred = cnn_audio.predict_generator(test_it, steps=len(test_it), verbose=0)
print("CNN done")

#help = []
#for elem in y_pred_vgg:
#    help.append(elem[1])

#y_pred_vgg = np.array(help)
#print(y_pred_vgg)

y_hat = []
for i in range(len(y_pred)):
    if y_pred[i,0] > y_pred[i,1]:
        y_hat.append([int(y[i]), 0])
    else:
        print(y_pred[i,0] + ' ::: ' + y_pred[i,1])
        y_hat.append([int(y[i]), 1])

y_hat = np.array(y_hat)
print(y_hat[:300])

help = []
for elem in y_pred:
    help.append(elem[1])

y_pred = np.array(help)
print(y_pred[:300])

#fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y, y_pred_cnn)
#auc_score_cnn = auc(fpr_cnn, tpr_cnn)

#fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(y, y_pred_vgg)
#auc_score_vgg = auc(fpr_vgg, tpr_vgg)

fpr, tpr, thresholds = roc_curve(y, y_pred)
auc_score = auc(fpr, tpr)

fpr_hat, tpr_hat, thresholds_hat = roc_curve(y, y_hat[:,1])
auc_score_hat = auc(fpr_hat, tpr_hat)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_cnn, tpr_cnn, label='Custom CNN (area = {:.3f})'.format(auc_score_cnn))
#plt.plot(fpr_vgg, tpr_vgg, label='VGG16 (area = {:.3f})'.format(auc_score_vgg))
plt.plot(fpr, tpr, label='Inception ResNet V2 (area = {:.3f})'.format(auc_score))
plt.plot(fpr_hat, tpr_hat, label='Inception ResNet V2 Pred (area = {:.3f})'.format(auc_score_hat))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


