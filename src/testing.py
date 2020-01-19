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

test_it_cnn = datagen.flow_from_directory('dataset/whole_dataset/test/', class_mode='binary', batch_size=64, target_size=(200, 200), seed=10)
test_it_vgg = datagen.flow_from_directory('dataset/whole_dataset/test/', class_mode='categorical', batch_size=64, target_size=(224, 224), seed=10)
test_it_incep = datagen.flow_from_directory('dataset/whole_dataset/test/', class_mode='categorical', batch_size=64, target_size=(299, 299), seed=10)

y = []
i = 0
while i < len(test_it_cnn):
    i += 1
    data = next(test_it_cnn)
    for elem in data[1]:
        y.append(elem)

print(y[:300])

print("Loading models...")
#cnn = joblib.load('models/cnn_model')
#print("CNN done")
#vgg = joblib.load('models/vgg16_model')
#print("VGG done")
incep = joblib.load('inception_model_all')
print("Inception done")

print("Predicting targets...")
#y_pred_cnn = cnn.predict_generator(test_it_cnn, steps=len(test_it_cnn), verbose=0)
#print("CNN done")
#y_pred_vgg = vgg.predict_generator(test_it_vgg, steps=len(test_it_vgg), verbose=0)
#print("VGG done")
y_pred_incep = incep.predict_generator(test_it_incep, steps=len(test_it_vgg), verbose=0)
print("Inception done")

#help = []
#for elem in y_pred_vgg:
#    help.append(elem[1])

#y_pred_vgg = np.array(help)
#print(y_pred_vgg)

y_hat = []
for i in range(len(y_pred_incep)):
    if y_pred_incep[i,0] > y_pred_incep[i,1]:
        y_hat.append([int(y[i]), 0])
    else:
        y_hat.append([int(y[i]), 1])

y_hat = np.array(y_hat)
print(y_hat[:300])

help = []
for elem in y_pred_incep:
    help.append(elem[1])

y_pred_incep = np.array(help)
print(y_pred_incep[:300])

#fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y, y_pred_cnn)
#auc_score_cnn = auc(fpr_cnn, tpr_cnn)

#fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(y, y_pred_vgg)
#auc_score_vgg = auc(fpr_vgg, tpr_vgg)

fpr_incep, tpr_incep, thresholds_incep = roc_curve(y, y_pred_incep)
auc_score_incep = auc(fpr_incep, tpr_incep)

fpr_incep_hat, tpr_incep_hat, thresholds_incep_hat = roc_curve(y, y_hat[:,1])
auc_score_incep_hat = auc(fpr_incep_hat, tpr_incep_hat)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_cnn, tpr_cnn, label='Custom CNN (area = {:.3f})'.format(auc_score_cnn))
#plt.plot(fpr_vgg, tpr_vgg, label='VGG16 (area = {:.3f})'.format(auc_score_vgg))
plt.plot(fpr_incep, tpr_incep, label='Inception ResNet V2 (area = {:.3f})'.format(auc_score_incep))
plt.plot(fpr_incep_hat, tpr_incep_hat, label='Inception ResNet V2 Pred (area = {:.3f})'.format(auc_score_incep_hat))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


