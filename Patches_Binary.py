# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:53:02 2021

@author: Nouman ahmad
"""

# pip install -U efficientnet
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np


import json
import math
from PIL import Image

from keras import layers
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from CNN_Feature_Extraction import Feature_Extraction
from algo_clustring import discriminative_patches
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools



#Binary Patches Dis
#Transfer 'png' images to an array IMG
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    # IMGB = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE,RESIZE))            
            feature=Feature_Extraction(img)
            IMG.append(np.array(feature))
    return IMG
	#add class path for each mag factor 40X,100X,200X AND 400X
benign_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/2cls_patches/40/benign',224))
malign_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/2cls_patches/40/malignant',224))


benign_train=discriminative_patches(benign_train)
malign_train=discriminative_patches(malign_train)

##  arry=benign_train.reshape(len(benign_train),-1)
#    from sklearn.cluster import AgglomerativeClustering
#
#cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
#labels=cluster.fit_predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels,
#            s=50, cmap='viridis');

#    
#import scipy.cluster.hierarchy as shc
#plt.figure(figsize=(10, 7))
#plt.title("Customer Dendograms")
#dend = shc.dendrogram(shc.linkage(X, method='ward'))



# Breast cancer Cancer: Malignant vs. Benign
# Create labels
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))


# Merge data 
X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)


# Shuffle train data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]


# To categorical
Y_train = to_categorical(Y_train, num_classes= 2)



x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, 
    test_size=0.33, 
    random_state=11
)



def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

x_train, x_test = color_preprocessing(x_train, x_test)

# # Display first 15 images of moles, and how they are classified
w=60
h=40
fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(x_train[i], interpolation='nearest')
plt.show()

x_train.shape

BATCH_SIZE = 16


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
					 rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
train_generator = ImageDataGenerator(**data_gen_args)



# KFold Cross Validation

n_split=10

for train_index,test_index in KFold(n_split).split(x_train, y_train):
    x_train,x_test=X[train_index],X[test_index]
    y_train,y_test=Y[train_index],Y[test_index]
  
  
	def build_model(backbone, lr=1e-4):
		model = Sequential()
		#backbone.summary()
		model.add(backbone)
		model.add(layers.GlobalAveragePooling2D())
		model.add(layers.Dropout(0.2))
		model.add(layers.BatchNormalization())
		model.add(layers.Dense(2, activation='softmax'))
		
		# loss function binary_crossentropy, poisson and focal loss 
		model.compile(
			loss='binary_crossentropy',
			optimizer=Adam(lr=lr),
			metrics=['accuracy']
		)
		return model

	K.clear_session()
	gc.collect()

	# EfficeintNet model B5
	efficeient =  efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(224,224,3))
	#efficeient.summary()
	model = build_model(efficeient ,lr = 1e-4)
	model.summary()

	# Learning Rate Reducer
	learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5,
									  verbose=1,factor=0.2, min_lr=1e-7)
	# Checkpoint
	filepath="weights.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	x_train.shape[0]/BATCH_SIZE

	history = model.fit_generator(
		train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
		steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
		epochs=200,
		validation_data=(x_test, y_test),
		callbacks=[learn_control, checkpoint]
	)



#Ploting Graph and Evaluation Matrixs
with open('history.json', 'w') as f:
    json.dump(str(history.history), f)


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


"""#Prediction"""

model.load_weights("weights.hdf5")

y_pred = model.predict(x_test)

accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

from sklearn.metrics import matthews_corrcoef

matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred)

from sklearn.metrics import f1_score

f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')

f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='micro')

f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')

precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='micro')

precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')

model.evaluate(x_test,y_test)

Y_pred = model.predict(x_test)



"""#Confusion Matrix"""

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

cm_plot_label =['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label)

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

cm_plot_label =['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, title ='')

from sklearn.metrics import classification_report
classification_report( np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))



"""#ROC CURVE"""

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
roc_log = roc_auc_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
plt.close()

i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==np.argmax(y_pred[i])):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==np.argmax(y_pred[i])):
        mis_class.append(i)
    if(len(mis_class)==8):
        break

# # Display first 8 images of benign
w=60
h=40
fig=plt.figure(figsize=(18, 10))
columns = 4
rows = 2

def Transfername(namecode):
    if namecode==0:
        return "Benign"
    else:
        return "Malignant"
    
for i in range(len(prop_class)):
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title("Predicted result:"+ Transfername(np.argmax(y_pred[prop_class[i]]))
                       +"\n"+"Actual result: "+ Transfername(np.argmax(y_test[prop_class[i]])))
    plt.imshow(x_test[prop_class[i]], interpolation='nearest')
plt.show()
