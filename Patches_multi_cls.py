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

# #add link of each datsets clasess for 40X, 100X, 200X and 400x at 40x,100x,200x and 400x
#Each class individualy 

BA_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100//B_A',224))
BF_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/B_F',224))
BPT_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/B_PT',224))
BTA_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100//B_TA',224))
MDC_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/M_DC',224))
MLC_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/M_LC',224))
MMC_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/M_MC',224))
MPC_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100/M_PC',224))




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


BA_train = discriminative_patches(BA_train)
BF_train = discriminative_patches(BF_train)
BPT_train = discriminative_patches(BPT_train)
BTA_train = discriminative_patches(BTA_train)
MDC_train = discriminative_patches(MDC_train)
MLC_train = discriminative_patches(MLC_train)
MMC_train = discriminative_patches(MMC_train)
MPC_train = discriminative_patches(MPC_train)




# BreakHis 	Patches Classfiaction for Multi class
# Create labels for each class

BA_train_label = np.zeros(len(BA_train))
BF_train_label = np.ones(len(BF_train))
BPT_train_label = np.full(len(BPT_train),2)
BTA_train_label = np.full(len(BTA_train),3)
MDC_train_label = np.full(len(MDC_train),4)
MLC_train_label = np.full(len(MLC_train),5)
MMC_train_label = np.full(len(MMC_train),6)
MPC_train_label = np.full(len(MPC_train),7)

# Merge data 
X_train = np.concatenate((BA_train, BF_train,BPT_train,BTA_train,MDC_train,MLC_train,MMC_train,MPC_train), axis = 0)
Y_train = np.concatenate((BA_train_label, BF_train_label,BPT_train_label,BTA_train_label,MDC_train_label,MLC_train_label,MMC_train_label,MPC_train_label), axis = 0)

# Shuffle train data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# To categorical
Y_train = to_categorical(Y_train, num_classes= 8)

x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, 
    test_size=0.25, 
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
        ax.title.set_text('BA')
    if np.argmax(Y_train[i]) == 1:
        ax.title.set_text('BF')
    if np.argmax(Y_train[i]) == 2:
        ax.title.set_text('BPT')
    if np.argmax(Y_train[i]) == 3:
        ax.title.set_text('BTA')
    if np.argmax(Y_train[i]) == 4:
        ax.title.set_text('MDC')
    if np.argmax(Y_train[i]) == 5:
      ax.title.set_text('MLC')
    if np.argmax(Y_train[i]) ==6:
        ax.title.set_text('MMC')
    if np.argmax(Y_train[i]) ==7:
        ax.title.set_text('MPC')         
  
    plt.imshow(x_train[i], interpolation='nearest')
plt.show()

BATCH_SIZE = 10
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
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
		model.add(layers.Dropout(0.5))
		model.add(layers.BatchNormalization())
		model.add(layers.Dense(8, activation='softmax'))
		
		model.compile(
			loss='categorical_crossentropy',
			optimizer=Adam(lr=lr),
			metrics=['accuracy']
		)
		return model

	K.clear_session()
	gc.collect()


	# EfficeintNet model B5

	efficeient = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(224,224,3))
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
		epochs=40,
		validation_data=(x_test, y_test),
		callbacks=[learn_control, checkpoint]
	)

with open('history.json', 'w') as f:
    json.dump(str(history.history), f)

history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

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

recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')

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

cm_plot_label =['BA', 'BF','BPT','BTA','MDC','MLC','MMC','MPC']
plot_confusion_matrix(cm, cm_plot_label)

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

cm_plot_label =['A', 'F','PT','TA','DC','LC','MC','PC']
plot_confusion_matrix(cm, cm_plot_label, title ='')

from sklearn.metrics import classification_report
classification_report( np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

"""#ROC CURVE"""

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
n_classes=8
# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

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
        return "BA"
    if namecode==1:
        return "BF"
    if namecode==2:
        return "BPT"
    if namecode==3:
        return "BTA"
    if namecode==4:
        return "MDC"
    if namecode==5:
        return "MLC"
    if namecode==6:
        return "MMC"
    if namecode==7:
        return "MPC"                           
    
for i in range(len(prop_class)):
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title("Predicted result:"+ Transfername(np.argmax(y_pred[prop_class[i]]))
                       +"\n"+"Actual result: "+ Transfername(np.argmax(y_test[prop_class[i]])))
    plt.imshow(x_test[prop_class[i]], interpolation='nearest')
plt.show()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

macro_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

