# -*- coding: utf-8 -*-
"""SVM B5 8-40-EfficeintNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oF8SHU3CW21Zn9-TTgL5G41LT7aLrKMS
"""

from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/My Drive/Using Keras Pre-trained Deep Learning models/mulcls-8/8-Efficientnet/B5


#import  efficeient library 
pip install -U efficientnet

# Commented out IPython magic to ensure Python compatibility.
import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
import efficientnet.keras as efn 
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
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



# %matplotlib inline

#Transfer 'png' images to an array IMG
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
           
            img = cv2.resize(img, (RESIZE,RESIZE))
           
            IMG.append(np.array(img))
    return IMG
	
	
#add link of each datsets clasess for 40X, 100X, 200X and 400x magnification factor 
	
benign_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100X/benign',224))
malign_train = np.array(Dataset_loader('/content/drive/My Drive/BreaKHis_v1/100X/malignant',224))

# BreakHis Cancer: Malignant vs. Benign
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
    test_size=0.30, 
    random_state=11
)


# Histopathology slide Color Preprocessing 

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

n_split=5

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

	#Pre traind EfficientNetB5 model 
	K.clear_session()
	gc.collect()
	efficeient =  efn.EfficientNetB5(weights='imagenet',include_top=True,input_shape=(224,224,3))
	#efficeient.summary()
	model = build_model(efficeient ,lr = 1e-4)

	from keras.models import Model
	model_feat = Model(input=model.input, output=model.get_layer('dense_1').output)

	feat_train = model_feat.predict(x_train)
	print(feat_train.shape)
	feat_test = model_feat.predict(x_test)
	print(feat_test.shape)

	y_train.shape


	#SVM model for training  
	from sklearn.svm import SVC

	svm = SVC(kernel='rbf')
	svm.fit(feat_train,y_train)

	print('fitting done !!!')

	svm.score(feat_train,y_train)
	svm.score(feat_test,y_test)
	y_pred = svm.predict(feat_test)


