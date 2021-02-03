# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:30:50 2021

@author: Nouman ahmad
"""

from keras.preprocessing import image
import numpy as np
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential


def Feature_Extraction(img):
    
    # Block 1
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    
    # Block 2
    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Block 3
    model.add(Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Block 4
    model.add(Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    # set of FC 
    model.add(Flatten())
    model.add(Dense(4096))
    
    #getting the summary of the model (architecture)
    model.summary()
    
    img_data = np.expand_dims(img, axis=0)
    feature = model.predict(img_data)
    return feature