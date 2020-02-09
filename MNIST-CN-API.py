# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:28:29 2020

@author: EBRAE
"""


from keras.datasets import mnist
from plot_history import plot_history
# Defining a function to plot ACC and LOSS =====================



##==============================================

# Get dataset

(train_images, train_labels),(test_images, test_labels)=mnist.load_data()
print(" train_images dimention : ", train_images.ndim)
print("train_images shape : " , train_images.shape)
print("train_images datatype", train_images.dtype)

## Preparing Data ========================================

X_train= train_images.reshape(60000,28,28,1)
X_test= test_images.reshape(10000, 28,28,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test/=255

 from keras.utils import np_utils
Y_train=np_utils.to_categorical(train_labels)
Y_test=np_utils.to_categorical(test_labels)

##Making FC Sequential Model ================================================

 from keras.models import Model
 from keras import layers
 #from keras.layers import Conv2D, MaxPool2D, Input, Flatten, Dense
 #from keras.optimizers import SGD
 import keras
  
 
 myInput=layers.Input(shape=(28,28,1))
 x=layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(myInput)
 #pool1=layers.MaxPool2D(pool_size=2) (x)
 x=layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x) #(x)
 #pool2=layers.MaxPool2D(pool_size=2) (x)
 x=layers.Flatten() (x)   #(x)
 out_layer=layers.Dense(10, activation='softmax')(x)
 myModel=Model(myInput, out_layer)
 
 
 


myModel.summary()
myModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#myModel.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

## Training the Model =======================================================
networh_history= myModel.fit(X_train, Y_train, batch_size=128, epochs=5, validation_split=0.2)
plot_history(networh_history)

## Evaluation ====================================
test_loss, test_acc = myModel.evaluate(X_test, Y_test)

print(test_loss)
print(test_acc)


test_label_pred=myModel.predict(X_test)
import numpy as np
test_label_pred=np.argmax(test_label_pred, axis=1)



















