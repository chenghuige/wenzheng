#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cifar10.py
#        \author   chenghuige  
#          \date   2018-07-11 15:34:51.963932
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import melt
import os
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization, advanced_activations
from keras.layers import initializers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
 
import gezi

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
classes_map = dict(zip(classes, range(len(classes))))

print('----------', classes_map)

import pandas as pd
l = pd.read_csv('./mount/data/kaggle/cifar-10/trainLabels.csv')
m = {}
for i in range(len(l)):
  m[l.id[i]] = l.label[i]

def loadData(path='train'):
  data = []
  labels = []
  #for i in range(10):
  #dir1 = './'+path+'/'+str(i)
  dir1 = './'+path+'/'
  listImg = os.listdir(dir1)
  for img in listImg:      
    #print('img-----', img)  
    imgIn = cv2.imread(dir1+'/'+img)
    if imgIn.size != 3072: print('Img error')
    data.append(imgIn)
    #data.append([numpy.array(Image.open(dir+'/'+img))])
    #labels.append(i)
    #print path, i, 'is read'
    labels.append(classes_map[m[int(gezi.strip_suffix(img, '.png'))]])
  return data, labels

def loadTrainValid(path='train'):
  train_data = []
  train_labels = []
  valid_data = []
  valid_labels = []
  #for i in range(10):
  #dir1 = './'+path+'/'+str(i)
  dir1 = './'+path+'/'
  listImg = os.listdir(dir1)
  for img in listImg:      
    #print('img-----', img)  
    imgIn = cv2.imread(dir1+'/'+img)
    id = int(gezi.strip_suffix(img, '.png'))
    if imgIn.size != 3072: print('Img error')
    if id % 10 == 0:
      data, labels = valid_data, valid_labels
    else:
      data, labels = train_data, train_labels
    data.append(imgIn)
    #data.append([numpy.array(Image.open(dir+'/'+img))])
    #labels.append(i)
    #print path, i, 'is read'
    labels.append(classes_map[m[id]])
  return train_data, train_labels, valid_data, valid_labels


def loadTest(path='test'):
  data = []
  dir1 = './'+path+'/'
  listImg = os.listdir(dir1)
  num_imgs = len(listImg)
  for i in range(num_imgs):
    imgIn = cv2.imread(dir1+'/'+'%d.png' % (i + 1))
    if imgIn.size != 3072: print('Img error')
    data.append(imgIn)
  return data
 
 
#trainData, trainLabels = loadData('train')
trainData, trainLabels, validData, validLabels = loadTrainValid()

testData = loadTest()

def preprocess(trainData, trainLabels=None):
  if trainLabels is not None:
    trainLabels = np_utils.to_categorical(trainLabels, 10)
  trainData = numpy.reshape(trainData, (len(trainData), 32, 32,3))
  trainData = trainData.astype(numpy.float32)
  trainData -= numpy.mean(trainData, axis=0)
  trainData /= numpy.std(trainData, axis=0)
  if trainLabels is not None:
    return trainData, trainLabels
  else:
    return trainData

print('---loading train')
trainData, trainLabels = preprocess(trainData, trainLabels)
print('num_train', len(trainData))
print('---loading valid')
validData, validLabels = preprocess(validData, validLabels)
print('num_valid', len(validData))
print('---loading test')
testData = preprocess(testData)
print('num_test', len(testData))
 
#print trainData[-1]
 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', input_shape=(32,32,3), data_format='channels_last', kernel_initializer=initializers.he_normal()))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-6))
model.add(Activation(advanced_activations.LeakyReLU(alpha=0.2)))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', data_format='channels_last', kernel_initializer=initializers.he_normal()))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-6))
model.add(Activation(advanced_activations.LeakyReLU(alpha=0.2)))
model.add(AveragePooling2D(pool_size=(2,2)))
 
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', data_format='channels_last', kernel_initializer=initializers.he_normal()))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-6))
model.add(Activation(advanced_activations.LeakyReLU(alpha=0.2)))
model.add(AveragePooling2D(pool_size=(2,2)))
 
model.add(Flatten())
 
model.add(Dense(1024, kernel_initializer=initializers.he_normal()))
model.add(BatchNormalization(epsilon=1e-6))
model.add(Activation(advanced_activations.LeakyReLU(alpha=0.2)))
 
model.add(Dense(256, activation=advanced_activations.LeakyReLU(alpha=0.1), kernel_initializer=initializers.he_normal()))
model.add(Dense(10, activation='softmax', kernel_initializer=initializers.he_normal()))
 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.summary()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
result = model.fit(trainData, trainLabels, 
                   batch_size=256, 
                   epochs=20, 
                   verbose=1, 
                   shuffle=True,
                   validation_data=(validData, validLabels))
 
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
 
plt.figure()
plt.plot(result.epoch, result.history['acc'], label='acc')
plt.scatter(result.epoch, result.history['acc'], marker='*')
plt.plot(result.epoch, result.history['val_acc'], label='val_acc')
plt.scatter(result.epoch, result.history['val_acc'], marker='*')
plt.legend(loc='right')
#plt.show()
plt.savefig('acc.png')


result = model.predict(testData, batch_size=256)

result = np.argmax(result, 1)
result = [classes[x] for x in result]

result_file = 'result.csv'
predicts = pd.DataFrame()
predicts['id'] = list(range(1, len(result) + 1))
predicts['label'] = result
predicts.to_csv(result_file, index=False)
