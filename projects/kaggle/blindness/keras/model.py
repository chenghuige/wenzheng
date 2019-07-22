#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:05.709750
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from keras.applications.densenet import DenseNet121,DenseNet169
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,
                          Multiply, Lambda)
from keras.models import Model
#from keras import backend as K
import keras
import tensorflow as tf

from config import *

def create_model(input_shape, n_out, loss_type=None):
  input_tensor = Input(shape=input_shape)
  base_model = DenseNet121(include_top=False,
                  weights=None,
                  input_tensor=input_tensor)
  base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
  x = GlobalAveragePooling2D()(base_model.output)
  x = Dropout(0.5)(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)

  if 'linear_regression' in loss_type:
    final_output = Dense(1, activation='linear', name='final_output')(x)
  elif 'sigmoid_regression' in loss_type:
    x = Dense(1, activation='sigmoid')(x)
    #final_output = Multiply(name='final_output')([x, tf.constant(10.)])
    #https://github.com/keras-team/keras/issues/10204
    #final_output = Lambda(lambda x: x * 10.0, name='final_output')(x)
    final_output = Lambda(lambda x: x * 4.0, name='final_output')(x)
  elif 'sigmoid2_regression' in loss_type:
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)
  elif loss_type == 'ordinal_classification':
    final_output = Dense(n_out, activation='sigmoid', name='final_output')(x)
  elif loss_type == 'ordinal2_classification':
    final_output = Dense(n_out - 1, activation='sigmoid', name='final_output')(x)
  else:
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
  
  model = Model(input_tensor, final_output) 
  return model

