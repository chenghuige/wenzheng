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
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)
from keras.models import Model

from config import *

def create_model(input_shape, n_out):
  input_tensor = Input(shape=input_shape)
  base_model = DenseNet121(include_top=False,
                  weights=None,
                  input_tensor=input_tensor)
  base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
  x = GlobalAveragePooling2D()(base_model.output)
  x = Dropout(0.5)(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  final_output = Dense(n_out, activation='softmax', name='final_output')(x)
  model = Model(input_tensor, final_output) 
  return model

