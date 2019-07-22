# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function


import melt

import tensorflow as tf
from tensorflow import keras

# from keras.applications.densenet import DenseNet121,DenseNet169
# from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
#                           BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)

# from keras.models import Model

from tensorflow.keras.applications.densenet import DenseNet121,DenseNet169
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)

from tensorflow.keras.models import Model

def create_model(input_shape, n_out):
  # TODO can we ignore input and output shape ? as for text might be undetermined length
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

# class Model(tf.keras.Model):
#   def __init__(self, input_shape, n_out):
#     super(Model, self).__init__()
#     self.model = create_model(input_shape, n_out)
     
#   def call(self, x, training=False):
##    x = x['image']
#     return model(x, training=training)

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    base_model = DenseNet121(include_top=False,
                  weights=None,
                  input_shape=(224, 224, 3),
                  )
    base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
    self.model = base_model
     
  def call(self, x, training=False):
    #x = x['image']
    x = self.model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    n_out = 3
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)    
    return final_output
