#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dynamic_dense.py
#        \author   chenghuige  
#          \date   2018-09-12 10:53:31.858019
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
tfe = tf.contrib.eager

tf.enable_eager_execution()

keras = tf.keras

class Layer(keras.layers.Layer):
  def __init__(self):
    super(Layer, self).__init__()
    self.abc = self.add_variable("abc", [1, 3], initializer=tf.ones_initializer(dtype=tf.float32))

  def call(self, x):
    result = x + self.abc
    self.abc = self.abc * 5
    return result


class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    #self.abc = self.add_variable("abc", [1, 3], initializer=tf.ones_initializer(dtype=tf.float32))
    #self.encode = keras.layers.Dense(5, activation=None)
    
    self.abc = tfe.Variable(tf.ones([1, 3], dtype=tf.float32), name='abc')
    #self.abc = self.abc * 5
    #self.abc = Layer()

    self.inited = False

  def call(self, x):
    if not self.inited:
      self.inited = True
      print(tf.shape(x)[-1])
      self.fc = keras.layers.Dense(tf.shape(x)[-1], activation=None)
    print(x)
    return self.fc(x)
    #return self.encode(x + self.abc)
    
    #result = x + self.abc
    #self.abc = self.abc * 5
    #return result

    #return self.abc(x)


model = Model() 
print(model(tf.constant([[1., 2., 3.]])))
#print(model.abc)

#m.save_weights('/home/gezi/tmp/model')

checkpoint = tf.train.Checkpoint(model=model)
ckpt_dir = '/home/gezi/tmp/model'

# latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
# checkpoint.restore(latest_checkpoint)
# print(model([[1., 2., 3.]]))


checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
checkpoint.save(checkpoint_prefix)

#model.abc = model.abc + 1. 
#checkpoint.save(checkpoint_prefix)

#model.abc -= 1.
# latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
# checkpoint.restore(latest_checkpoint)

# print(model([[1., 2., 3.]]))

