#!/usr/bin/env python
# ==============================================================================
#          \file   mlp.py
#        \author   chenghuige  
#          \date   2016-08-16 17:13:04.699501
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import tensorflow.contrib.slim as slim

#from D:\other\tensorflow\tensorflow\tensorflow\contrib\layers\python\layers\layers.py

from tensorflow.python.framework import tensor_shape

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import  normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

import melt

def forward(inputs,
            num_outputs, 
            input_dim=None,
            hiddens=[200], 
            activation_fn=tf.nn.relu,
            weights_initializer=initializers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=init_ops.zeros_initializer(),
            biases_regularizer=None,
            reuse=None,
            scope=None
            ):
  """
  similary as melt.slim.layers.mlp but the first step(from input to first hidden adjusted so input can be sparse)
  """

  scope = 'mlp' if scope is None else scope
  with tf.variable_scope(scope):
    if len(hiddens) == 0:
      #logistic regression
      return melt.linear(inputs, 
                     num_outputs, 
                     input_dim=input_dim,
                     weights_initializer=weights_initializer,
                     weights_regularizer=weights_regularizer,
                     biases_initializer=biases_initializer,
                     biases_regularizer=biases_regularizer,
                     scope='linear')

    outputs = melt.layers.fully_connected(inputs, 
                                   hiddens[0], 
                                   input_dim=input_dim,
                                   activation_fn=activation_fn,
                                   weights_initializer=weights_initializer,
                                   weights_regularizer=weights_regularizer,
                                   biases_initializer=biases_initializer,
                                   biases_regularizer=biases_regularizer,
                                   reuse=reuse,
                                   scope='fc_0')

   #--------other hidden layers
    # for i in xrange(len(hiddens) -1):
    #   outputs = slim.fully_connected(outputs, hiddens[i + 1], 
    #                          activation_fn=activation_fn, 
    #                          weights_initializer=weights_initializer,
    #                          weights_regularizer=weights_regularizer,
    #                          biases_initializer=biases_initializer, 
    #                          biases_regularizer=biases_regularizer,
    #                          scope='fc_%d'%i+1)

    slim.stack(outputs, slim.fully_connected, 
      hiddens[1:], 
      activation_fn=activation_fn,
      weights_initializer=weights_initializer,
      weights_regularizer=weights_regularizer,
      biases_initializer=biases_initializer,
      biases_regularizer=biases_regularizer,
      scope='fc')

    return slim.linear(outputs, 
                     num_outputs, 
                     weights_initializer=weights_initializer,
                     weights_regularizer=weights_regularizer,
                     biases_initializer=biases_initializer,
                     biases_regularizer=biases_regularizer,
                     scope='linear')
