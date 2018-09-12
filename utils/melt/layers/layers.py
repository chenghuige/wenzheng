#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   layers.py
#        \author   chenghuige  
#          \date   2016-08-19 23:22:44.032101
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt
#since not from melt.layers.layers import * this is safe
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

import functools
import six

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

logging = melt.logging

def fully_connected(inputs,
                    num_outputs,
                    input_dim=None,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):

  use_bias = biases_initializer is not None

  #--------TODO: use commented code as layers.fully_connected then, for app code you must manully pass scope like 'mlp' otherwise evaluate will fail try to use Mlp_1 not resue=True b
  #--------by tf.variable_scope.reuse_variables() see http://stackoverflow.com/questions/40536665/tensorflow-varscope-reuse-variables
  #with variable_scope.variable_scope(
  #  scope, 'Mlp', [inputs],
  #  reuse=reuse) as vs:
  with tf.variable_scope(scope, 'fully_connected', [inputs], reuse=reuse):
    is_dense_input = True if isinstance(inputs, tf.Tensor) else False
    dtype=inputs.dtype.base_dtype if is_dense_input else inputs[1].values.dtype.base_dtype
    #sparse input must tell input_dim
    assert is_dense_input or input_dim is not None 
    if is_dense_input:
      shape = inputs.get_shape().as_list() 
      input_dim = shape[-1].value
      assert len(shape) == 2, "now only consider X shape dim as 2, TODO: make > 2 ok like layers.fully_connected"

    #-----------deal first hidden
    if is_dense_input:
     w_h =  tf.get_variable('weights',
                            shape=[input_dim, num_outputs],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            dtype=dtype,
                            trainable=trainable)
    else:
     with tf.device('/cpu:0'):
       w_h =  tf.get_variable('weights',
                              shape=[input_dim, num_outputs],
                              initializer=weights_initializer,
                              regularizer=weights_regularizer,
                              dtype=dtype,
                              trainable=trainable)

    if use_bias:
     b_h = tf.get_variable('biases',
                           shape=[num_outputs,],
                           initializer=biases_initializer,
                           regularizer=biases_regularizer,
                           dtype=dtype,
                           trainable=trainable)

    outputs = melt.matmul(inputs, w_h)
    if use_bias:
     outputs = nn.bias_add(outputs, b_h)
    if activation_fn is not None:
     outputs = activation_fn(outputs)  # pylint: disable=not-callable

    return outputs

linear = functools.partial(fully_connected, activation_fn=None)

def mlp(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp' if scope is None else scope
  with tf.variable_scope(scope):
    hidden = fully_connected(x, hidden_size, activation)
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    b_o = melt.get_bias('b_o', [output_size])
    return tf.nn.xw_plus_b(hidden, w_o, b_o)

def mlp_nobias(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp_nobias' if scope is None else scope
  with tf.variable_scope(scope):
    input_dim = utils.last_dimension(x.get_shape(), min_rank=2)
    if isinstance(x, tf.Tensor):
      w_h = melt.get_weights('w_h', [input_dim, hidden_size])
    else:
      with tf.device('/cpu:0'):
        w_h = melt.get_weights('w_h', [input_dim, hidden_size]) 
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    return  melt.mlp_forward_nobias(x, w_h, w_o, activation)


#def self_attention(outputs, seq_len, hidden_size, activation=tf.nn.tanh, scope='self_attention'):
def self_attention(outputs, seq_len, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, seq_len)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, seq_len, 1] -> [batch_size, seq_len]
    alphas = tf.squeeze(alphas) 
    return encoding, alphas

def self_attention_outputs(outputs, seq_len, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, seq_len)
    outputs = outputs * alphas
    return outputs

#def self_attention(outputs, seq_len, hidden_size, activation=tf.nn.tanh, scope='self_attention'):
def attention_layer(outputs, seq_len, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, seq_len)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, seq_len, 1] -> [batch_size, seq_len]
    alphas = tf.squeeze(alphas) 
    return encoding, alphas

def attention_outputs(outputs, seq_len, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, seq_len)
    outputs = outputs * alphas
    return outputs

def batch_norm(x, is_training=False, name=''):  
  return tf.contrib.layers.batch_norm(inputs=x,
                                      decay=0.95,
                                      center=True,
                                      scale=True,
                                      is_training=is_training,
                                      updates_collections=None,
                                      fused=True,
                                      scope=(name + 'batch_norm')) 

import melt 
from melt import dropout, softmax_mask

def gate_layer(res, keep_prob=1.0, is_train=None, scope='gate'):
  with tf.variable_scope(scope):
    dim = melt.get_shape(res, -1)
    d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
    gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name=scope)
    return res * gate

def gate(res, keep_prob=1.0, is_train=None, scope='gate'):
  with tf.variable_scope(scope):
    dim = melt.get_shape(res, -1)
    d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
    gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name=scope)
    return res * gate

# DO not only use semantic_fusion without droput, overfit prone
def semantic_fusion(input_vector, fusion_vectors, scope='semantic_fusion'):
  """Runs a semantic fusion unit on the input vector and the fusion vectors to produce an output. Input: input_vector: Vector of size [batch_size, ..., input_dim]. This vector must have the same size as each of the vectors in the fusion_vectors list. fusion_vectors: List of vectors of size [batch_size, ..., input_dim]. The vectors in this list must all have the same size as the input_vector. input_dim: The last dimension of the vectors (python scalar) Output: Vector of the same size as the input_vector. """
  with tf.variable_scope(scope):
      assert len(fusion_vectors) > 0
      vectors = tf.concat([input_vector] + fusion_vectors, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
      dim = melt.get_shape(input_vector, -1)
      r =  tf.layers.dense(vectors, dim, use_bias=True, activation=tf.nn.tanh, name='composition')
      g = tf.layers.dense(vectors, dim, use_bias=True, activation=tf.nn.sigmoid, name='gate')
      return g * r + (1 - g) * input_vector 

def semantic_fusion_combine(x, y):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_fc')
  return semantic_fusion(x, [y, x * y, x - y])

def dsemantic_fusion(input_vector, fusion_vectors, keep_prob=1.0, is_train=None, scope='semantic_fusion'):
  """Runs a semantic fusion unit on the input vector and the fusion vectors to produce an output. Input: input_vector: Vector of size [batch_size, ..., input_dim]. This vector must have the same size as each of the vectors in the fusion_vectors list. fusion_vectors: List of vectors of size [batch_size, ..., input_dim]. The vectors in this list must all have the same size as the input_vector. input_dim: The last dimension of the vectors (python scalar) Output: Vector of the same size as the input_vector. """
  with tf.variable_scope(scope):
      assert len(fusion_vectors) > 0
      vectors = tf.concat([input_vector] + fusion_vectors, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
      dim = melt.get_shape(input_vector, -1)
      dv = dropout(vectors, keep_prob=keep_prob, is_train=is_train)
      r =  tf.layers.dense(dv, dim, use_bias=True, activation=tf.nn.tanh, name='composition')
      g = tf.layers.dense(dv, dim, use_bias=True, activation=tf.nn.sigmoid, name='gate')
      return g * r + (1 - g) * input_vector 

def dsemantic_fusion_combine(x, y, keep_prob=1.0, is_train=None):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_fc')
  return dsemantic_fusion(x, [y, x * y, x - y], keep_prob, is_train)

def dsemantic_fusion_simple_combine(x, y, keep_prob=1.0, is_train=None):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_simple_fc')
  return dsemantic_fusion(x, [y], keep_prob, is_train, scope='semantic_fusion_simple')  
  
# # used in toxic for self attetion, from squad HSTKU rnet code
# # TODO dot attention semantic fu
# def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
#   with tf.variable_scope(scope):
#     d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
#     d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
#     JX = tf.shape(inputs)[1]
    
#     with tf.variable_scope("attention"):
#       inputs_ = tf.layers.dense(d_inputs, hidden, use_bias=False, activation=tf.nn.relu, name='inputs')
#       memory_ = tf.layers.dense(d_memory, hidden, use_bias=False, activation=tf.nn.relu, name='memory')
#       # [batch, input_len, meory_len]
#       outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
#       if mask is not None:
#         mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
#         logits = tf.nn.softmax(softmax_mask(outputs, mask))
#       else:
#         logits = tf.nn.softmax(outputs)
#       outputs = tf.matmul(logits, memory)

#       res = tf.concat([inputs, outputs], axis=2)

#     # self attention also using gate ? TODO SFU ?
#     # from R-NET part 3.2 gated attention-based recurrent networks
#     with tf.variable_scope("gate"):
#       dim = melt.get_shape(res, -1)
#       d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
#       gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name='gate')
#       return res * gate

# TODO duplicate code  
def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, combiner='gate', scope="dot_attention"):
  with tf.variable_scope(scope):
    d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
    d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
    JX = tf.shape(inputs)[1]
    
    with tf.variable_scope("attention"):
      inputs_ = tf.layers.dense(d_inputs, hidden, use_bias=False, activation=tf.nn.relu, name='inputs')
      memory_ = tf.layers.dense(d_memory, hidden, use_bias=False, activation=tf.nn.relu, name='memory')
      # [batch, input_len, meory_len]
      outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
      if mask is not None:
       mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
       logits = tf.nn.softmax(softmax_mask(outputs, mask))
      else:
        logits = tf.nn.softmax(outputs)
      outputs = tf.matmul(logits, memory)

    # for label attention seems gate is better then sfu
    if combiner == 'gate':
      logging.info('dot attention using gate')
      res = tf.concat([inputs, outputs], axis=2)
      with tf.variable_scope("gate"):
        dim = melt.get_shape(res, -1)
        d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
        gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name='gate')
        return res * gate
    elif combiner == 'sfu' or combiner == 'dsfu':
      logging.info('dot attention using sfu')
      with tf.variable_scope("sfu"):
        if melt.get_shape(outputs, -1) !=  melt.get_shape(inputs, -1):
          outputs = tf.layers.dense(outputs, melt.get_shape(inputs, -1), use_bias=False, activation=None, name='outputs_fc')
        return dsemantic_fusion_combine(inputs, outputs, keep_prob, is_train)
    elif combiner == 'simple_sfu' or combiner == 'simple_dsfu' or combiner == 'ssfu':
      logging.info('dot attention using ssfu')
      with tf.variable_scope("ssfu"):
        if melt.get_shape(outputs, -1) !=  melt.get_shape(inputs, -1):
          outputs = tf.layers.dense(outputs, melt.get_shape(inputs, -1), use_bias=False, activation=None, name='outputs_fc')
        return dsemantic_fusion_simple_combine(inputs, outputs, keep_prob, is_train)     
    else:
      raise ValueError(combiner)

keras = tf.keras
layers = tf.keras.layers
class MaxPooling(keras.layers.Layer):
  def call(self, outputs, sequence_length=None, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling(outputs, sequence_length, axis, reduce_func)

class MaxPooling2(keras.layers.Layer):
  def call(self, outputs, sequence_length, sequence_length2, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling2(outputs, sequence_length, sequence_length2, axis, reduce_func)

from melt import dropout
from melt.rnn import OutputMethod, encode_outputs

class InitState(keras.layers.Layer):
  def __init__(self, num_layers, num_units, name='init_fw'):
    super(InitState, self).__init__()
    self.init = [None] * num_layers
    for layer in range(num_layers):
      self.init[layer] = self.add_variable("%s_%d" % (name, layer), [1, num_units], initializer=tf.zeros_initializer())

  def call(self, layer, batch_size=None):
    if batch_size is None:
      return self.init[layer]
    else:
      return tf.tile(self.init[layer], [batch_size, 1])

class Dropout(keras.layers.Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout).__init__(self, **kwargs)
    self.rate = rate 
    self.noise_shape = noise_shape
    self.seed = seed
  
  def call(self, inputs, training=False):
    if not training or self.rate <= 0.:
      return inputs
    else:
      scale = 1.
      shape = tf.shape(inputs)
      if mode == 'embedding':
        noise_shape = [shape[0], 1]
        scale = 1 - self.rate
      elif mode == 'recurrent' and len(inputs.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]] 
      return tf.nn.dropout(inputs, 1 - self.rate, noise_shape=noise_shape) * scale


# TODO for safe both graph and eager, do not use keep prob, just pass Dropout result to call
# FIXME so share_dropout only work for graph mode !! and now also eager mode has droput problem... as each batch call use 
# same dropout mask if share_dropout=True..
class CudnnRnn(keras.Model):
  def __init__(self,  
                num_layers, 
                num_units, 
                keep_prob=1.0, 
                share_dropout=True,
                train_init_state=True,
                cell='gru', 
                **kwargs):
    super(CudnnRnn, self).__init__(**kwargs)
    self.cell = cell
    if isinstance(cell, str):
      if cell == 'gru':
        if tf.test.is_gpu_available():
          self.Cell = layers.CuDNNGRU
        else:
          self.Cell = layers.GRU
      elif cell == 'lstm':
        if tf.test.is_gpu_available():
          self.Cell = layers.CuDNNLSTM
        else:
          self.Cell = layers.LSTM
      else:
        raise ValueError(cell)

    logging.info('cudnn cell:', self.cell)
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    assert num_units % 4 == 0, 'bad performance for units size not % 4'
    self.num_units = num_units

    # for share dropout between like context and question in squad (machine reading task)
    # rnn = gru(num_layers=FLAGS.num_layers, num_units=d, keep_prob=keep_prob, is_train=self.is_training)
    # c = rnn(c_emb, seq_len=c_len)
    # scope.reuse_variables()
    # q = rnn(q_emb, seq_len=q_len)
    self.share_dropout = share_dropout
    

    # TODO FIXME hack for model.save try to save self.dropout_mask_fw , even though I think should not... TODO  how to NoDependency
    # ValueError: Unable to save the object ListWrapper([<tf.Tensor: id=115958, shape=(32, 1, 300), dtype=float32, numpy=
    #array(
    #        1.4285715]]], dtype=float32)>]) (a list wrapper constructed to track checkpointable TensorFlow objects).
    #  A list element was replaced (__setitem__), deleted, or inserted. In order to support restoration on object creation, tracking is exclusively for append-only data structures.
    # if you don't need this list checkpointed, wrap it in a tf.contrib.checkpoint.NoDependency object; it will be automatically un-wrapped and subsequently ignored.

    self.dropout_mask_fw = [None] * num_layers
    self.dropout_mask_bw = [None] * num_layers 

    self.train_init_state = train_init_state
    self.init_fw = [None] * num_layers
    self.init_bw = [None] * num_layers 

    self.state = None
    
    def gru(units):
      # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a 
      # significant speedup).
      if tf.test.is_gpu_available():
        return self.Cell(units, 
                         return_sequences=True, 
                         return_state=True, 
                         recurrent_initializer='glorot_uniform')
      else:
        return self.Cell(units, 
                         return_sequences=True, 
                         return_state=True, 
                         recurrent_activation='sigmoid', 
                         recurrent_initializer='glorot_uniform')

    # well seems will only consider to save item or list ont list of tuples so self.grus = [] self.grus.append((gru_w, gru_bw)) will not save..  TODO FIXME
    self.gru_fws = []
    self.gru_bws = []
    for layer in range(num_layers):
      #with tf.variable_scope('fw'):
      gru_fw = gru(num_units)
      #with tf.variable_scope('bw'):
      gru_bw = gru(num_units)
      self.gru_fws.append(gru_fw)
      self.gru_bws.append(gru_bw)

    # if self.train_init_state:
    #   for layer in range(num_layers):
    #     # well TODO! add_variable not allowed in keras.Model but using keras.layers.Layer you should not use other layers otherwise not save them
    #     self.init_fw[layer] = self.add_variable("init_fw_%d" % layer, [1, num_units], initializer=tf.zeros_initializer())
    #     self.init_bw[layer] = self.add_variable("init_bw_%d" % layer, [1, num_units], initializer=tf.zeros_initializer()) 

    if self.train_init_state:
      # well TODO! add_variable not allowed in keras.Model but using keras.layers.Layer you should not use other layers otherwise not save them
      self.init_fw_layer = InitState(num_layers, num_units, 'init_fw')
      self.init_bw_layer = InitState(num_layers, num_units, 'init_bw')

  def set_dropout_mask(self, mask_fw, mask_bw):
    self.dropout_mask_fw = mask_fw 
    self.dropout_mask_bw = mask_bw

  def set_init_states(self, init_fw, init_bw):
    self.init_fw = init_fw
    self.init_bw = init_bw

  def reset_init_states(self):
    self.init_fw = [None] * self.num_layers
    self.init_bw = [None] * self.num_layers   

  # @property
  # def trainable_weights(self):
  #   if self.trainable and self.built:
  #     return [[self.init_fw[layer], self.init_bw[layer], self.grus[0][layer], self.grus[1][layer] for layer in range(num_layers)]
  #   return []  

  def call(self, 
           inputs, 
           seq_len, 
           emb=None, 
           mask_fws = None,
           mask_bws = None,
           concat_layers=True, 
           output_method=OutputMethod.all, 
           training=False):
    if emb is not None:
      inputs = tf.nn.embedding_lookup(emb, inputs)
      
    outputs = [inputs]

    #states = []
    keep_prob = self.keep_prob
    num_units = self.num_units
    batch_size = melt.get_batch_size(inputs)

    for layer in range(self.num_layers):
      input_size_ = melt.get_shape(inputs, -1) if layer == 0 else 2 * num_units

      gru_fw, gru_bw = self.gru_fws[layer], self.gru_bws[layer]
      
      if mask_fws is not None:
        mask_fw = mask_fws[layer]
      else:
        if not self.share_dropout:
          mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                        keep_prob=keep_prob, is_train=training, mode=None)
        else:
          if self.dropout_mask_fw[layer] is None or (tf.executing_eagerly() and batch_size != self.dropout_mask_fw[layer].shape[0]):
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                      keep_prob=keep_prob, is_train=training, mode=None)
            self.dropout_mask_fw[layer] = mask_fw
          else:
            mask_fw = self.dropout_mask_fw[layer]
      
      if self.train_init_state:
        #init_fw = tf.tile(self.init_fw[layer], [batch_size, 1])
        #init_fw = tf.tile(self.init_fw_layer(layer), [batch_size, 1])
        init_fw = self.init_fw_layer(layer, batch_size)
      else:
        init_fw = None

      out_fw, state_fw = gru_fw(outputs[-1] * mask_fw, init_fw)

      if mask_bws is not None:
        mask_bw = mask_bws[layer]
      else:
        if not self.share_dropout:
          mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob, is_train=training, mode=None)
        else:
          if self.dropout_mask_bw[layer] is None or (tf.executing_eagerly() and batch_size != self.dropout_mask_bw[layer].shape[0]):
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                      keep_prob=keep_prob, is_train=training, mode=None)
            self.dropout_mask_bw[layer] = mask_bw
          else:
            mask_bw = self.dropout_mask_bw[layer]

      inputs_bw = tf.reverse_sequence(
          outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
      
      if self.train_init_state:
        #init_bw = tf.tile(self.init_bw[layer], [batch_size, 1])
        #init_bw = tf.tile(self.init_bw_layer(layer), [batch_size, 1])
        init_bw = self.init_bw_layer(layer, batch_size)
      else:
        init_bw = None

      out_bw, state_bw = gru_bw(inputs_bw, init_bw)
      out_bw = tf.reverse_sequence(
          out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)

      outputs.append(tf.concat([out_fw, out_bw], axis=2))

    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
    else:
      res = outputs[-1]

    res = encode_outputs(res, output_method=output_method, sequence_length=seq_len)

    self.state = (state_fw, state_bw)

    return res

class CudnnRnn2(CudnnRnn):
  def __init__(self,  
               **kwargs):
    super(CudnnRnn2, self).__init__(**kwargs)
    
  def call(self, 
           inputs, 
           seq_len,
           inputs2,
           seq_len2, 
           mask_fws,
           mask_bws,
           concat_layers=True, 
           output_method=OutputMethod.all, 
           training=False):
      
    outputs = [inputs]
    outputs2 = [inputs2]

    keep_prob = self.keep_prob
    num_units = self.num_units
    batch_size = melt.get_batch_size(inputs)

    for layer in range(self.num_layers):
      input_size_ = melt.get_shape(inputs, -1) if layer == 0 else 2 * num_units

      gru_fw, gru_bw = self.gru_fws[layer], self.gru_bws[layer]
      
      
      if self.train_init_state:
        init_fw = self.init_fw_layer(layer, batch_size)
      else:
        init_fw = None

      mask_fw = mask_fws[layer]
      out_fw, state_fw = gru_fw(outputs[-1] * mask_fw, init_fw)
      out_fw2, state_fw2 = gru_fw(outputs2[-1] * mask_fw, state_fw)

      mask_bw = mask_bws[layer]
      inputs_bw = tf.reverse_sequence(
          outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
      inputs_bw2 = tf.reverse_sequence(
          outputs2[-1] * mask_bw, seq_lengths=seq_len2, seq_dim=1, batch_dim=0)
      
      if self.train_init_state:
        init_bw = self.init_bw_layer(layer, batch_size)
      else:
        init_bw = None

      out_bw, state_bw = gru_bw(inputs_bw, init_bw)
      out_bw2, state_bw2 = gru_bw(inputs_bw2, state_bw)

      outputs.append(tf.concat([out_fw, out_bw], axis=2))
      outputs2.append(tf.concat([out_fw2, out_bw2], axis=2))

    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
      res2 = tf.concat(outputs2[1:], axis=2)
    else:
      res = outputs[-1]
      res2 = outpus2[-1]

    res = tf.concat([res, res2], axis=1)

    res = encode_outputs(res, output_method=output_method, sequence_length=seq_len)

    self.state = (state_fw2, state_bw2)
    return res
  
class DotAttention(keras.Model):
  def __init__(self,
               hidden,
               keep_prob=1.0,
               combiner='gate',
               scope='dot_attention',  
               **kwargs):
    super(DotAttention, self).__init__(**kwargs)
    self.hidden = hidden
    self.keep_prob = keep_prob
    self.combiner = combiner
    self.scope = scope

    self.inputs_dense = keras.layers.Dense(hidden, use_bias=False, activation=tf.nn.relu)
    self.memory_dense = keras.layers.Dense(hidden, use_bias=False, activation=tf.nn.relu)

    self.step = -1

  def call(self, inputs, memory, mask, training=False):
    self.step += 1
    combiner = self.combiner
    with tf.variable_scope(self.scope):
      d_inputs = dropout(inputs, keep_prob=self.keep_prob, training=training)
      d_memory = dropout(memory, keep_prob=self.keep_prob, training=training)
      JX = tf.shape(inputs)[1]
      
      with tf.variable_scope("attention"):
        inputs_ = self.inputs_dense(d_inputs)
        memory_ = self.memory_dense(d_memory)

        outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (self.hidden ** 0.5)
        if mask is not None:
          mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
          logits = tf.nn.softmax(softmax_mask(outputs, mask))
        else:
          logits = tf.nn.softmax(outputs)

        outputs = tf.matmul(logits, memory)

      if combiner == 'gate':
        res = tf.concat([inputs, outputs], axis=2)
        with tf.variable_scope("gate"):
          dim = melt.get_shape(res, -1)
          d_res = dropout(res, keep_prob=self.keep_prob, training=training)
          if self.step == 0:
            self.gate_dense = keras.layers.Dense(dim, use_bias=False, activation=tf.nn.sigmoid)
          gate = self.gate_dense(d_res)
          return res * gate
      elif combiner == 'sfu' or combiner == 'dsfu':
        if melt.get_shape(outputs, -1) !=  melt.get_shape(inputs, -1):
          if self.step == 0:
            self.sfu_dense = keras.layers.Dense(melt.get_shape(inputs, -1), use_bias=False, activation=None)
          outputs = self.sfu_dense(outputs)
        return melt.dsemantic_fusion_combine(inputs, outputs, keep_prob, training)
      else:
        raise ValueError(combiner)
      
