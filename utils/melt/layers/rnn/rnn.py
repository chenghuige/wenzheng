#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2018-09-20 07:31:06.789341
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

keras = tf.keras
layers = tf.keras.layers
Layer = layers.Layer

import melt
logging = melt.logging
from melt.rnn import encode_outputs, OutputMethod

from melt import dropout

class InitStates(Layer):
  def __init__(self, num_layers, num_units, name='init_fw'):
    super(InitStates, self).__init__()
    # problem in naming.. might use check point list ?
    self.init = [None] * num_layers
    for layer in range(num_layers):
      self.init[layer] = self.add_variable("%s_%d" % (name, layer), [1, num_units], initializer=tf.zeros_initializer())

    self.init = tf.contrib.checkpoint.List(self.init)

  def call(self, layer, batch_size=None):
    if batch_size is None:
      return self.init[layer]
    else:
      return tf.tile(self.init[layer], [batch_size, 1])


# TODO for safe both graph and eager, do not use keep prob, just pass Dropout result to call
# FIXME so share_dropout only work for graph mode !! and now also eager mode has droput problem... as each batch call use 
# same dropout mask if share_dropout=True..
class CudnnRnn(keras.Model):
  def __init__(self,  
                num_layers, 
                num_units, 
                keep_prob=1.0, 
                share_dropout=False,
                recurrent_dropout=True,
                bw_dropout=False,
                train_init_state=True,
                concat_layers=True, 
                output_method=OutputMethod.all, 
                return_state=False,
                residual_connect=False,
                cell='gru', 
                **kwargs):
    super(CudnnRnn, self).__init__(**kwargs)
    self.cell = cell
    self.return_state = return_state
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
    # c = rnn(c_emb, sequence_length=c_len)
    # scope.reuse_variables()
    # q = rnn(q_emb, sequence_length=q_len)
    self.share_dropout = share_dropout
    self.recurrent_dropout = recurrent_dropout
    # when not using recurrent_dropout if bw_dropout will let backward rnn using different dropout then forward
    self.bw_dropout = bw_dropout 

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

    # no concat for rent will convergence slower and hurt performance
    self.concat_layers = concat_layers
    self.output_method = output_method
    self.residual_connect = residual_connect

    # hurt performance a lot in rnet 
    if self.residual_connect:
      self.residual_linear = layers.Dense(self.num_units * 2)
      self.layer_norm = melt.layers.LayerNorm()
    
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

    self.gru_fws = tf.contrib.checkpoint.List(self.gru_fws)
    self.gru_bws = tf.contrib.checkpoint.List(self.gru_bws)

    # if self.train_init_state:
    #   for layer in range(num_layers):
    #     # well TODO! add_variable not allowed in keras.Model but using keras.layers.Layer you should not use other layers otherwise not save them
    #     self.init_fw[layer] = self.add_variable("init_fw_%d" % layer, [1, num_units], initializer=tf.zeros_initializer())
    #     self.init_bw[layer] = self.add_variable("init_bw_%d" % layer, [1, num_units], initializer=tf.zeros_initializer()) 

    if self.train_init_state:
      # well TODO! add_variable not allowed in keras.Model but using keras.layers.Layer you should not use other layers otherwise not save them
      # TODO name is not very ok... without scope ...
      # embedding_kernel (DT_FLOAT) [20,300]  # should in r_net
      # global_step (DT_INT64) []
      # init_bw_0 (DT_FLOAT) [1,200]  # should in cudnn_rnn
      # init_bw_0_1 (DT_FLOAT) [1,200]
      # init_bw_0_2 (DT_FLOAT) [1,200]
      # init_bw_0_3 (DT_FLOAT) [1,200]
      # init_bw_1 (DT_FLOAT) [1,200]
      # init_fw_0 (DT_FLOAT) [1,200]
      # init_fw_0_1 (DT_FLOAT) [1,200]
      # init_fw_0_2 (DT_FLOAT) [1,200]
      # init_fw_0_3 (DT_FLOAT) [1,200]
      # init_fw_1 (DT_FLOAT) [1,200]
      # learning_rate_weight (DT_FLOAT) []
      # r_net/cudnn_rnn_2/cu_dnngru_6/bias (DT_FLOAT) [1200]
      # r_net/cudnn_rnn_2/cu_dnngru_6/kernel (DT_FLOAT) [1100,600]
      # r_net/cudnn_rnn_2/cu_dnngru_6/recurrent_kernel (DT_FLOAT) [200,600]
      # r_net/cudnn_rnn_2/cu_dnngru_7/bias (DT_FLOAT) [1200]
      # r_net/cudnn_rnn_2/cu_dnngru_7/kernel (DT_FLOAT) [1100,600]
      # r_net/cudnn_rnn_2/cu_dnngru_7/recurrent_kernel (DT_FLOAT) [200,600]
      # r_net/cudnn_rnn_3/cu_dnngru_8/bias (DT_FLOAT) [1200]

      self.init_fw_layer = InitStates(num_layers, num_units, 'init_fw')
      self.init_bw_layer = InitStates(num_layers, num_units, 'init_bw')
      if self.cell == 'lstm':
        self.init_fw2_layer = InitStates(num_layers, num_units, 'init_fw2')
        self.init_bw2_layer = InitStates(num_layers, num_units, 'init_bw2')


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
           x, 
           sequence_length=None, 
           mask_fws = None,
           mask_bws = None,
           concat_layers=None, 
           output_method=None, 
           training=False):

    concat_layers = concat_layers or self.concat_layers
    output_mehtod = output_method or self.output_method

    if self.residual_connect:
      x = self.residual_linear(x)

    outputs = [x]

    #states = []
    keep_prob = self.keep_prob
    num_units = self.num_units
    batch_size = melt.get_batch_size(x)

    if sequence_length is None:
      len_ = melt.get_shape(x, 1)
      sequence_length = tf.ones([batch_size,], dtype=tf.int64) * len_

    for layer in range(self.num_layers):
      input_size_ = melt.get_shape(x, -1) if layer == 0 else 2 * num_units

      gru_fw, gru_bw = self.gru_fws[layer], self.gru_bws[layer]
      
      if self.train_init_state:
        #init_fw = tf.tile(self.init_fw[layer], [batch_size, 1])
        #init_fw = tf.tile(self.init_fw_layer(layer), [batch_size, 1])
        init_fw = self.init_fw_layer(layer, batch_size)
        if self.cell == 'lstm':
          init_fw = (init_fw, self.init_fw2_layer(layer, batch_size))
      else:
        init_fw = None

      if self.recurrent_dropout:
        if mask_fws is not None:
          mask_fw = mask_fws[layer]
        else:
          if not self.share_dropout:
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                          keep_prob=keep_prob, training=training, mode=None)
          else:
            if self.dropout_mask_fw[layer] is None or (tf.executing_eagerly() and batch_size != self.dropout_mask_fw[layer].shape[0]):
              mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                        keep_prob=keep_prob, training=training, mode=None)
              self.dropout_mask_fw[layer] = mask_fw
            else:
              mask_fw = self.dropout_mask_fw[layer]
        
        inputs_fw = outputs[-1] * mask_fw
      else:
        inputs_fw = dropout(outputs[-1], keep_prob=keep_prob, training=training, mode=None)

      # https://stackoverflow.com/questions/48233400/lstm-initial-state-from-dense-layer
      # gru and lstm different ... state lstm need tuple (,) states as input state\
      if self.cell == 'gru':
        out_fw, state_fw = gru_fw(inputs_fw, init_fw)
      else:
        out_fw, state_fw1, state_fw2 = gru_fw(inputs_fw, init_fw)
        state_fw = (state_fw1, state_fw2)

      if self.train_init_state:
        #init_bw = tf.tile(self.init_bw[layer], [batch_size, 1])
        #init_bw = tf.tile(self.init_bw_layer(layer), [batch_size, 1])
        init_bw = self.init_bw_layer(layer, batch_size)
        if self.cell == 'lstm':
          init_bw = (init_bw, self.init_bw2_layer(layer, batch_size))
      else:
        init_bw = None

      if mask_bws is not None:
        mask_bw = mask_bws[layer]
      else:
        if not self.share_dropout:
          mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob, training=training, mode=None)
        else:
          if self.dropout_mask_bw[layer] is None or (tf.executing_eagerly() and batch_size != self.dropout_mask_bw[layer].shape[0]):
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                      keep_prob=keep_prob, training=training, mode=None)
            self.dropout_mask_bw[layer] = mask_bw
          else:
            mask_bw = self.dropout_mask_bw[layer]

      if self.recurrent_dropout:
        inputs_bw = outputs[-1] * mask_bw
      else:
        if self.bw_dropout:
          inputs_bw = dropout(outputs[-1], keep_prob=keep_prob, training=training, mode=None)
        else:
          inputs_bw = inputs_fw

      inputs_bw = tf.reverse_sequence(
          inputs_bw, seq_lengths=sequence_length, seq_axis=1, batch_axis=0)

      if self.cell == 'gru': 
        out_bw, state_bw = gru_bw(inputs_bw, init_bw)
      else:
        out_bw, state_bw1, state_bw2 = gru_bw(inputs_bw, init_bw)
        state_bw = (state_bw1, state_bw2)
           
      out_bw = tf.reverse_sequence(
          out_bw, seq_lengths=sequence_length, seq_axis=1, batch_axis=0)

      outputs.append(tf.concat([out_fw, out_bw], axis=2))
      if self.residual_connect:
        outputs[-1] = self.batch_norm(outputs[-2] + outputs[-1])

    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
    else:
      res = outputs[-1]

    res = encode_outputs(res, output_method=output_method, sequence_length=sequence_length)

    self.state = (state_fw, state_bw)
    if not self.return_state:
      return res
    else:
      return res, self.state

# depreciated..
class CudnnRnn2(CudnnRnn):
  def __init__(self,  
               **kwargs):
    super(CudnnRnn2, self).__init__(**kwargs)
    
  def call(self, 
           inputs, 
           sequence_length,
           inputs2,
           sequence_length2, 
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
          outputs[-1] * mask_bw, sequence_lengthgths=sequence_length, seq_axis=1, batch_axis=0)
      inputs_bw2 = tf.reverse_sequence(
          outputs2[-1] * mask_bw, sequence_lengthgths=sequence_length2, seq_axis=1, batch_axis=0)
      
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

    res = encode_outputs(res, output_method=output_method, sequence_length=sequence_length)

    self.state = (state_fw2, state_bw2)
    return res
