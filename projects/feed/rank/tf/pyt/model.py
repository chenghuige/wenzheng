#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-08-01 23:08:36.979020
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

import lele

import numpy as np

# output logits!
class Wide(nn.Module):
  def __init__(self):
    super(Wide, self).__init__()
    self.emb = nn.Embedding(FLAGS.feature_dict_size + 1, 1)
    #self.bias = torch.zeros(1, requires_grad=True).cuda()
    # https://discuss.pytorch.org/t/tensors-are-on-different-gpus/1450/28 
    # without below multiple gpu will fail
    # RuntimeError: binary_op(): expected both inputs to be on same device, but input a is on cuda:0 and input b is on cuda:7 
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self, input):
    # print('--------------', input['index'][0])
    # print(len(input['index']), len(input['index'][0]))
    # exit(0)
    ids = input['index']
    values = input['value']

    x = self.emb(ids)
    x = x.squeeze(-1)
    # strange, eval for wide only addval will got worse result
    if FLAGS.wide_addval:
      x = x * values
    x = x.sum(1)
    x = x + self.bias
    return x  

class Deep(nn.Module):
  def __init__(self):
    super(Deep, self).__init__()
    self.emb = nn.Embedding(FLAGS.feature_dict_size + 1, FLAGS.hidden_size)
    self.emb_dim = FLAGS.hidden_size
    if FLAGS.field_emb:
      self.field_emb = nn.Embedding(FLAGS.field_dict_size + 1, FLAGS.hidden_size)
      self.emb_dim += FLAGS.hidden_size

    olen = self.emb_dim    
    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      self.mlp = nn.Linear(self.emb_dim, dims[0])
      # self.mlp = melt.layers.Mlp(dims, activation=FLAGS.dense_activation, drop_rate=FLAGS.mlp_drop)
      olen = dims[-1]

    act = FLAGS.dense_activation if FLAGS.deep_final_act else None

    self.dense = nn.Linear(olen, 1)

    if FLAGS.pooling != 'allsum':
      self.pooling = self.pooling = lele.layers.Pooling(FLAGS.pooling)         
   
  def forward(self, input):
    ids = input['index']
    values = input['value']
    fields = input['field']

    ids_mask = ids.eq(0)
    
    # if FLAGS.hidden_size > 50:
    #   with tf.device('/cpu:0'):
    #     x = self.emb(ids)
    # else:
    x = self.emb(ids)
    #print('x', x)
    if FLAGS.field_emb:
      x = torch.cat([x, self.field_emb(fields)], -1)

    if FLAGS.deep_addval:
      values = torch.unsqueeze(values, -1)
      x = x* values

    if FLAGS.field_concat:
      num_fields = FLAGS.field_dict_size
      #x = tf.math.unsorted_segment_sum(x, fields, num_fields)      
      x = melt.unsorted_segment_sum_emb(x, fields, num_fields)
      # like [512, 100 * 50]
      x = K.reshape(x, [-1, num_fields * self.emb_dim])
    else:
      if FLAGS.pooling == 'allsum':
        x = torch.sum(x, 1)
      else:
        assert FLAGS.index_addone, 'can not calc length for like 0,1,2,0,0,0'
        x = self.pooling(x, ids_mask)

    #print('x after pooling', x)
    if self.mlp:
      x = self.mlp(x)
      #print('x after mlp', x)
      x = F.dropout(F.relu(x), p=FLAGS.mlp_drop, training=self.training)
      #print('x after dropout', x)
    x = self.dense(x)
    x = x.squeeze(-1)

    #exit(0)
    #-----FIXME why large value ?
    return x

class WideDeep(nn.Module):   
  def __init__(self):
    super(WideDeep, self).__init__()
    self.wide = Wide()
    self.deep = Deep() 
    self.dense = nn.Linear(2, 1)

  # TODO verify we can remove training ? since we use K.Phrase() when sess.run
  def forward(self, input):
    w = self.wide(input)
    d = self.deep(input)
    if FLAGS.deep_wide_combine == 'concat':
      x = torch.stack([w, d], 1)
      x = self.dense(x)
      x = x.squeeze(-1)
    else:
      x = w + d
    return x

