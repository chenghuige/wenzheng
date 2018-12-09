#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-17 20:34:23.281520
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
import numpy as np

from algos.weights import *
from algos.config import NUM_CLASSES, NUM_ATTRIBUTES

import lele
import torch
from torch import nn
from torch.nn import functional as F

class Criterion(object):
  def __init__(self, class_weights=None):
    self.class_weights = class_weights
    self.loss_fn = torch.nn.CrossEntropyLoss()
    self.loss_fn2 = torch.nn.CrossEntropyLoss(reduction='none')
    self.bloss_fn = nn.BCEWithLogitsLoss()
    # if FLAGS.use_class_weights:
    #   self.weighted_loss_fn = [None] * NUM_ATTRIBUTES
    #   class_weights = np.reshape(class_weights, [NUM_ATTRIBUTES, NUM_CLASSES])
    #   class_weights = np.log(np.log(class_weights + 1.) + 1.)
    #   logging.info('class_weights', class_weights)
    #   for i in range(NUM_ATTRIBUTES):
    #     self.weighted_loss_fn[i] = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights[i]).cuda())

  def calc_soft_label_loss(self, y_, y, num_classes):
    y = y.view(-1, num_classes)
    y_ = y_.view(-1, num_classes)
    log_probs = F.log_softmax(y_, dim=-1)
    loss = -torch.sum(log_probs * y, -1)
    loss = loss.mean()
    return loss

  def forward(self, model, x, y, training=False):
    y_ = model(x)

    if FLAGS.use_soft_label:
      return self.calc_soft_label_loss(y_, y, NUM_CLASSES)
    
    #print(y.shape, y_.shape)
    # without view Expected target size (32, 4), got torch.Size([32, 20])
    if training and FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
      loss = self.loss_fn2(y_.view(-1, NUM_CLASSES), y.view(-1)).view(-1, NUM_ATTRIBUTES)
      # stop some gradients due to learning_rate weights
      loss = lele.adjust_lrs(loss)
      loss = loss.mean()
    # elif FLAGS.use_class_weights:
    #   losses = []
    #   for i in range(NUM_ATTRIBUTES):
    #     loss = self.weighted_loss_fn[i](y_[:,i,:], y[:,i])
    #     losses.append(loss)
    #   loss = torch.mean(torch.stack(losses))
    else:
      loss = self.loss_fn(y_.view(-1, NUM_CLASSES), y.view(-1))  
    
    # depreciated add neu binary not help final ensemble
    if FLAGS.loss_type == 'add_neu_binary':
      cid = 2
      y_ = y_[:,:,cid]
      y = (y == 2).float()
      
      bloss = self.bloss_fn(y_, y)
      loss = loss + bloss * FLAGS.other_loss_factor

    return loss
 
