#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-02-24 11:39:22.566492
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

#import sys, os
import melt
#logging = melt.utils.logging
#import logging
#print(dir(melt))

# FIXME TODO Why... not found melt.utils ??!
#import melt.utils.logging as logging

#import gezi

import numpy as np

class WeightDecay(object):
  def __init__(self, 
               weight_op='learning_rate_weight', 
               patience=3, 
               decay=0.8,  
               cmp=None,
               min_weight=None,
               min_learning_rate=None,
               initial_learning_rate = None,
               sess=None):
    import melt.utils.logging as logging
    if not tf.executing_eagerly():
      self.sess = sess or melt.get_session()
    if isinstance(weight_op, str):
      self.weight_op = tf.get_collection(weight_op)[-1]
      self.name = weight_op
    else:
      self.weight_op = weight_op
      self.name = 'weight'

    if cmp == 'less':
      self.cmp = lambda x, y: x < y
    elif cmp== 'greater':
      self.cmp = lambda x, y: x > y  
    else:
      self.cmp = cmp
    self.score = None

    self.max_patience = patience
    self.decay = decay
    self.patience = 0
    self.count = 0
    self.min_weight = min_weight

    if not self.min_weight:
      self.min_weight = min_learning_rate / (initial_learning_rate or FLAGS.learning_rate)

    # This is done in melt.flow
    # weight = self.sess.run(self.weight_op)
    # if 'learning_rate' in self.name:
    #   melt.set_learning_rate(tf.constant(weight, dtype=tf.float32), self.sess)

  def add(self, score):
    import melt.utils.logging as logging

    if not tf.executing_eagerly():
      weight = self.sess.run(self.weight_op)
    else:
      weight = self.weight_op
    #print(weight, score, self.score, self.patience)
    
    if (not self.cmp) and self.score:
      if score > self.score:
        self.cmp = lambda x, y: x > y  
      else:
        self.cmp = lambda x, y: x < y
      logging.info('decay cmp:', self.cmp)

    if not self.score or self.cmp(score, self.score):
      self.score = score 
      self.patience = 0
    else:
      self.patience += 1
      # TODO why not print ..
      logging.info('patience', self.patience)
      if self.patience >= self.max_patience:
        self.count += 1
        self.patience = 0
        self.score = score
        decay = self.decay
        pre_weight = weight
        weight *= decay
        
        # decay
        if self.min_weight and weight < self.min_weight:
          weight = self.min_weight
          decay = weight / pre_weight
          if decay >  1.:
            decay = 1.

        logging.info('!decay count:', self.count, self.name, 'now:', weight)
        if not tf.executing_eagerly():
          self.sess.run(tf.assign(self.weight_op, tf.constant(weight, dtype=tf.float32)))
        else:
          self.weight_op = weight
        
        if 'learning_rate' in self.name:
          if not tf.executing_eagerly():
            melt.multiply_learning_rate(tf.constant(decay, dtype=tf.float32), self.sess)
          else:
            # TODO need to test eager mode
            #learning_rate =  tf.get_collection('learning_rate')[-1]
            #if learning_rate * decay > self.min_learning_rate:
            tf.get_collection('learning_rate')[-1] *= decay
    return weight


class WeightsDecay(object):
  def __init__(self, 
               weights_op='learning_rate_weight', 
               patience=3, 
               decay=0.8, 
               cmp=None,
               names=None,
               num_weights=None, 
               min_weights=None,
               sess=None):
    import melt.utils.logging as logging
    if not tf.executing_eagerly():
      self.sess = sess or melt.get_session()
    if num_weights is None:
      assert names
      num_weights = len(names)

    if isinstance(weights_op, str):
      self.weights_op = tf.get_collection(weights_op)[-1]
    else:
      self.weights_op = weights_op

    logging.info('decay:', decay, 'cmp:', cmp)
    assert cmp == 'less' or cmp == 'greater'

    if cmp == 'less':
      self.cmp = lambda x, y: x < y
    elif cmp == 'greater':
      self.cmp = lambda x, y: x > y  
    else:
      self.cmp = cmp

    self.scores = None

    self.max_patience = patience
    self.decay = decay

    # TODO patience also varaible so can save and restore ?
    self.patience = [0] * num_weights
    self.count = [0] * num_weights
    self.names = names or list(map(str, range(num_weights)))

    self.min_weights = min_weights

  def add(self, scores):
    import melt.utils.logging as logging
    scores = np.array(scores)
    logging.info('diff:', list(zip(self.names, scores - self.scores)))

    if not tf.executing_eagerly():
      weights = self.sess.run(self.weights_op)
    else:
      weights = self.weights_op

    if (not self.cmp) and self.scores:
      if scores[0] > self.scores[0]:
        self.cmp = lambda x, y: x > y  
      else:
        self.cmp = lambda x, y: x < y
      logging.info('decay cmp:', self.cmp)

    for i, score in enumerate(scores):
      if not self.scores or self.cmp(score, self.scores[i]):
        self.scores[i] = score 
        self.patience[i] = 0
      else:
        self.patience[i] += 1
        logging.info('patience_%s %d' % (self.names[i], self.patience[i]))
        if self.patience[i] >= self.max_patience:
          self.count[i] += 1
          self.patience[i] = 0
          self.scores[i] = score
          weights[i] *= self.decay if not isinstance(self.decay, (list, tuple)) else self.decay[i]
          
          if not self.min_weights:
            if weights[i] < self.min_weights[i]:
              weights[i] = self.min_weights[i]

          logging.info('!%s decay count:%d decay ratio:%f lr ratios now:%f' % (self.names[i], self.count[i], self.decay, weights[i]))
          if not tf.executing_eagerly():
            self.sess.run(tf.assign(self.weights_op, tf.constant(weights, dtype=tf.float32)))
          else:
            self.weights_op = weights

    return weights
          