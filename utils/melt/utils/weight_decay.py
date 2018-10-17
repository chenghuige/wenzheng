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

tfe = tf.contrib.eager

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
               initial_learning_rate=None,
               decay_start_epoch=0,
               sess=None):
    import melt.utils.logging as logging
    if not tf.executing_eagerly():
      self.sess = sess or melt.get_session()
    if isinstance(weight_op, str):
      try:
        # by default melt.apps.train will generate weight op Var named 'learning_rate_weight' TODO may be hold it just here
        # so currently graph model will go here
        self.weight_op = tf.get_collection(weight_op)[-1]
        self.name = weight_op
      except Exception:
        raise 'TODO..'
        # print('-------------------------Weight Decay change!')
        # so currently eager mode will go here
        #learning_rate_weight = tf.get_variable('learning_rate_weight', initializer= tf.ones(shape=(), dtype=tf.float32))
        #learning_rate_weight = tf.Variable(tf.ones(shape=(), dtype=tf.float32), name='learning_rate_weight')
        # TODO tfe.Var should only be used in keras.Model init ? notice eager mode can not use tf.Variable
        # learning_rate_weight = tfe.Variable(tf.ones(shape=(), dtype=tf.float32), name='learning_rate_weight')
        # tf.add_to_collection('learning_rate_weight', learning_rate_weight)
        # self.weight_op = learning_rate_weight
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

    self.decay_start_epoch = decay_start_epoch

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
      # epoch is set during training loop
      epoch = melt.epoch()
      logging.info('patience:', self.patience)
      if epoch < self.decay_start_epoch:
        return
      if self.patience >= self.max_patience:
        self.count += 1
        self.patience = 0
        self.score = score
        decay = self.decay
        pre_weight = weight
        #weight *= decay
        weight = weight * decay
        
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

            #tf.get_collection('learning_rate')[-1] *= decay
            tf.get_collection('learning_rate')[-1].assign(tf.get_collection('learning_rate')[-1] * decay)

    return weight


class WeightsDecay(object):
  def __init__(self, 
               weights_op='learning_rate_weights', 
               patience=3, 
               decay=0.8, 
               cmp=None,
               names=None,
               num_weights=None, 
               min_weight=None,
               min_learning_rate=None,
               initial_learning_rate=None,
               initial_score=None,
               decay_start_epoch=0,
               sess=None):
    import melt.utils.logging as logging
    if not tf.executing_eagerly():
      self.sess = sess or melt.get_session()

    if num_weights is None:
      assert names
      num_weights = len(names)

    logging.info('decay:', decay, 'cmp:', cmp)
    assert cmp == 'less' or cmp == 'greater'

    if cmp == 'less':
      self.cmp = lambda x, y: x < y
      self.scores = np.ones([num_weights]) * 1e10
    elif cmp == 'greater':
      self.cmp = lambda x, y: x > y  
      self.scores = np.ones([num_weights]) * -1e10
    else:
      # TODO...
      self.cmp = cmp
      assert initial_score
      self.scores = [initial_score] * num_weights

    #self.scores = None

    self.max_patience = patience
    self.decay = decay

    # TODO patience also varaible so can save and restore ?
    self.patience = [0] * num_weights
    self.count = [0] * num_weights
    self.names = names or list(map(str, range(num_weights)))

    self.min_weight = min_weight

    self.decay_start_epoch = decay_start_epoch

    if not self.min_weight:
      self.min_weight = min_learning_rate / (initial_learning_rate or FLAGS.learning_rate)

    if isinstance(weights_op, str):
      try:
        self.weights_op = tf.get_collection(weights_op)[-1]
      except Exception:
        #self.weights_op = tf.get_variable('lr_ratios', initializer=tf.ones([num_classes], dtype=tf.float32))
        #tf.add_to_collection('lr_ratios', lr_ratios)
        raise 'TODO..'
    else:
      self.weights_op = weights_op

  def add(self, scores):
    import melt.utils.logging as logging
    scores = np.array(scores)

    #print(scores.shape, self.scores.shape, len(self.names))
    logging.info('diff:', list(zip(self.names, scores - self.scores)))

    if not tf.executing_eagerly():
      weights = self.sess.run(self.weights_op)
      weights_ = weights
    else:
      weights = self.weights_op
      weights_ = weights.numpy()

    if (not self.cmp) and self.scores:
      if scores[0] > self.scores[0]:
        self.cmp = lambda x, y: x > y  
      else:
        self.cmp = lambda x, y: x < y
      logging.info('decay cmp:', self.cmp)
      
        # epoch is set during training loop
    epoch = melt.epoch()

    for i, score in enumerate(scores):
      if self.scores is None or self.cmp(score, self.scores[i]):
        self.scores[i] = score 
        self.patience[i] = 0
      else:
        self.patience[i] += 1        
        
        logging.info('patience_%s %d' % (self.names[i], self.patience[i]))
        if epoch < self.decay_start_epoch:
          continue

        if self.patience[i] >= self.max_patience:
          self.count[i] += 1
          self.patience[i] = 0
          self.scores[i] = score
          
          decay = self.decay if not isinstance(self.decay, (list, tuple)) else self.decay[i]

          weights_[i] *= decay

          if not self.min_weight:
            if weights_[i] < self.min_weight:
              weights_[i] = self.min_weight

          #logging.info('!%s decay count:%d decay ratio:%f lr ratios now:%f' % (self.names[i], self.count[i], self.decay, weights[i]))
          if not tf.executing_eagerly():
            self.sess.run(tf.assign(self.weights_op, tf.constant(weights_, dtype=tf.float32)))
          else:
            self.weights_op.assign(weights_)

    return weights_
          