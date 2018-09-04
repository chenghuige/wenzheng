#!/usr/bin/env python
# ==============================================================================
#          \file   embsim.py
#        \author   chenghuige  
#          \date   2017-08-09 15:17:18.499275
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf 
import melt

def zero_first_row(emb):
    #return tf.concat([tf.zeros([1, emb.get_shape()[1]]), tf.ones([emb.get_shape()[0] - 1, emb.get_shape()[1]])], 0)   
    shape = tf.shape(emb)
    return tf.concat([tf.zeros([1, shape[1]]), tf.ones([shape[0] - 1, shape[1]])], 0)   

class EmbeddingSim:
  def __init__(self, emb=None, fixed_emb=None, 
              name=None, fixed_name=None, 
              model_dir=None, model_name=None, 
              sess=None):
    self._sess = sess or tf.InteractiveSession()
 
    if os.path.isdir(emb):
      model_dir = emb 

    if model_dir is None:
      if isinstance(emb, str):
        emb = np.load(emb)
        emb = melt.load_constant(emb, name=name)
      if isinstance(fixed_emb, str):
        #fixed_emb is corpus embeddings, all sumed and normed already
        fixed_emb = melt.load(fixed_emb, name=fixed_emb)
    else:
      model_path = melt.get_model_path(model_dir, model_name)
      emb = tf.Variable(0., name=name, validate_shape=False)
      #emb = tf.Variable(0., name=name)
      #like word2vec the name is 'w_in'
      embedding_saver = tf.train.Saver({name: emb})
      embedding_saver.restore(self._sess, model_path)
 
    #assume 0 index not used, 0 for PAD
    mask = zero_first_row(emb)
    emb = tf.multiply(emb, mask)

    self._emb = emb 
    self._fixed_emb = fixed_emb
    self._normed_emb = None

  def sum_emb(self, ids):
    """
    ids [batch_size, max_words_per_sentence]
    """
    return melt.batch_embedding_lookup_sum(self._emb, ids)

  def to_feature(self, ids):
    return tf.nn.l2_normalize(self.sum_emb(ids), -1)

  def sim(self, left_ids, right_ids):
    """
    [x, ..] [y, ..] -> [x, y]
    """
    left_feature = self.to_feature(left_ids)
    right_feature = self.to_feature(right_ids)
    return melt.dot(left_feature, right_feature) 

  def top_sim(self, left_ids, right_ids, topn=50, sorted=True):
    """
    top sim text
    """
    score = self.sim(left_ids, right_ids)
    return tf.nn.top_k(score, topn, sorted=sorted)

  def all_score(self, ids):
    feature = self.to_feature(ids)
    if self._normed_emb is None:
      self._normed_emb = tf.nn.l2_normalize(self._emb, -1) 
    return melt.dot(feature, self._normed_emb)
  
  def nearby(self, ids, topn=50, sorted=True):
    """
    ids [x, ..] -> [x, vocab_size]
    nearby word
    """
    score = self.all_score(ids)
    return tf.nn.top_k(score, topn, sorted=sorted)

  def fixed_sim(self, left_ids, right_ids):
    left_feature = tf.nn.embedding_lookup(self._fixed_emb, left_ids)
    right_feature = tf.nn.embedding_lookup(self._fixed_emb, right_ids)
    return melt.dot(left_feature, right_feature)

  def fixed_right_sim(self, left_ids):
    left_feature = self.to_feature(left_ids)
    right_feature = tf.nn.embedding_lookup(self._fixed_emb, right_ids)
    return melt.dot(left_feature, right_feature)

  def fixed_all_score(self, ids):
    feature = self.embedding_lookup(self._fixed_emb, ids)
    return melt.dot(feature, self._fixed_emb)

  def fixed_nearyby(self, ids, topn=50, sorted=True):
    score = self.fixed_all_score(ids)
    return tf.nn.top_k(score, topn, sorted=sorted)

  def fixed_right_all_score(self, ids):
    feature = self.to_feature(ids)
    return melt.dot(feature, self._fixed_emb)

  def fixed_right_nearyby(self, ids, topn=50, sorted=True):
    score = self.fixed_right_all_score(ids)
    return tf.nn.top_k(score, topn, sorted=sorted)
    
  def dum_fixed_emb(self, ids, ofile):
    feature = self.to_feature(ids)
    np.save(ofile, feature)

  def dum_emb(self, ofile):
    np.save(ofile, self._emb)
    
