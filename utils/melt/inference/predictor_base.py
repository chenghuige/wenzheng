#!/usr/bin/env python
# ==============================================================================
#          \file   predictor_base.py
#        \author   chenghuige  
#          \date   2016-08-17 23:57:11.987515
#   \Description  
# ==============================================================================

"""
This is used for train predict, predictor building graph not read from meta
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt, gezi

import numpy as np 

def get_tensor_from_key(key, index=-1):
  if isinstance(key, str):
    try:
      return tf.get_collection(key)[index]
    except Exception:
      print('Warning:', key, ' not find in graph')
      return tf.no_op()
  else:
    return key

class PredictorBase(object):
  #TODO maybe like predictor.py need graph?
  def __init__(self, sess=None):
    super(PredictorBase, self).__init__()
    if sess is None:
      self.sess = melt.get_session()
    else:
      self.sess = sess

  def load(self, model_dir, var_list=None, model_name=None, sess = None):
    """
    only load varaibels from checkpoint file, you need to 
    create the graph before calling load
    """
    if sess is not None:
      self.sess = sess
    self.model_path = melt.get_model_path(model_dir, model_name)
    timer = gezi.Timer('load model ok %s' % self.model_path)
    saver = melt.restore_from_path(self.sess, self.model_path, var_list)
    timer.print()
    return self.sess

  def restore_from_graph(self):
    pass

  def restore(self, model_dir, model_name=None, sess=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    if sess is not None:
      self.sess = sess
    self.model_path = model_path = melt.get_model_path(model_dir, model_name)
    timer = gezi.Timer('restore meta grpah and model ok %s' % model_path)
    meta_filename = '%s.meta'%model_path
    saver = tf.train.import_meta_graph(meta_filename)
    self.restore_from_graph()
    saver.restore(self.sess, model_path)
    #---TODO not work remove can run but hang  FIXME add predictor + exact_predictor during train will face
    #@gauravsindhwani , can you still run the code successfully after you remove these two collections since they are actually part of the graph. 
    #I try your way but find the program is stuck after restoring."
    #https://github.com/tensorflow/tensorflow/issues/9747
    #tf.get_default_graph().clear_collection("queue_runners")
    #tf.get_default_graph().clear_collection("local_variables")
    #--for num_epochs not 0
    #self.sess.run(tf.local_variables_initializer())
    timer.print()
    return self.sess

  def run(self, key, feed_dict=None):
    return self.sess.run(key, feed_dict)

  def inference(self, key, feed_dict=None, index=-1):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(get_tensor_from_key(key, index), feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      keys = [get_tensor_from_key(key, index) for key,index in zip(keys, indexes)]
      return self.sess.run(keys, feed_dict=feed_dict)

  def elementwise_predict(self, ltexts, rtexts):
    scores = []
    if len(rtexts) >= len(ltexts):
      for ltext in ltexts:
        stacked_ltexts = np.array([ltext] * len(rtexts))
        score = self.predict(stacked_ltexts, rtexts)
        score = np.squeeze(score) 
        scores.append(score)
    else:
      for rtext in rtexts:
        stacked_rtexts = np.array([rtext] * len(ltexts))
        score = self.predict(ltexts, stacked_rtexts)
        score = np.squeeze(score) 
        scores.append(score)
    return np.array(scores)  
