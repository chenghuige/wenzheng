#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_bool('draw_pr', True, '')
flags.DEFINE_float('precision_thre', 0.8, '')

from tensorboard import summary as summary_lib

from collections import defaultdict

import numpy as np

from gezi import Timer
import gezi
import melt 
logging = melt.logging
import pandas as pd
pd.set_option('display.max_colwidth', -1)

import deepiu

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def write(ids, predicts, model_path, labels=None, images=None, suffix='valid_info'):
  tb = pd.DataFrame()
  tb['id'] = ids 
  
  predicts = [classes[x] for x in predicts]

  if labels is not None:
    tb['predict'] = predicts 
    labels = [classes[x] for x in labels]
    tb['label'] = labels

    ofile = model_path + '.valid_error.png'
    from deepiu.visualize.classify import plot_example_errors
    plot_example_errors(images, 
                        labels, 
                        predicts,
                        smooth=True, 
                        #smooth=False, 
                        #class_names=classes,
                        image_names=ids, 
                        out_file=ofile,
                        max_show=9)
  else:
    tb['label'] = predicts

  ofile = model_path + '.%s' % suffix
  tb.to_csv(ofile, index=False)

def evaluate(eval_ops, iterator, model_path=None, sess=None):
  if model_path:
    ids_list = []
    
  predictions_list = []
  labels_list = []
  losses = []

  images_list = []

  id_, loss_, predicts_, labels_, images_ = eval_ops
  if not sess:
    sess = melt.get_session()
  
  # for prcurve
  sess.run(iterator.initializer)

  try:
    while True:
      ids, loss, predictions, labels, images = sess.run(eval_ops)
      images = images.astype(np.uint8)
      losses.append(loss)
      predictions_list.append(predictions)
      labels_list.append(labels)

      images_list.append(images)

      if model_path:
        ids_list.append(ids)
  except tf.errors.OutOfRangeError:
    loss = np.mean(losses)
    predicts = np.concatenate(predictions_list)
    labels = np.concatenate(labels_list)

    images = np.concatenate(images_list)

    acc = np.mean(np.equal(predicts, labels))

    results = [loss, acc] 
    names = ['metric/valid/loss/avg', 'metric/valid/acc'] 
    
    if model_path:
      write(np.concatenate(ids_list), 
            predicts,
            model_path,
            labels,
            images,
            suffix='valid_info'
            )

    return results, names 


def inference(ops, iterator, model_path=None, sess=None):
  assert model_path
  if model_path:
    ids_list = []
    
  predictions_list = []

  id_, predicts_ = ops
  if not sess:
    sess = melt.get_session()
  
  # for prcurve
  sess.run(iterator.initializer)

  try:
    while True:
      ids, predictions = sess.run(ops)
      
      predictions_list.append(predictions)

      if model_path:
        ids_list.append(ids)
  except tf.errors.OutOfRangeError:
    predicts = np.concatenate(predictions_list)
    
    if model_path:
      write(np.concatenate(ids_list), 
            predicts,
            model_path,
            labels=None,
            suffix='infer_info'
            )

