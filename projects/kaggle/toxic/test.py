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
from collections import defaultdict

import numpy as np

from gezi import Timer
import gezi
import melt 
import algos_factory
import string
import re
from collections import Counter 
import json

from sklearn import metrics
import pandas as pd

from evaluator import write_evaluate, write_inference

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '../../../mount/temp/toxic/model/gru.baseline/', '')
flags.DEFINE_integer('batch_size_', 64, '')
flags.DEFINE_string('algo', 'gru_baseline', '')
#flags.DEFINE_integer('fold', None, '')

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

ids_list = []
classes_list = []
comment_str_list = []
comment_tokens_str_list = []
predicts_list = []

def run_once(sess, step, predicts, result):
  id_, classes_, predicts_, comment_str_, comment_tokens_str_ = sess.run([result.id, result.classes, predicts, result.comment_str, result.comment_tokens_str])
  id_ = gezi.decode(id_)
  if step % 10 == 0:
    print(step, id_[0], classes_[0], list(map('{0:03f}'.format, predicts_[0])))

  ids_list.append(id_)
  classes_list.append(classes_)
  predicts_list.append(predicts_)
  
  comment_str_list.append(comment_str_)
  comment_tokens_str_list.append(comment_tokens_str_)

from melt.flow import tf_flow
from input import Result
import input as tfrecord_input 
def read_records(files):
  inputs, decode = tfrecord_input.get_decodes()
  ops = inputs(
    files,
    decode_fn=decode,
    batch_size=FLAGS.batch_size,
    bucket_boundaries=FLAGS.buckets,
    bucket_batch_sizes=FLAGS.batch_sizes,
    length_index=2,
    num_epochs=1, 
    num_threads=1,
    seed=1234,
    allow_smaller_final_batch=True,
    shuffle_batch=False,
    )

  return ops

  
def run(validator):
  if 'TEST' in os.environ and os.environ['TEST'] == '1':
    FLAGS.fold = None
  if FLAGS.fold is not None:
    input = FLAGS.train_input
    inputs = gezi.list_files(input)
    inputs = [x for x in inputs if x.endswith('%d.record' % FLAGS.fold)]
  else:
    input = FLAGS.train_input.replace('train', 'test').split(',')[0]
    inputs = gezi.list_files(input)
  
  inputs.sort()
  print('inputs', inputs)
  ops = read_records(inputs)
  
  result = Result(*ops)
  validator.build(result)
  #predicts = validator.logits
  predicts = validator.predictions

  sess = tf.Session()
  timer = Timer('sess run test')
  num_steps = tf_flow(lambda sess, step: run_once(sess, step, predicts, result), model_dir=FLAGS.model_dir, sess=sess)
  
  print('num_steps:', num_steps)
  timer.print_elapsed() 

  ids =  np.concatenate(ids_list)
  print('ids.shape', ids.shape)
  classes = np.concatenate(classes_list)
  predicts = np.concatenate(predicts_list)

  comment_strs = np.concatenate(comment_str_list)
  comment_tokens_strs = np.concatenate(comment_tokens_str_list)

  model_path = melt.get_model_path(FLAGS.model_dir)
  if FLAGS.fold is not None:
    timer = gezi.Timer('calc auc')
    total_auc = 0.
    for i, class_ in enumerate(CLASSES):
      fpr, tpr, thresholds = metrics.roc_curve(classes[:, i], predicts[:, i])
      auc = metrics.auc(fpr, tpr)
      print(class_, auc)
      total_auc += auc
    print('AVG', total_auc / len(CLASSES))
    timer.print_elapsed()
    write_evaluate(ids, predicts, classes, comment_strs, comment_tokens_strs, model_path)
  else:
    write_inference(ids, predicts, comment_strs, comment_tokens_strs, model_path)

def main(_):
  pd.set_option('display.max_colwidth', -1)
  FLAGS.num_epochs = 1  
  global_scope = FLAGS.algo 
  with tf.variable_scope('%s/main' % global_scope) as scope:
    validator = algos_factory.gen_validator(FLAGS.algo, reuse=False)
    run(validator)  

if __name__ == '__main__':
  tf.app.run()
