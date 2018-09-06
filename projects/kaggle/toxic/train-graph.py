#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-01-13 16:32:26.966279
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './mount/temp/kaggle/toxic/model/gru.old', '')
flags.DEFINE_string('algo', 'gru_baseline', '')

import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from dataset import *
from algos.model import *

from evaluate import *

eval_fn = calc_auc

# TODO make this evaluate , infrence and melt train_flow as standard melt.apps.train 
def evaluate(ops, iterator, num_steps, num_examples, eval_fn, model_path=None, names=None, num_gpus=1, suffix='.valid', sess=None):
  ids_list = []  
  predictions_list = []
  labels_list = []

  if not sess:
    sess = melt.get_session()
  
  sess.run(iterator.initializer)

  for _ in tqdm(range(num_steps), total=num_steps, ascii=True):
    results = sess.run(ops)
    for i in range(num_gpus):
      ids, labels, predictions = results[i]
      ids = gezi.decode(ids)     
      ids_list.append(ids)   
      predictions_list.append(predictions)
      labels_list.append(labels)

  ids = np.concatenate(ids_list)[:num_examples]
  predicts = np.concatenate(predictions_list)[:num_examples]
  labels = np.concatenate(labels_list)[:num_examples]

  if model_path:
    ofile = model_path +  suffix
    with open(ofile, 'w') as out:
      if names:
        print(*names, sep=',', file=out)
      for id, label, predict in zip(ids, labels, predicts):
        print(*([id] + list(label) + list(predict)), sep=',', file=out)

  names, vals = eval_fn(labels, predicts)
  return vals, names

def inference(ops, iterator, num_steps, num_examples, model_path, names=None, num_gpus=1, suffix='.infer', sess=None):
  ids_list = []  
  predictions_list = []

  if not sess:
    sess = melt.get_session()
  
  sess.run(iterator.initializer)

  for _ in tqdm(range(num_steps), total=num_steps, ascii=True):
    results = sess.run(ops)
    for i in range(num_gpus):
      ids, predictions = results[i]
      ids = gezi.decode(ids)     
      ids_list.append(ids)   
      predictions_list.append(predictions)

  ids = np.concatenate(ids_list)[:num_examples]
  predicts = np.concatenate(predictions_list)[:num_examples]

  ofile = model_path +  suffix
  with open(ofile, 'w') as out:
    if names:
      print(*names, sep=',', file=out)
    for id, predict in zip(ids, predicts):
      print(*([id] + list(predict)), sep=',', file=out)

def main(_):
  FLAGS.emb_dim = 300
  FLAGS.learning_rate = 0.001
  
  FLAGS.optimizer = 'adam'
  #FLAGS.optimizer = 'sgd'

  FLAGS.interval_steps = 1000
  FLAGS.num_epochs = 20
  base = '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove'
  FLAGS.train_input = f'{base}/train/*record,'
  #FLAGS.train_input = f'{base}/train/1.record,'
  FLAGS.test_input = f'{base}/test/*record,'
  FLAGS.vocab = f'{base}/vocab.txt'
  FLAGS.fold = 0
  FLAGS.batch_size = 64
  FLAGS.word_embedding_file = f'{base}/glove.npy'
  FLAGS.finetune_word_embedding = False
  FLAGS.rnn_hidden_size = 100

  # # set legnth index to comment
  # FLAGS.length_index = 2
  # #FLAGS.length_index = 1
  # FLAGS.buckets = '100,400'
  # FLAGS.batch_sizes = '64,32,16'
  FLAGS.save_interval_steps = 10000

  FLAGS.batch_size = 32

  melt.apps.train.init()

  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  batch_size = melt.batch_size()
  num_gpus = melt.num_gpus()
  batch_size_per_gpu = FLAGS.batch_size

  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  logging.info('inputs', inputs)

  dataset = Dataset('train')

  num_examples = dataset.num_examples_per_epoch('train') 
  num_all_examples = num_examples
  if num_examples:
    if FLAGS.fold is not None:
      num_examples = int(num_examples * (len(inputs) / (len(inputs) + 1)))
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None

  if FLAGS.fold is not None:
    valid_inputs = [x for x in all_inputs if x not in inputs]
  else:
    valid_inputs = gezi.list_files(FLAGS.valid_input)
  
  logging.info('valid_inputs', valid_inputs)
  if FLAGS.fold is not None:
    if num_examples:
      num_valid_examples = int(num_all_examples * (1 / (len(inputs) + 1)))
      num_valid_steps_per_epoch = -(-num_valid_examples // batch_size)
    else:
      num_valid_steps_per_epoch = None
  else:
    num_valid_examples = valid_dataset.num_examples_per_epoch('valid')
    num_valid_steps_per_epoch = -(-num_valid_examples // batch_size) if num_valid_examples else None

  if valid_inputs:
    valid_dataset = Dataset('valid')
  else:
    valid_dataset = None

  test_inputs = gezi.list_files(FLAGS.test_input)
  logging.info('test_inputs', test_inputs)
  
  if test_inputs:
    test_dataset = Dataset('test')
    num_test_examples = test_dataset.num_examples_per_epoch('test')
    num_test_steps_per_epoch = -(-num_test_examples // batch_size) if num_test_examples else None
  else:
    test_dataset = None

  with tf.variable_scope('model') as scope:
    iter = dataset.make_batch(batch_size, inputs, repeat=True, initializable=False)
    batch = iter.get_next()
    x, y = melt.split_batch(batch, batch_size, num_gpus)
    model = Model()
    #loss = criterion(model, x, y, training=True)
    loss = melt.tower(lambda i: criterion(model, x[i], y[i], training=True), num_gpus)
    ops = [loss]
    scope.reuse_variables()
    
    if valid_dataset:
      valid_iter2 = valid_dataset.make_batch(batch_size, valid_inputs, repeat=True, initializable=False)
      valid_batch2 = valid_iter2.get_next()
      valid_x2, valid_y2 = melt.split_batch(valid_batch2, batch_size, num_gpus, training=False)
      valid_loss = melt.tower(lambda i: criterion(model, valid_x2[i], valid_y2[i], training=False), num_gpus, training=False)
      valid_loss = tf.reduce_mean(valid_loss)
      eval_ops = [valid_loss]

      valid_iter = valid_dataset.make_batch(batch_size, valid_inputs, repeat=True, initializable=True)
      valid_batch = valid_iter.get_next()
      valid_x, valid_y = melt.split_batch(valid_batch, batch_size, num_gpus, training=False)

      def valid_fn(i):
        valid_predict = model(valid_x[i])
        return valid_x[i].id, valid_y[i], valid_predict

      valid_ops = melt.tower(valid_fn, num_gpus, training=False)

      metric_eval_fn = lambda model_path=None: \
                                    evaluate(valid_ops, 
                                              valid_iter,
                                              num_steps=num_valid_steps_per_epoch,
                                              num_examples=num_valid_examples,
                                              eval_fn=eval_fn,
                                              names=['id'] + [x + '_y' for x in CLASSES] + CLASSES,
                                              model_path=model_path,
                                              num_gpus=num_gpus)
    else:
      eval_ops = None 
      metric_eval_fn = None

    if test_dataset:
      test_iter = test_dataset.make_batch(batch_size, test_inputs, repeat=True, initializable=True)
      test_batch = test_iter.get_next()
      test_x, test_y = melt.split_batch(test_batch, batch_size, num_gpus, training=False)

      def infer_fn(i):
        test_predict = model(test_x[i])
        return test_x[i].id, test_predict

      test_ops = melt.tower(infer_fn, num_gpus, training=False)

      inference_fn = lambda model_path=None: \
                                    inference(test_ops, 
                                              test_iter,
                                              num_steps=num_test_steps_per_epoch,
                                              num_examples=num_test_examples,
                                              names=['id'] + CLASSES,
                                              model_path=model_path,
                                              num_gpus=num_gpus)
    else:
      inference_fn = None

  melt.apps.train_flow(ops, 
                       eval_ops=eval_ops,
                       model_dir=FLAGS.model_dir,
                       metric_eval_fn=metric_eval_fn,
                       inference_fn=inference_fn,
                       num_steps_per_epoch=num_steps_per_epoch)

if __name__ == '__main__':
  tf.app.run()  
