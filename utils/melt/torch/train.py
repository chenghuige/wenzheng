#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-09-26 11:31:25.070813
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import torch
except Exception:
  pass

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

tfe = tf.contrib.eager

import sys 
import os
from tqdm import tqdm 
import numpy as np
import inspect
import traceback

import gezi
import melt
logging = melt.logging

def torch_(x):
  for dim in x.shape:
    if dim == 0:
      return x

  #x = x.numpy()
  # if x.dtype == np.int64 or x.dtype == np.int32:
  #   x = torch.LongTensor(x)
    
  #   if torch.cuda.is_available():
  #     x = x.cuda()   
  #     x.requires_grad = False 
  #   #x = torch.cuda.LongTensor(x)
  # elif x.dtype == np.float32 or x.dtype == np.float64:
  #   x = torch.FloatTensor(x)
    
  #   if torch.cuda.is_available():
  #     x = x.cuda() 
  #     x.requires_grad = False   
  #   #x = torch.cuda.FloatTensor(x)
  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
      x = x.cuda()

  return x


def to_torch(x, y=None):
  if y is not None:
    y = torch_(y)

  for key in x:
    x[key] = torch_(x[key])
  if y is None:
    return x
  else:
    return x, y

  
def train(Dataset, 
          model, 
          loss_fn, 
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          write_valid=True,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer',
          write_streaming=False,
          sep=','):
  if FLAGS.torch:
    if torch.cuda.is_available():
      model.cuda()
  
  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  batch_size = FLAGS.batch_size

  num_gpus = melt.num_gpus()
  if num_gpus > 1:
    assert False, 'Eager mode train currently not support for num gpus > 1'

  #batch_size_ = batch_size if not FLAGS.batch_sizes else int(FLAGS.batch_sizes.split(',')[-1])
  batch_size_ = batch_size

  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  logging.info('inputs', inputs)

  dataset = Dataset('train')
  num_examples = dataset.num_examples_per_epoch('train') 
  num_all_examples = num_examples

  # if FLAGS.fold is not None:
  #   valid_inputs = [x for x in all_inputs if x not in inputs]
  # else:
  #   valid_inputs = gezi.list_files(FLAGS.valid_input)
  
  # logging.info('valid_inputs', valid_inputs)

  # if valid_inputs:
  #   valid_dataset_ = Dataset('valid')
  #   valid_dataset = valid_dataset_.make_batch(batch_size_, valid_inputs)
  #   valid_dataset2 = valid_dataset_.make_batch(batch_size_, valid_inputs, repeat=True)
  # else:
  #   valid_datsset = None
  #   valid_dataset2 = None

  if num_examples:
    if FLAGS.fold is not None:
      num_examples = int(num_examples * (len(inputs) / (len(inputs) + 1)))
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None

  # if FLAGS.fold is not None:
  #   if num_examples:
  #     num_valid_examples = int(num_all_examples * (1 / (len(inputs) + 1)))
  #     num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_)
  #   else:
  #     num_valid_steps_per_epoch = None
  # else:
  #   num_valid_examples = valid_dataset_.num_examples_per_epoch('valid')
  #   num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_) if num_valid_examples else None

  # test_inputs = gezi.list_files(FLAGS.test_input)
  # logging.info('test_inputs', test_inputs)
  
  # if test_inputs:
  #   test_dataset_ = Dataset('test')
  #   test_dataset = test_dataset_.make_batch(batch_size_, test_inputs) 
  #   num_test_examples = test_dataset_.num_examples_per_epoch('test')
  #   num_test_steps_per_epoch = -(-num_test_examples // batch_size_) if num_test_examples else None
  # else:
  #   test_dataset = None
  
  summary = tf.contrib.summary
  # writer = summary.create_file_writer(FLAGS.model_dir + '/epoch')
  # writer_train = summary.create_file_writer(FLAGS.model_dir + '/train')
  # writer_valid = summary.create_file_writer(FLAGS.model_dir + '/valid')
  writer = summary.create_file_writer(FLAGS.model_dir)
  writer_train = summary.create_file_writer(FLAGS.model_dir)
  writer_valid = summary.create_file_writer(FLAGS.model_dir)
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  tf.add_to_collection('learning_rate', learning_rate)

  learning_rate_weight = tf.get_collection('learning_rate_weight')[-1]
  try:
    learning_rate_weights = tf.get_collection('learning_rate_weights')[-1]
  except Exception:
    learning_rate_weights = None

  ckpt_dir = FLAGS.model_dir + '/ckpt'

  #TODO FIXME now I just changed tf code so to not by default save only latest 5
  # refer to https://github.com/tensorflow/tensorflow/issues/22036
    # manager = tf.contrib.checkpoint.CheckpointManager(
  #     checkpoint, directory=ckpt_dir, max_to_keep=5)
  # latest_checkpoint = manager.latest_checkpoint

  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  logging.info('Latest checkpoint:', latest_checkpoint)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

  if not FLAGS.torch:
    optimizer = melt.get_optimizer(FLAGS.optimizer)(learning_rate)
    
    # TODO...
    if  learning_rate_weights is None:
      checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            model=model,
            optimizer=optimizer,
            global_step=global_step)
    else:
      checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            learning_rate_weights=learning_rate_weights,
            model=model,
            optimizer=optimizer,
            global_step=global_step)
      
    if os.path.exists(FLAGS.model_dir + '.index'):
      latest_checkpoint = FLAGS.model_dir   

    checkpoint.restore(latest_checkpoint)

    start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint else 0
  else:
    # TODO torch with learning rate adjust
    optimizer = torch.optim.Adamax(model.parameters(), lr=FLAGS.learning_rate)

    if latest_checkpoint:
      checkpoint = torch.load(latest_checkpoint + '.pyt')
      start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      model.eval()
    else:
      start_epoch = 0

    if learning_rate_weights is None:
      checkpoint = tf.train.Checkpoint(
          learning_rate=learning_rate, 
          learning_rate_weight=learning_rate_weight,
          global_step=global_step)
    else:
      checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            learning_rate_weights=learning_rate_weights,
            global_step=global_step)

  #model.load_weights(os.path.join(ckpt_dir, 'ckpt-1'))
  #model.save('./weight3.hd5')

  # TODO currently not support 0.1 epoch.. like this
  num_epochs = FLAGS.num_epochs
  
 
  class PytObj(object):
    def __init__(self, x):
      self.x = x
    def numpy(self):
      return self.x

  class PytMean(object):
    def __init__(self):
      self._val = 0. 
      self.count = 0

      self.is_call = True

    def clear(self):
      self._val = 0
      self.count = 0

    def __call__(self, val):
      if not self.is_call:
        self.clear()
        self.is_call = True
      self._val += val.item()
      self.count += 1

    def result(self):
      if self.is_call:
        self.is_call = False
      if not self.count:
        val = 0
      else:
        val = self._val / self.count
      # TODO just for compact with tf ..
      return PytObj(val)
      
  # TODO consider multiple gpu for torch 

  iter = dataset.make_batch(batch_size, inputs, repeat=False, initializable=False)
  batch = iter.get_next()
  #x, y = melt.split_batch(batch, batch_size, num_gpus)
  x_, y_ = batch
  
  Mean =  tfe.metrics.Mean if not FLAGS.torch else PytMean
  epoch_loss_avg = Mean()
  epoch_valid_loss_avg = Mean()

  sess = melt.get_session(device_count={'GPU': 0})
  global_step = 0
  for epoch in range(start_epoch, num_epochs):
    melt.set_global('epoch', '%.4f' % (epoch))
    sess.run(iter.initializer)

    model.train()

    #..... still OOM... FIXME TODO
    try:
      for _ in tqdm(range(num_steps_per_epoch), total=num_steps_per_epoch, ascii=True):
        x, y = sess.run([x_, y_])
        x, y = to_torch(x, y)
        
        optimizer.zero_grad()
        loss = loss_fn(model, x, y)
        loss.backward()
        optimizer.step()

        epoch_loss_avg(loss) 

        if global_step % FLAGS.interval_steps == 0:
          print(global_step, epoch_loss_avg.result().numpy())

        global_step += 1
    except tf.errors.OutOfRangeError:
      print('epoch:%d loss:%f' % (epoch, epoch_loss_avg.result().numpy()))


