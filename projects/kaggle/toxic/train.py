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

flags.DEFINE_string('model_dir', './mount/temp/kaggle/toxic/model/baseline', '')
flags.DEFINE_string('algo', 'gru_baseline', '')

#flags.DEFINE_string('input', '/home/gezi/mount/temp/toxic/tfrecords/glove/train/*record,', '')


import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from algos.model import *
from dataset import Dataset

tfe = tf.contrib.eager
tf.enable_eager_execution()

from algos.config import CLASSES
from sklearn import metrics
import pandas as pd

def calc_auc(predicts, classes):
  total_auc = 0. 
  aucs = [0.] * len(CLASSES)
  for i, class_ in enumerate(CLASSES):
    fpr, tpr, thresholds = metrics.roc_curve(classes[:, i], predicts[:, i])
    auc = metrics.auc(fpr, tpr)
    aucs[i] = auc
    total_auc += auc
  auc = total_auc / len(CLASSES) 
  return auc, aucs

from algos.config import CLASSES
def evaluate(model, dataset, num_steps_per_epoch=None):
    predicts_list = []
    classes_list = []
    for input in tqdm(dataset, total=num_steps_per_epoch, ascii=True):
      predicts = model(input)
      predicts_list.append(predicts)
      classes_list.append(input.classes)
    predicts = np.concatenate(predicts_list)
    classes = np.concatenate(classes_list)
    auc, aucs = calc_auc(predicts, classes)
    names = ['auc/avg'] + ['auc/%s' % x for x in CLASSES]
    return names, [auc] + aucs

def inference(model, dataset, model_path, num_steps_per_epoch, suffix='.infer'):
  ofile = model_path + suffix
  names = ['id'] + CLASSES
  with open(ofile, 'w') as out:
    print(*names, sep=',', file=out)
    for input in tqdm(dataset, total=num_steps_per_epoch, ascii=True):
      predicts = model(input).numpy()
      # here id is str in py3 will be bytes
      ids = gezi.decode(input.id.numpy())
      for id, predict in zip(ids, predicts):
        print(*([id] + list(predict)), sep=',', file=out)

def main(_):
  FLAGS.emb_dim = 300
  FLAGS.learning_rate = 0.001
  FLAGS.optimizer = 'adam'
  FLAGS.interval_steps = 1000
  FLAGS.num_epochs = 20
  FLAGS.train_input = '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove/train/*record,'
  FLAGS.test_input = '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove/test/*record,'

  melt.apps.train.init()

  #input_ = FLAGS.input 
  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  FLAGS.vocab = '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove/vocab.txt'

  FLAGS.fold = 0

  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  print('inputs', inputs, file=sys.stderr)

  batch_size = 64
  train_dataset_ = Dataset('train')
  train_dataset = train_dataset_.make_batch(batch_size, inputs)

  valid_inputs = [x for x in all_inputs if x not in inputs]

  print('valid_inputs', inputs, file=sys.stderr)

  valid_dataset_ = Dataset('valid')
  valid_dataset = valid_dataset_.make_batch(batch_size, valid_inputs)

  valid_dataset2_ = Dataset('valid')
  valid_dataset2 = valid_dataset2_.make_batch(batch_size, valid_inputs)

  num_examples = train_dataset_.num_examples_per_epoch('train') 

  if FLAGS.fold is not None:
    num_examples = int(num_examples * (len(inputs) / (len(inputs) + 1)))
  num_steps_per_epoch = num_examples // batch_size

  num_valid_examples = int(num_examples * (1 / (len(inputs) + 1)))
  num_valid_steps_per_epoch = num_valid_examples // batch_size

  test_dataset_ = Dataset('test')
  test_dataset = test_dataset_.make_batch(batch_size, gezi.list_files(FLAGS.test_input))
  num_test_examples = train_dataset_.num_examples_per_epoch('test')
  num_test_steps_per_epoch = num_test_examples // batch_size

  learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  optimizer = melt.get_optimizer(FLAGS.optimizer)(learning_rate)
  
  model = Model()

  summary = tf.contrib.summary
  writer = summary.create_file_writer(FLAGS.model_dir + '/epoch')
  writer_train = summary.create_file_writer(FLAGS.model_dir + '/train')
  writer_valid = summary.create_file_writer(FLAGS.model_dir + '/valid')
  global_step = tf.train.get_or_create_global_step()

  checkpoint = tf.train.Checkpoint(
        learning_rate=learning_rate, 
        model=model,
        optimizer=optimizer,
        global_step=global_step)

  ckpt_dir = FLAGS.model_dir + '/ckpt'
  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  checkpoint.restore(latest_checkpoint)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

  logging.info('Latest checkpoint:', latest_checkpoint)

  start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint else 0

  num_epochs = FLAGS.num_epochs
  for epoch in range(start_epoch, num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_valid_loss_avg = tfe.metrics.Mean()
    for i, input in tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True):
      global_step.assign_add(1)
      loss, grads = melt.eager.grad(model, input, calc_loss)
      optimizer.apply_gradients(zip(grads, model.variables))
      epoch_loss_avg(loss)  # add current batch loss
      if i % FLAGS.interval_steps == 0:
        valid_loss = calc_loss(model, next(iter(valid_dataset2)))
        epoch_valid_loss_avg(valid_loss)

        logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                     'step:%d' % global_step.numpy(), 
                     'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                     'valid_loss::%.4f' % epoch_valid_loss_avg.result().numpy())
        
        with writer_train.as_default(), summary.always_record_summaries():
          summary.scalar('step/loss', epoch_loss_avg.result().numpy())
          writer_train.flush()
        with writer_valid.as_default(), summary.always_record_summaries():
          summary.scalar('step/loss', epoch_valid_loss_avg.result().numpy())
          writer_valid.flush()

    logging.info('epoch:%d/%d' % (epoch + 1, num_epochs), 
                 'step:%d' % global_step.numpy(), 
                 'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                 'valid_loss::%.4f' % epoch_valid_loss_avg.result().numpy())

    names, vals = evaluate(model, valid_dataset, num_valid_steps_per_epoch)
    logging.info2('epoch:%d/%d' % (epoch + 1, num_epochs), 
                  'step:%d' % global_step.numpy(), 
                  list(zip(names, vals)))

    with writer.as_default(), summary.always_record_summaries():
      temp = global_step.value()
      global_step.assign(epoch + 1)
      summary.scalar('epoch/train/loss', epoch_loss_avg.result())
      for name, val in zip(names, vals):
        summary.scalar(f'epoch/valid/{name}', val)
      writer.flush()
      global_step.assign(temp)

    checkpoint.save(checkpoint_prefix)
    inference(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)

    

if __name__ == '__main__':
  tf.app.run()  
