#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-09-03 15:40:04.947138
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys 
import os

def train():
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  model = Model()

  summary = tf.contrib.summary
  summary_writer = summary.create_file_writer(FLAGS.model_dir)
  global_step = tf.train.get_or_create_global_step()

  num_epochs = 20
  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    for i, input in tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True):
      global_step.assign_add(1)
      #input = Input(*input)
      loss_value, grads = grad(model, input, calc_loss)
      optimizer.apply_gradients(zip(grads, model.variables))
      epoch_loss_avg(loss_value)  # add current batch loss
      if i % 1000 == 0:
        print(epoch, i, epoch_loss_avg.result().numpy())

    print(i, epoch_loss_avg.result().numpy())

    predicts_list = []
    classes_list = []
    for valid_input in tqdm(valid_dataset, total=num_valid_steps_per_epoch, ascii=True):
      #valid_input = Input(*valid_input)
      predicts = model(valid_input)
      predicts_list.append(predicts)
      classes_list.append(valid_input.classes)
    predicts = np.concatenate(predicts_list)
    classes = np.concatenate(classes_list)
    auc, aucs = calc_auc(predicts, classes)
    print(epoch, auc, aucs)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      summary.scalar('auc', auc)
      summary_writer.flush()

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      temp = global_step.value()
      global_step.assign(epoch)
      summary.scalar('epoch/auc', auc)
      summary_writer.flush()
      global_step.assign(temp)