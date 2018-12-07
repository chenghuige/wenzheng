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

flags.DEFINE_string('model_dir', './mount/temp/cifar10/model/resnet', '')
flags.DEFINE_string('algo', 'resnet', '')

import numpy as np

import melt 
logging = melt.logging
import gezi
import traceback

#import evaluator

import cifar10
import cifar10_model
import cifar10_utils

import evaluator

eval_names = None
label_batch = None

def tower_loss(model, feature, label):
  logits = model.forward_pass(feature, input_data_format='channels_last')
  weight_decay = 0.0002

  loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=label)
  loss = tf.reduce_mean(loss)

  model_params = tf.trainable_variables()
  loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  return loss

# import tfmpl
# @tfmpl.figure_tensor
# def draw_confusion_matrix(matrix):
#     '''Draw confusion matrix for MNIST.'''
#     fig = tfmpl.create_figure(figsize=(7,7))
#     ax = fig.add_subplot(111)
#     ax.set_title('Confusion matrix for MNIST classification')
    
#     tfmpl.plots.confusion_matrix.draw(
#         ax, matrix,
#         axis_labels=['Digit ' + str(x) for x in range(10)],
#         normalize=True
#     )

#     return fig

def main(_):
  num_train_examples = 45000
  melt.apps.train.init()

  batch_size = melt.batch_size()
  num_gpus = melt.num_gpus()

  batch_size_per_gpu = FLAGS.batch_size

  # batch size not changed but FLAGS.batch_size will change to batch_size / num_gpus
  #print('--------------batch_size, FLAGS.batch_size, num_steps_per_epoch', batch_size, FLAGS.batch_size, num_train_examples // batch_size)

  global_scope = FLAGS.algo 
  with tf.variable_scope(global_scope) as global_scope:
    data_format = 'channels_first'
    num_layers = 44
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-05
    data_dir = './mount/data/cifar10/' 
    with tf.variable_scope('main') as scope:
      model = cifar10_model.ResNetCifar10(
        num_layers,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=True,
        data_format=data_format)

      dataset = cifar10.Cifar10DataSet(data_dir, subset='train', use_distortion=True)

      # iterator = dataset.make_batch(batch_size_per_gpu)
      # def loss_function():
      #   global label_batch
      #   _, image_batch, label_batch = iterator.get_next()
      #   return tower_loss(model, image_batch, label_batch)

      # this is faster then above method
      iterator = dataset.make_batch(batch_size)
      batch = iterator.get_next()
      _, image_batches, label_batches = melt.split_batch(batch, batch_size, num_gpus)
      def loss_function(i):
        return tower_loss(model, image_batches[i], label_batches[i])
      label_batch = label_batches[-1]

      #loss_function = lambda: tower_loss(model, image_batch, label_batch)
      loss = melt.tower(loss_function, num_gpus)
      pred = model.predict()
      pred = pred['classes']
      #label_batch = dataset.label_batch 
      acc = tf.reduce_mean(tf.to_float(tf.equal(pred, label_batch)))

      #tf.summary.image('train/image', dataset.image_batch)
      # # Compute confusion matrix
      # matrix = tf.confusion_matrix(label_batch, pred, num_classes=10)
      # # Get a image tensor for summary usage
      # image_tensor = draw_confusion_matrix(matrix)
      # tf.summary.image('train/confusion_matrix', image_tensor)

      scope.reuse_variables()
      ops = [loss, acc]

      validator = cifar10_model.ResNetCifar10(
        num_layers,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=False,
        data_format=data_format)

      valid_dataset = cifar10.Cifar10DataSet(data_dir, subset='valid', use_distortion=False)
      valid_iterator = valid_dataset.make_batch(batch_size)
      valid_batch = valid_iterator.get_next()
      valid_id_batches, valid_image_batches, valid_label_batches = melt.split_batch(valid_batch, batch_size, num_gpus)

      def valid_loss_fn(i):
        valid_loss = tower_loss(validator, valid_image_batches[i], valid_label_batches[i])
        valid_pred = validator.predict()
        return valid_id_batches[i], valid_loss, valid_pred['classes'], valid_label_batches[i]
      
      num_valid_examples = dataset.num_examples_per_epoch(subset='valid')
      valid_ops = melt.tower(valid_loss_fn, num_gpus, is_training=False)

      ## seems not work with non rpeat mode..
      #tf.summary.image('valid/image', valid_image_batch)
      ## Compute confusion matrix
      #matrix = tf.confusion_matrix(valid_label_batch, valid_pred, num_classes=10)
      ## Get a image tensor for summary usage
      #image_tensor = draw_confusion_matrix(matrix)
      #tf.summary.image('valid/confusion_matrix', image_tensor)

      #loss_function = lambda: tower_loss(validator, val_image_batch, val_label_batch)
      #val_loss = melt.tower_losses(loss_function, FLAGS.num_gpus, is_training=False)
      #eval_ops = [val_loss]

      metric_eval_fn = lambda model_path=None: \
                          evaluator.evaluate(valid_ops, 
                                             valid_iterator,
                                             num_steps=-(-num_valid_examples // batch_size),
                                             num_examples=num_valid_examples,
                                             model_path=model_path,
                                             num_gpus=num_gpus)

      predictor = cifar10_model.ResNetCifar10(
        num_layers,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=False,
        data_format=data_format)

      predictor.init_predict()

      test_dataset = cifar10.Cifar10DataSet(data_dir, subset='test', use_distortion=False)
      test_iterator = test_dataset.make_batch(batch_size)

      test_batch = test_iterator.get_next()
      test_id_batches, test_image_batches, test_label_batches = test_iterator.get_next() 

      def test_fn(i):
        test_pred = predictor.predict(test_image_batches[i])     
        test_pred = test_pred['classes']
        return test_id_batches[i], test_pred
      
      num_test_examples = dataset.num_examples_per_epoch(subset='test')
      test_ops = melt.tower(test_fn, num_gpus, is_training=False)
      inference_fn = lambda model_path=None: \
                          evaluator.inference(test_ops, 
                                              test_iterator,
                                              num_steps=-(-num_test_examples // batch_size),
                                              num_examples=num_test_examples,
                                              model_path=model_path)

      global eval_names
      names = ['loss', 'acc']

    melt.apps.train_flow(ops, 
                         names = names,
                         metric_eval_fn=metric_eval_fn,
                         inference_fn=inference_fn,
                         model_dir=FLAGS.model_dir,
                         num_steps_per_epoch=num_train_examples // batch_size)

if __name__ == '__main__':
  tf.app.run()  
