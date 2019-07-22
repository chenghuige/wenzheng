#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:40.029962
#   \Description   still multiple gpu not correct result...
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app as absl_app
from absl import flags
FLAGS = flags.FLAGS

from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 

# create callbacks list
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger,
                             TensorBoard)

from keras.utils import to_categorical
import keras

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import * 
from dataset import Dataset
from evaluate import QWKEvaluation
from model import create_model

def get_num_gpus():
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print("os.environ['CUDA_VISIBLE_DEVICES']", os.environ['CUDA_VISIBLE_DEVICES'])
    if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
      return 0
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print('CUDA_VISIBLE_DEVICES is %s'%(os.environ['CUDA_VISIBLE_DEVICES']))
    return num_gpus
  else:
    return None

def get_dataset(subset, use_distortion=None):
  data_dir = '../input/tfrecords/' 
  if use_distortion is None:
    use_distortion = False
    if subset == 'train':
      use_distortion = True
  return Dataset(data_dir, subset=subset, use_distortion=use_distortion) 

def main(_):
  # TODO FIXME RuntimeError: tf.placeholder() is not compatible with eager execution.
  tf.enable_eager_execution()

  num_gpus = get_num_gpus()
  assert num_gpus
  print('num_gpus', get_num_gpus())

  batch_size = FLAGS.batch_size

  #----------  init
  train_data = get_dataset('train', use_distortion=False).make_batch(batch_size)
  batch_size_ = int(batch_size * (3 / 4)) if num_gpus == 1 else batch_size
  print('batch_size_', batch_size_)
  train_mixup = get_dataset('train', use_distortion=True).make_batch(batch_size_)
  valid_data = get_dataset('valid').make_batch(batch_size)

  # train step1, warm up model
  model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=NUM_CLASSES)
  
  for layer in model.layers:
    layer.trainable = False

  for i in range(-3,0):
    model.layers[i].trainable = True

  if num_gpus > 1:
    smodel = model
    model = keras.utils.multi_gpu_model(model, num_gpus, cpu_merge=False)

  model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-3))

  tb = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

  qwk = QWKEvaluation(validation_data=(valid_data, valid_y),
                      batch_size=batch_size, interval=1)  

  model.fit_generator(
    train_data,
    epochs=2,
    workers=WORKERS, 
    use_multiprocessing=True,
    verbose=1,
    callbacks=[qwk])

  # train step2, train all layers
  checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min', save_weights_only = True)
  reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                    verbose=1, mode='auto', epsilon=0.0001)
  early = EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=9)

  csv_logger = CSVLogger(filename='../working/training_log.csv',
                        separator=',',
                        append=True)

  if num_gpus > 1:
    model = smodel

  for layer in model.layers:
    layer.trainable = True
  callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early, qwk]

  if num_gpus > 1:
    smodel = model 
    model = keras.utils.multi_gpu_model(model, num_gpus, cpu_merge=False)

  model.compile(loss='categorical_crossentropy',
                # loss=kappa_loss,
                optimizer=Adam(lr=1e-4))

  model.fit_generator(
    train_mixup,
    validation_data=valid_data,
    epochs=epochs,
    verbose=1,
    workers=1, 
    use_multiprocessing=False,
    callbacks=callbacks_list)

if __name__ == '__main__':
  flags.DEFINE_integer('batch_size', 32, '')
  flags.DEFINE_float('valid_portion', 0.1, '')

  absl_app.run(main) 
