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
import keras
from keras.utils import to_categorical

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

def main(_):
  num_gpus = get_num_gpus()
  assert num_gpus
  print('num_gpus', get_num_gpus())

  batch_size = FLAGS.batch_size

  #----------- read input
  df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
  df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
  
  x = df_train['id_code']
  y = df_train['diagnosis']

  x, y = shuffle(x, y, random_state=8)
  y = to_categorical(y, num_classes=NUM_CLASSES)
  train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=FLAGS.valid_portion,
                                                      stratify=y, random_state=8)
  #----------  init
  train_data = Dataset(train_x, train_y, 128, is_train=True)
  #batch_size_ = int(batch_size * (3 / 4)) if num_gpus == 1 else batch_size
  batch_size_ = int(batch_size * (3 / 4))

  print('batch_size_', batch_size_)
  train_mixup = Dataset(train_x, train_y, batch_size_, is_train=True, mix=False, augment=True)
  valid_data = Dataset(valid_x, valid_y, batch_size, is_train=False)

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
    validation_data=valid_data,
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
    callbacks=callbacks_list,
    initial_epoch=2)

if __name__ == '__main__':
  flags.DEFINE_integer('batch_size', 32, '')
  flags.DEFINE_float('valid_portion', 0.1, '')

  absl_app.run(main) 
