#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:40.029962
#   \Description   still multiple gpu not correct result...
#                  seems keras bug, train 90% valid 10% multiple gpu ok
#                  train 80% valid 20% multiple gpu wrong..
# to reproduce
# c01 python train2.py ok
# c01 python train2.py --valid_portion=0.1 wrong
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import * 
from dataset import *
from evaluate import Evaluator
from model import create_model
from loss import get_loss

import folds
from util import *

def to_regression(y):
  bs = len(y)
  y = np.argmax(y, -1).reshape([bs, 1])
  return y 

def to_regression2(y):
  bs = len(y)
  y = np.argmax(y, -1).reshape([bs, 1])
  y = y / (NUM_CLASSES - 1)
  return y 

def to_ordinal(y):
  # [0,0,1,0,0] ->[1,1,1,0,0] for multi label loss
  y_ = np.empty(y.shape, dtype=y.dtype)
  y_[:, NUM_CLASSES - 1] = y[:, NUM_CLASSES - 1]

  for i in range(NUM_CLASSES - 2, -1, -1):
      y_[:, i] = np.logical_or(y[:, i], y_[:, i+1])
  y = y_  
  return y

def to_ordinal2(y):
  y = to_ordinal(y)
  y = y[:, 1:]
  return y

def trans_y(y, loss_type):
  if 'regression' in loss_type:
    if 'sigmoid2' in loss_type:
      return to_regression2(y)
    else:
      return to_regression(y)
  elif 'ordinal' in loss_type:
    if '2' in loss_type:
      return to_ordinal2(y)
    else:
      return to_ordinal(y)
  return y

def main(_):
  num_gpus = get_num_gpus()
  assert num_gpus
  print('num_gpus', get_num_gpus())

  batch_size = FLAGS.batch_size

  # ordinal2 better then ordinal since do not need 5 bits just need 4 bits 
  # so classification, sigmoid_regression and ordinal2_classification similar result, classification maybe slightly better
  # for regression, tend to predict less 4..
  loss_types = ['classification', 'linear_regression', \
                'sigmoid_regression', 'sigmoid_regression_mae', 'sigmoid2_regression', \
                'ordinal_classification', 'ordinal2_classification', 
                'earth_classification', 'kappa_classification']
  
  loss_type = FLAGS.loss_type
  print('loss_type', loss_type)
  assert loss_type in loss_types

  #----------- read input
  df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
  df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
  
  x = df_train['id_code']
  y = df_train['diagnosis']
  x, y = shuffle(x, y, random_state=random_state)
  
  # https://stackoverflow.com/questions/48508036/sklearn-stratifiedkfold-valueerror-supported-target-types-are-binary-mul
  # Supported target types are: ('binary', 'multiclass'). Got 'multilabel-indicator' instead. 
  # can not put after to_categorical
  train_x, valid_x, train_y, valid_y = folds.get_train_valid(x, y, FLAGS.fold, FLAGS.num_folds, random_state=2019)

  # # check if exactly same as gen-folds, so fold 0 valid.csv should be same as ../input/train_0.csv
  # df = pd.DataFrame()
  # df['id_code'] = valid_x
  # df['diagnosis'] = valid_y 
  # df.to_csv('valid.csv', index=False)

  train_y = to_categorical(train_y, num_classes=NUM_CLASSES)
  train_y = trans_y(train_y, loss_type)

  valid_y = to_categorical(valid_y, num_classes=NUM_CLASSES)
  valid_y = trans_y(valid_y, loss_type)
  

  #----------  init
  train_data = Dataset(train_x, train_y, 128, is_train=True)
  
  #batch_size_ = int(batch_size * (3 / 4)) if num_gpus == 1 else batch_size
  batch_size_ = int(batch_size * (3 / 4)) * FLAGS.multiplier  # 24 48
  
  print('batch_size_', batch_size_)
  train_mixup = Dataset(train_x, train_y, batch_size_, is_train=True, mix=False, augment=True)
  valid_data = Dataset(valid_x, valid_y, batch_size, is_train=False)

  # train step1, warm up model
  model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=NUM_CLASSES,
    loss_type=loss_type)
  
  for layer in model.layers:
    layer.trainable = False

  for i in range(-3,0):
    model.layers[i].trainable = True

  if num_gpus > 1:
    smodel = model
    model = keras.utils.multi_gpu_model(model, num_gpus, cpu_merge=False)

  loss_fn = get_loss(loss_type)

  print('loss_fn', loss_fn)

  model.compile(
    loss=loss_fn,
    optimizer=Adam(1e-3))

  #model.summary()

  dir = '../working/{}/{}'.format(FLAGS.fold, loss_type)
  if FLAGS.multiplier > 1:
    dir += '_{}'.format(FLAGS.multiplier)
  if num_gpus > 1:
    dir += '_{}gpu'.format(num_gpus)

  print('dir:', dir)

  tb = TensorBoard(log_dir=dir, histogram_freq=0,
                   write_graph=True, write_images=False)

  eval = Evaluator(dir,
                   validation_data=(valid_data, valid_y),
                   interval=1, 
                   loss_type=loss_type)  

  ## for faster check evaluate probelm but may cuase problem for mutltigpu wrong eval TODO FIXME
  # eval.model = model
  # eval.on_epoch_end(-1)
  # print('image_dict size', len(image_dict))

  model.fit_generator(
    train_data,
    validation_data=valid_data,
    epochs=1,
    workers=WORKERS, 
    use_multiprocessing=True,
    verbose=1,
    callbacks=[eval, tb])

  # seems if use_multiprocessing=True will not update image_dict
  print('image_dict size', len(image_dict))
  

  # train step2, train all layers
  checkpoint = ModelCheckpoint('{}/densenet_.h5'.format(dir), monitor='val_loss', verbose=1, 
                               save_best_only=True, mode='min', save_weights_only = True)
  reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                     verbose=1, mode='auto', epsilon=0.0001)
  early = EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=9)

  csv_logger = CSVLogger(filename='{}/training_log.csv'.format(dir),
                        separator=',',
                        append=True)

  # from lr import WarmUpCosineDecayScheduler
  # warmup_epoch = 2
  # warmup_steps = warmup_epoch * len(train_mixup)
  # warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=2e-4,
  #                                         total_steps=len(train_mixup) * (epochs - 2),
  #                                         warmup_learning_rate=0.0,
  #                                         warmup_steps=warmup_steps,
  #                                         hold_base_rate_steps=0)

  if num_gpus > 1:
    model = smodel

  for layer in model.layers:
    layer.trainable = True
  
  callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early, eval, tb]
  #callbacks_list = [checkpoint, csv_logger, warm_up_lr, early, eval, tb]


  if num_gpus > 1:
    smodel = model 
    model = keras.utils.multi_gpu_model(model, num_gpus, cpu_merge=False)

  lr = 1e-4
  lr *= FLAGS.multiplier
  # Notice if using warm_up_lr then lr here not on effect
  model.compile(loss=loss_fn,
                optimizer=Adam(lr=lr))

  epoch_now = 1
  model.fit_generator(
    train_mixup,
    validation_data=valid_data,
    epochs=epochs,
    verbose=1,
    ## FIXME probelm with callback save model OSError: Unable to create file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
    ## will be slower (50->57) not using multiprocessing seems problem only occur for validation when saving checkpoint
    # workers=WORKERS, 
    # use_multiprocessing=True,
    workers=1, 
    use_multiprocessing=False,
    callbacks=callbacks_list,
    initial_epoch=epoch_now)

if __name__ == '__main__':
  flags.DEFINE_integer('batch_size', 32, '')
  flags.DEFINE_integer('num_folds', 5, '')
  flags.DEFINE_integer('fold', 0, '')
  flags.DEFINE_integer('multiplier', 1, '')

  flags.DEFINE_string('loss_type', 'classification', 
                      'classification, linear_regression, sigmoid_regression, ordinal_classification')

  absl_app.run(main) 
