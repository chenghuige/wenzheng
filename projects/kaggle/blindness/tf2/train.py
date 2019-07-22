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
from tensorflow import keras

flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('model_dir', './mount/temp/cifar10/model/resnet', '')
flags.DEFINE_string('algo', 'resnet', '')

import numpy as np

import melt 
logging = melt.logging
import gezi
import traceback

#from model import create_model
from model import Model 

from dataset import DataSet, HEIGHT, WIDTH

from loss import criterion
import evaluate as ev

NUM_CLASSES = 5

def get_dataset(subset):
  data_dir = '../input/tfrecords/' 
  use_distortion = False
  if subset == 'train':
    use_distortion = True
  return DataSet(data_dir, subset=subset, use_distortion=use_distortion) 

# TODO FIXME why 1gpu ok 2gpu fail ? train.py all ok... ai2018/sentiment/model.py bert is also all ok why here wrong ?
def main(_):
  tf.enable_eager_execution()
  melt.apps.init()

  #model = create_model((HEIGHT,WIDTH,3), NUM_CLASSES)
  #model = Model((HEIGHT,WIDTH,3), NUM_CLASSES)
  model = Model()

  #print(model.summary())

  # # logging.info(model)
  # fit = melt.apps.get_fit()

  # fit(get_dataset,
  #     model,  
  #     criterion,
  #     eval_fn=ev.evaluate,
  #     valid_write_fn=ev.valid_write,
  #     infer_write_fn=ev.infer_write,
  #     valid_suffix='.valid.csv',
  #     infer_suffix='.infer.csv',
  #     #optimizer=keras.optimizers.Adam(lr=1e-4)
  #     )   
  
  dt = get_dataset('train')
  train_data = dt.make_batch(FLAGS.batch_size, ['../input/tfrecords/train.tfrecords'])
  #train_data[0] = train_data[0]['image']
  dt_valid = get_dataset('valid')
  valid_data = dt_valid.make_batch(FLAGS.batch_size, ['../input/tfrecords/valid.tfrecords'])
  #valid_data[0] = valid_data[0]['image']

  model.compile(
          loss='sparse_categorical_crossentropy',
          optimizer=keras.optimizers.Adam(lr=1e-4),
          #metrics=['accuracy'])
  )

  model.fit(
            train_data,
            steps_per_epoch=np.ceil(DataSet.num_examples_per_epoch('train') / FLAGS.batch_size),
            validation_data=valid_data,
            validation_steps=np.ceil(DataSet.num_examples_per_epoch('valid') / FLAGS.batch_size),
            epochs=5)

  # model.fit_generator(
  #           train_data,
  #           steps_per_epoch=np.ceil(DataSet.num_examples_per_epoch('train') / FLAGS.batch_size),
  #           validation_data=valid_data,
  #           validation_steps=np.ceil(DataSet.num_examples_per_epoch('valid') / FLAGS.batch_size),
  #           epochs=5)

if __name__ == '__main__':
  tf.app.run()  
