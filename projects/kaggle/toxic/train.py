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


import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from algos.model import *
from dataset import Dataset

tfe = tf.contrib.eager

from evaluate import *

def main(_):
  FLAGS.model_dir = './mount/temp/kaggle/toxic/model/baseline1'

  FLAGS.emb_dim = 300
  FLAGS.learning_rate = 0.001
  
  FLAGS.optimizer = 'adam'
  #FLAGS.optimizer = 'sgd'

  FLAGS.interval_steps = 1000
  FLAGS.num_epochs = 2
  base = './mount/temp/kaggle/toxic/tfrecords/glove'
  FLAGS.train_input = f'{base}/train/*record,'
  #FLAGS.train_input = f'{base}/train/1.record,'
  FLAGS.test_input = f'{base}/test/*record,'
  FLAGS.vocab = f'{base}/vocab.txt'
  FLAGS.fold = 0
  FLAGS.batch_size = 64
  FLAGS.word_embedding_file = f'{base}/glove.npy'
  #FLAGS.finetune_word_embedding = False
  FLAGS.rnn_hidden_size = 100

  #FLAGS.save_interval_epochs = 0.2 

  # # set legnth index to comment
  # FLAGS.length_index = 1
  # FLAGS.buckets = '100,400'
  # FLAGS.batch_sizes = '64,32,16'

  FLAGS.batch_size = 32

  melt.apps.init()

  model = Model()

  train = melt.apps.get_train()
  
  train(Dataset,
        model,  
        criterion,
        eval_fn=calc_auc, 
        write_valid=True,
        infer_names=['id'] + CLASSES)   

if __name__ == '__main__':
  tf.app.run()  
