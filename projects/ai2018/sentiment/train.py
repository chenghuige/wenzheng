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

from wenzheng.utils import input_flags 

# import projects
# algos = projects.ai2018.sentiment.algos
#from algos.model import *
from algos.loss import criterion
import algos.model as base
from dataset import Dataset
import evaluate as ev

def main(_):
  FLAGS.num_folds = 8
  FLAGS.model = FLAGS.model or 'RNet'
  melt.apps.init()

  ev.init()

  embedding = None
  if FLAGS.word_embedding_file and os.path.exists(FLAGS.word_embedding_file):
    embedding = np.load(FLAGS.word_embedding_file)
    FLAGS.emb_dim = embedding.shape[1]

  model = getattr(base, FLAGS.model)(embedding)

  logging.info(model)

  train = melt.apps.get_train()

  init_fn = None
  # TODO FIXME should like below but now has problem 
  #     File "/home/gezi/mine/wenzheng/utils/melt/util.py", line 38, in create_restore_fn
  #     assert variables_to_restore, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  # AssertionError: [<tf.Variable 'learning_rate_weight:0' shape=() dtype=float32_ref>, <tf.Variable 'init_fw_0:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_bw_0:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_fw_0_1:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_fw_1:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_bw_0_1:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_bw_1:0' shape=(1, 200) dtype=float32>, <tf.Variable 'embedding_kernel:0' shape=(20, 300) dtype=float32>, <tf.Variable 'init_fw_0_2:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_bw_0_2:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_fw_0_3:0' shape=(1, 200) dtype=float32>, <tf.Variable 'init_bw_0_3:0' shape=(1, 200) dtype=float32>]

  # if FLAGS.lm_path:
  #   init_fn = melt.create_restore_fn(FLAGS.lm_path, FLAGS.model, 'TextEncoder')

  train(Dataset,
        model,  
        criterion,
        eval_fn=ev.evaluate,
        init_fn=init_fn, 
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        valid_suffix='.valid.csv',
        infer_suffix='.infer.csv')   

if __name__ == '__main__':
  tf.app.run()  
