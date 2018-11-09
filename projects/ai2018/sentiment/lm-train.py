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
from lm_dataset import Dataset

def main(_):
  FLAGS.num_folds = 8
  melt.apps.init()

  embedding = None
  if FLAGS.word_embedding_file and os.path.exists(FLAGS.word_embedding_file):
    embedding = np.load(FLAGS.word_embedding_file)
    FLAGS.emb_dim = embedding.shape[1]

  model = getattr(base, FLAGS.model)(embedding, lm_model=True)

  logging.info(model)

  train = melt.apps.get_train()

  train(Dataset,
        model,  
        melt.losses.bilm_loss)   

if __name__ == '__main__':
  tf.app.run()  
