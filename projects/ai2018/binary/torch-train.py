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

import torch
import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback
import lele

from wenzheng.utils import input_flags 

from algos import config

from torch_algos.loss import Criterion
import torch_algos.model as base
from dataset import Dataset
import evaluate as ev

def main(_):
  FLAGS.torch = True
  melt.apps.init()
  
  ev.init()
  
  embedding = None
  if FLAGS.word_embedding_file and os.path.exists(FLAGS.word_embedding_file):
    embedding = np.load(FLAGS.word_embedding_file)
    FLAGS.emb_dim = embedding.shape[1]
    #model = getattr(base, FLAGS.model)(embedding)
    model = getattr(base, 'MReader')(embedding)

  logging.info('model\n', model)


  if FLAGS.lm_path:
    _, updated_params = lele.load(model, FLAGS.lm_path)
    assert updated_params, FLAGS.lm_path

  train = melt.apps.get_train()
  criterion = Criterion()
  
  train(Dataset,
        model,  
        criterion.forward,
        eval_fn=ev.evaluate)   

if __name__ == '__main__':
  tf.app.run()  
