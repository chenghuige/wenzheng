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
# torch_algos = projects.ai2018.sentiment.torch_algos
#from algos.model import *
from torch_algos.loss import criterion
import torch_algos.model as base
from dataset import Dataset
import evaluate as ev

def freeze_embedding(self, grad_input, grad_output):
  #print(grad_input)
  #print(grad_output)
  grad_output[0][FLAGS.num_finetune_words:, :] = 0

def freeze_char_embedding(self, grad_input, grad_output):
  #print(grad_input)
  #print(grad_output)
  grad_output[0][FLAGS.num_finetune_chars:, :] = 0

def main(_):
  FLAGS.num_folds = 8
  FLAGS.torch = True
  melt.apps.init()
  
  ev.init()

  embedding = None
  if FLAGS.word_embedding_file and os.path.exists(FLAGS.word_embedding_file):
    embedding = np.load(FLAGS.word_embedding_file)
    FLAGS.emb_dim = embedding.shape[1]

  model = getattr(base, FLAGS.model)(embedding)
  if FLAGS.num_finetune_words:
    model.embedding.register_backward_hook(freeze_embedding)
  if FLAGS.num_finetune_chars:
    model.char_embedding.register_backward_hook(freeze_char_embedding)

  logging.info(model)

  train = melt.apps.get_train()

  train(Dataset,
        model,  
        criterion,
        eval_fn=ev.evaluate, 
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        valid_suffix='.valid.csv',
        infer_suffix='.infer.csv')   

if __name__ == '__main__':
  tf.app.run()  
