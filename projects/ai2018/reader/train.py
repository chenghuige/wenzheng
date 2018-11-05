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

import sys, os, io
# for p40...
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

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

#from algos.model import *
from algos.loss import criterion
import algos.model as base
from dataset import Dataset
import evaluate as ev

def main(_):
  FLAGS.num_folds = 10
  melt.apps.init()

  Model = getattr(base, FLAGS.model)
  logging.info('Using tensorflow Model:', Model)

  model = Model()
  logging.info(model)
  
  train = melt.apps.get_train()

  ev.init()
  
  train(Dataset,
        model,  
        criterion,
        eval_fn=ev.calc_acc, 
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        valid_names=ev.valid_names,
        valid_suffix='.valid.csv',
        infer_debug_names=ev.valid_names,
        infer_suffix='.infer.txt',
        write_streaming=True)   

if __name__ == '__main__':
  tf.app.run()  
