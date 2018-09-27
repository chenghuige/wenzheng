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

import torch_algos.baseline.model as base
from torch_algos.baseline.model import criterion
from dataset import Dataset
import evaluate as ev


def main(_):
  FLAGS.torch = True
  melt.apps.init()
  
  Model = getattr(base, FLAGS.model)
  
  model = Model()

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
