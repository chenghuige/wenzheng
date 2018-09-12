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

from algos.model import *
import algos.model as base
from dataset import Dataset
import evaluate as ev

def main(_):
  melt.apps.init()

  model = getattr(base, FLAGS.model)()

  train = melt.apps.get_train()

  ev.init()
  
  train(Dataset,
        model,  
        criterion,
        eval_fn=ev.calc_acc, 
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        write_valid=True,
        valid_names=['id', 'label', 'predict', 'query', 'passage'],
        valid_suffix='.valid.csv',
        infer_suffix='.infer.txt')   

if __name__ == '__main__':
  tf.app.run()  
