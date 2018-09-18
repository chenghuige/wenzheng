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

tf.enable_eager_execution()

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

from prepare.text2ids import text2ids
from wenzheng.utils import ids2text
import numpy as np

def main(_):
  #melt.apps.init()
  
  #ev.init()

  model = getattr(base, FLAGS.model)()

  #melt.eager.restore(model)

  ids2text.init()

  content = '这是一个很好的餐馆，我还想再去'
  #x = {'content': [text2ids(content)]}
  cids = text2ids(content)
  print(cids)
  print(ids2text.ids2text(cids))
  x = {'content': tf.constant([cids])}
  print(model(x))

if __name__ == '__main__':
  tf.app.run()  
