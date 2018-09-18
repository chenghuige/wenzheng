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

from algos.config import ATTRIBUTES


def main(_):
  melt.apps.init()
  
  #ev.init()

  model = getattr(base, FLAGS.model)()

  melt.eager.restore(model)

  ids2text.init()

  content = '这是一个很好的餐馆，菜很不好吃，我还想再去'
  content = '这是一个很差的餐馆，菜很不好吃，我不想再去'
  content = '这是一个很好的餐馆，菜很好吃，我还想再去'
  content = '这是一个很好的餐馆，只是菜很难吃，我还想再去'
  content = '这是一个很好的餐馆，只是菜很不好吃，我还想再去'

  #content = '味道不错的面馆，性价比也相当之高，分量很足～女生吃小份，胃口小的，可能吃不完呢。环境在面馆来说算是好的，至少看上去堂子很亮，也比较干净，一般苍蝇馆子还是比不上这个卫生状况的。中午饭点的时候，人很多，人行道上也是要坐满的，隔壁的冒菜馆子，据说是一家，有时候也会开放出来坐吃面的人。'
  #x = {'content': [text2ids(content)]}
  cids = text2ids(content)
  print(cids)
  print(ids2text.ids2text(cids))
  x = {'content': tf.constant([cids])}
  logits = model(x)[0]
  probs = gezi.softmax(logits, 1)
  print(probs)
  print(list(zip(ATTRIBUTES, probs)))

  predicts = np.argmax(logits, -1) - 2
  print('predicts ', predicts)
  print(list(zip(ATTRIBUTES, predicts)))
  adjusted_predicts = ev.to_predict(logits)
  print('apredicts', adjusted_predicts)
  print(list(zip(ATTRIBUTES, adjusted_predicts)))


if __name__ == '__main__':
  tf.app.run()  
