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

from algos.config import CLASSES


def main(_):
  melt.apps.init()
  
  #ev.init()

  model = getattr(base, FLAGS.model)()
  model.debug = True

  melt.eager.restore(model)

  ids2text.init()
  vocab = ids2text.vocab

  # query = '阿里和腾讯谁更流氓'
  # passage = '腾讯比阿里流氓'

  # query = 'c罗和梅西谁踢球更好'
  # passage = '梅西比c罗踢的好'
  query = '青光眼遗传吗'
  passage = '青光眼有遗传因素的，所以如果是您的父亲是青光眼的话，那我们在这里就强烈建议您，自己早期到医院里面去做一个筛查，测一下，看看眼，尤其是检查一下视野，然后视网膜的那个情况，都做一个早期的检查。'
  
  qids = text2ids(query)
  qwords = [vocab.key(qid) for qid in qids]
  print(qids)
  print(ids2text.ids2text(qids))
  pids = text2ids(passage)
  pwords = [vocab.key(pid) for pid in pids]
  print(pids)
  print(ids2text.ids2text(pids))

  x = {
        'query': [qids], 
        'passage':  [pids],
        'type': [0],
      }

  logits = model(x)[0]
  probs = gezi.softmax(logits)
  print(probs)
  print(list(zip(CLASSES, [x for x in probs])))

  predict = np.argmax(logits, -1) 
  print('predict', predict, CLASSES[predict])

  # print words importance scores
  word_scores_list = model.pooling.word_scores

  for word_scores in word_scores_list:
    print(list(zip(pwords, word_scores[0].numpy())))


if __name__ == '__main__':
  tf.app.run()  
