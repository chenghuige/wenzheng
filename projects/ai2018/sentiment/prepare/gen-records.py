#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2018-08-29 15:20:35.282947
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', './mount/data/ai2018/sentiment/valid.csv', '') 
flags.DEFINE_string('vocab_', './mount/temp/ai2018/sentiment/tfrecord/vocab.txt', 'vocabulary txt file')
#flags.DEFINE_string('seg_method', 'basic', '') 
flags.DEFINE_bool('binary', False, '')
flags.DEFINE_integer('limit', 1000, '')
flags.DEFINE_bool('feed_single_en', True, '')
flags.DEFINE_bool('to_lower', True, '')
flags.DEFINE_integer('threads', None, '')

import traceback
from sklearn.utils import shuffle
import numpy as np
import glob
import json
import pandas as pd

from gezi import Vocabulary
import gezi
import melt
from deepiu.util import text2ids

from multiprocessing import Value, Manager
counter = Value('i', 0) 

def _text2ids(text):
  return text2ids.text2ids(text, seg_method=FLAGS.seg_method, 
                           feed_single_en=FLAGS.feed_single_en,
                           to_lower=FLAGS.to_lower,
                           norm_digit=False,
                           pad=False)

def get_mode(path):
  if 'train' in path:
    return 'train'
  elif 'valid' in path:
    return 'valid' 
  elif 'test' in path:
    return 'test'
  elif '.pm' in path:
    return 'pm'
  return 'train'

def build_features(file_):
  mode = get_mode(FLAGS.input)
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/{1}_{2}.tfrecord'.format(mode, os.path.basename(os.path.dirname(file_)), os.path.basename(file_))
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('infile', file_, 'out_file', out_file)

  df = pd.read_csv(file_)
  
  num = 0
  num_whether = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for i in range(len(df)):
      try:
        row = df.iloc[i]
        id = row[0]
        content = row[1]
        label = list(row[2:])
        #num_labels = len(label)

        limit = 5000
        content = content[:limit]
        content_ids = _text2ids(content)

        feature = {
                    'id': melt.bytes_feature(str(id)),
                    'label': melt.int64_feature(label),
                    'content':  melt.int64_feature(content_ids),
                    'content_str': melt.bytes_feature(content),     
                  }

        # TODO currenlty not get exact info wether show 1 image or 3 ...
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        if num % 1000 == 0:
          print(num)

        writer.write(record)
        num += 1
        global counter
        with counter.get_lock():
          counter.value += 1
      except Exception:
        print(traceback.format_exc(), file=sys.stderr)


def main(_):  
  text2ids.init(FLAGS.vocab_)
  print('to_lower:', FLAGS.to_lower, 'feed_single_en:', FLAGS.feed_single_en, 'seg_method', FLAGS.seg_method)
  print(text2ids.ids2text(_text2ids('傻逼脑残B')))
  print(text2ids.ids2text(_text2ids('喜欢玩孙尚香的加我好友：2948291976')))
  
  build_features(FLAGS.input)

  # for safe some machine might not use cpu count as default ...
  print('num_records:', counter.value)
  mode = get_mode(FLAGS.input)

  os.system('mkdir -p %s/%s' % (os.path.dirname(FLAGS.vocab_), mode))
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  tf.app.run()
