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
flags.DEFINE_integer('limit', 3000, '')
flags.DEFINE_integer('threads', None, '')
flags.DEFINE_integer('num_records_', 7, '10 or 5?')
#flags.DEFINE_integer('start_index', 0, 'set it to 1 if you have valid file which you want to put in train as fold 0')
flags.DEFINE_bool('use_fold', False, '')

import traceback
from sklearn.utils import shuffle
import numpy as np
import glob
import json
import pandas as pd

from gezi import Vocabulary
import gezi
import melt

from text2ids import text2ids as text2ids_

from wenzheng.utils import text2ids

import multiprocessing
from multiprocessing import Value, Manager
counter = Value('i', 0)
total_words = Value('i', 0)

df = None


def get_mode(path):
  if 'train' in path:
    return 'train'
  elif 'valid' in path:
    if not FLAGS.use_fold:
      return 'valid' 
    else:
      return 'train'
  elif 'test' in path:
    return 'test'
  elif '.pm' in path:
    return 'pm'
  elif 'trans' in path:
    return 'trans' 
  elif 'deform' in path:
    return 'deform'
  elif 'canyin' in path:
    return 'canyin'
  elif 'dianping' in path:
    return 'dianping'
  return 'train'

def build_features(index):
  mode = get_mode(FLAGS.input)

  start_index = 0 if not FLAGS.use_fold else 1
  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/{1}.record'.format(mode, index + start_index)
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('---out_file', out_file)
  # TODO now only gen one tfrecord file 

  total = len(df)
  num_records = FLAGS.num_records_ 
  if mode in ['valid', 'test', 'dev', 'pm']:
    num_records = 1
  start, end = gezi.get_fold(total, num_records, index)

  print('infile', FLAGS.input, 'out_file', out_file)

  max_len = 0
  max_num_ids = 0
  num = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for i in range(start, end):
      try:
        row = df.iloc[i]
        id = row[0]
        content = row[1] 

        #print(content, type(content))
        if len(content) > max_len:
          max_len = len(content)
          print('max_len', max_len)

        if len(content) > 3000:
          print(id, content)
          if mode not in ['test', 'valid']:
            continue 

        label = list(row[2:])
        
        #label = [x + 2 for x in label]
        #num_labels = len(label)

        content_ids = text2ids_(content)

        if len(content_ids) < 5 and mode not in ['test', 'valid']:
          continue

        limit = FLAGS.limit
        if len(content_ids) > max_num_ids:
          max_num_ids = len(content_ids)
          print('max_num_ids', max_num_ids) 
        content_ids = content_ids[:limit]
        
        feature = {
                    'id': melt.bytes_feature(str(id)),
                    'label': melt.int64_feature(label),
                    'content':  melt.int64_feature(content_ids),
                    'content_str': melt.bytes_feature(content),    
                    'sorce': melt.bytes_feature(mode), 
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
        global total_words
        with total_words.get_lock():
          total_words.value += len(content_ids)
      except Exception:
        #print(traceback.format_exc(), file=sys.stderr)
        pass


def main(_):  
  text2ids.init(FLAGS.vocab_)
  print('to_lower:', FLAGS.to_lower, 'feed_single_en:', FLAGS.feed_single_en, 'seg_method', FLAGS.seg_method)
  print(text2ids.ids2text(text2ids_('傻逼脑残B')))
  print(text2ids.ids2text(text2ids_('喜欢玩孙尚香的加我好友：2948291976')))

  global df
  df = pd.read_csv(FLAGS.input, lineterminator='\n')

  mode = get_mode(FLAGS.input)

  
  pool = multiprocessing.Pool()

  if mode in ['valid', 'test', 'dev', 'pm']:
    FLAGS.num_records_ = 1

  print('num records file to gen', FLAGS.num_records_)

  #FLAGS.num_records_ = 1

  pool.map(build_features, range(FLAGS.num_records_))
  pool.close()
  pool.join()

  #build_features(FLAGS.input)

  # for safe some machine might not use cpu count as default ...
  print('num_records:', counter.value)

  os.system('mkdir -p %s/%s' % (os.path.dirname(FLAGS.vocab_), mode))
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

  print('mean words:', total_words.value / counter.value)

if __name__ == '__main__':
  tf.app.run()
