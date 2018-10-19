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
flags.DEFINE_integer('threads', None, '')
flags.DEFINE_integer('num_records_', 7, '10 or 5?')
flags.DEFINE_integer('start_index', 0, 'set it to 1 if you have valid file which you want to put in train as fold 0')
flags.DEFINE_bool('use_fold', True, '')
flags.DEFINE_bool('augument', False, '')

import six
import traceback
from sklearn.utils import shuffle
import numpy as np
import glob
import json
import pandas as pd

import jieba.posseg

from gezi import Vocabulary
import gezi
from gezi import melt

from text2ids import text2ids as text2ids_

import wenzheng
from wenzheng.utils import text2ids

import config
from projects.ai2018.sentiment.prepare import filter

import multiprocessing
from multiprocessing import Value, Manager
counter = Value('i', 0)
total_words = Value('i', 0)

df = None

vocab = None
char_vocab = None
tag_vocab = None

def get_mode(path):
  mode = 'train'
  if 'train' in path:
    mode ='train'
  elif 'valid' in path:
    mode = 'train'
  elif 'test' in path:
    mode = 'test'
  elif '.pm' in path:
    mode = 'pm'
  elif 'trans' in path:
    mode = 'trans' 
  elif 'deform' in path:
    mode = 'deform'
  elif 'canyin' in path:
    mode = 'canyin'
  elif 'dianping' in path:
    mode = 'dianping'
  if FLAGS.augument:
    mode = 'aug.' + mode
  return mode

def build_features(index):
  mode = get_mode(FLAGS.input)

  start_index = FLAGS.start_index

  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/{1}.record'.format(mode, index + start_index)
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('---out_file', out_file)
  # TODO now only gen one tfrecord file 

  total = len(df)
  num_records = FLAGS.num_records_ 
  if mode.split('.')[-1] in ['valid', 'test', 'dev', 'pm'] or 'valid' in FLAGS.input:
    num_records = 1
  start, end = gezi.get_fold(total, num_records, index)

  print('total', total, 'infile', FLAGS.input, 'out_file', out_file)

  max_len = 0
  max_num_ids = 0
  num = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for i in range(start, end):
      try:
        row = df.iloc[i]
        id = str(row[0])

        if start_index > 0:
          id == 't' + id

        content = row[1] 

        content = filter.filter(content)

        label = list(row[2:])
        
        #label = [x + 2 for x in label]
        #num_labels = len(label)

        content_ids, words = text2ids_(content, return_words=True)
        assert len(content_ids) == len(words)

        if len(content_ids) > max_len:
          max_len = len(content_ids)
          print('max_len', max_len)

        if len(content_ids) > FLAGS.word_limit:
          print(id, content)
          if mode not in ['test', 'valid']:
            continue 

        if len(content_ids) < 5 and mode not in ['test', 'valid']:
          continue

        content_ids = content_ids[:FLAGS.word_limit]
        words = words[:FLAGS.word_limit]

        if FLAGS.use_char:
          chars = [list(word) for word in words]
          char_ids = np.zeros([len(content_ids), FLAGS.char_limit], dtype=np.int32)
          
          vocab_ = char_vocab if char_vocab else vocab

          for i, token in enumerate(chars):
            for j, ch in enumerate(token):
              if j == FLAGS.char_limit:
                break
              char_ids[i, j] = vocab_.id(ch)

          char_ids = list(char_ids.reshape(-1))
        else:
          char_ids = [0]

        if tag_vocab:
          l = list(jieba.posseg.cut(content))
          tags = [tag for word, tag in l]
          
          #assert len(tags) == len(content_ids)
          tags = tags[:FLAGS.word_limit]
          # print(len(words), len(tags), len(content))
          # print(content)
          # print(list(zip(range(len(words)), words)))
          # print(list(zip(range(len(l)),[word for word, tag in l])))
          for i in range(len(tags)):
            if words[i] == '\x01' or words[i] == '\x02' or words[i] == '\x03':
              tags[i] = 'sep'
          tag_ids = [tag_vocab.id(x) for x in tags]
          #print(content, list(zip(words, tag_ids, tags)))
          #exit(0)
        else:
          tag_ids = [0]

        feature = {
                    'id': melt.bytes_feature(id),
                    'label': melt.int64_feature(label),
                    'content':  melt.int64_feature(content_ids),
                    'content_str': melt.bytes_feature(content), 
                    'chars': melt.int64_feature(char_ids),
                    'source': melt.bytes_feature(mode), 
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
        print(traceback.format_exc(), file=sys.stderr)
        pass


def main(_):  
  assert FLAGS.use_fold
  text2ids.init(FLAGS.vocab_)
  global vocab, char_vocab, tag_vocab
  vocab = text2ids.vocab
  char_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'char_vocab.txt')
  if os.path.exists(char_vocab_file):
    char_vocab = Vocabulary(char_vocab_file)
  tag_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'tag_vocab.txt')
  if os.path.exists(tag_vocab_file):
    tag_vocab = Vocabulary(tag_vocab_file)

  print('to_lower:', FLAGS.to_lower, 'feed_single_en:', FLAGS.feed_single_en, 'seg_method', FLAGS.seg_method)
  print(text2ids.ids2text(text2ids_('傻逼脑残B')))
  print(text2ids.ids2text(text2ids_('喜欢玩孙尚香的加我好友：2948291976')))

  global df
  df = pd.read_csv(FLAGS.input, lineterminator='\n')

  mode = get_mode(FLAGS.input)

  
  pool = multiprocessing.Pool()

  if mode.split('.')[-1] in ['valid', 'test', 'dev', 'pm'] or 'valid' in FLAGS.input:
    FLAGS.num_records_ = 1

  print('num records file to gen', FLAGS.num_records_)

  #FLAGS.num_records_ = 1

  pool.map(build_features, range(FLAGS.num_records_))
  pool.close()
  pool.join()

  # for i in range(FLAGS.num_records_):
  #   build_features(i)

  # for safe some machine might not use cpu count as default ...
  print('num_records:', counter.value)

  os.system('mkdir -p %s/%s' % (os.path.dirname(FLAGS.vocab_), mode))
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

  print('mean words:', total_words.value / counter.value)

if __name__ == '__main__':
  tf.app.run()
