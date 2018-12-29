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

flags.DEFINE_string('input', './mount/data/ai2018/binary/valid.txt', '') 
flags.DEFINE_string('vocab_', './mount/temp/ai2018/binary/tfrecord/vocab.txt', 'vocabulary txt file')
#flags.DEFINE_string('seg_method', 'basic', '') 
flags.DEFINE_bool('binary', False, '')
flags.DEFINE_integer('threads', None, '')
flags.DEFINE_integer('num_records_', None, '10 or 5?')
flags.DEFINE_integer('start_index', 0, 'set it to 1 if you have valid file which you want to put in train as fold 0')
flags.DEFINE_bool('use_fold', True, '')
flags.DEFINE_bool('augument', False, '')
flags.DEFINE_string('mode', None, '')
flags.DEFINE_string('mode_', None, '')
flags.DEFINE_bool('ignore_start_end', False, 'If you have not remove start and end quota before,you can filter here')
flags.DEFINE_bool('add_start_end_', True, '')
flags.DEFINE_bool('has_position', False, '')
flags.DEFINE_bool('fixed_vocab', False, '')
flags.DEFINE_string('start_mark', '<S>', '')
flags.DEFINE_string('end_mark', '</S>', '')
flags.DEFINE_string('unk_word', '<UNK>', '')
flags.DEFINE_bool('word_only', False, '')
flags.DEFINE_bool('use_soft_label_', False, '')
flags.DEFINE_bool('is_soft_label', False, '')

import six
import traceback
from sklearn.utils import shuffle
import numpy as np
import glob
import json
import pandas as pd

from tqdm import tqdm

from gezi import Vocabulary
import gezi
#assert gezi.env_has('JIEBA_POS')
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
pos_vocab = None
ner_vocab = None

seg_result = None
pos_result = None
ner_result = None

def get_mode(path):
  mode = 'train'
  if 'train' in path:
    mode ='train'
  elif 'valid' in path:
    mode = 'valid'
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
  elif 'ensemble.infer.debug.csv' in path:
    mode = 'test'
  if FLAGS.augument:
    mode = 'aug.' + mode
  if FLAGS.mode:
    mode = FLAGS.mode
  return mode

def build_features(index):
  mode = get_mode(FLAGS.input)

  start_index = FLAGS.start_index

  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/{1}.record'.format(mode, index + start_index)
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('---out_file', out_file)
  # TODO now only gen one tfrecord file 

  total = len(df)
  num_records = FLAGS.num_records_ 
  ## TODO FIXME whty here still None ? FLAGS.num_records has bee modified before in main as 7 ...
  #print('---------', num_records, FLAGS.num_records_)
  if not num_records:
    if mode.split('.')[-1] in ['valid', 'test', 'dev', 'pm'] or 'valid' in FLAGS.input:
      num_records = 1
    else:
      num_records = 1
  #print('------------------', num_records, FLAGS.num_records_)
  start, end = gezi.get_fold(total, num_records, index)

  print('total', total, 'infile', FLAGS.input, 'out_file', out_file, 'num_records', num_records,  'start', start, 'end', end)

  max_len = 0
  max_num_ids = 0
  num = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for i in tqdm(range(start, end), ascii=True):
      try:
        #row = df.iloc[i]
        row = df[i]
        id = str(row[0])

        words = row[-1].split('\t')

        content = row[2] 
        content_ori = content
        content = filter.filter(content)
        
        label = int(row[1])

        content_ids = [vocab.id(x) for x in words]

        if len(content_ids) > max_len:
          max_len = len(content_ids)
          print('max_len', max_len)

        if len(content_ids) > FLAGS.word_limit and len(content_ids) < 5:
          print('{} {} {}'.format(id, len(content_ids), content_ori))

        content_ids = content_ids[:FLAGS.word_limit]
        words = words[:FLAGS.word_limit]

        # NOTICE different from tf, pytorch do not allow all 0 seq for rnn.. if using padding mode  
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
          if np.sum(char_ids) == 0:
            print('------------------------bad id', id)
            print(content_ids)
            print(words)
            exit(0)
        else:
          char_ids = [0]

        feature = {
                    'id': melt.bytes_feature(id),
                    'content':  melt.int64_feature(content_ids),
                    'content_str': melt.bytes_feature(content_ori), 
                    'char': melt.int64_feature(char_ids),
                    'source': melt.bytes_feature(mode), 
                  }
        feature['label'] = melt.int64_feature(label) 

        # TODO currenlty not get exact info wether show 1 image or 3 ...
        record = tf.train.Example(features=tf.train.Features(feature=feature))

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
  mode = get_mode(FLAGS.input)
 
  global vocab, char_vocab 
  vocab = gezi.Vocabulary(FLAGS.vocab_, fixed=FLAGS.fixed_vocab, unk_word=FLAGS.unk_word)
  char_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'char_vocab.txt')
  if os.path.exists(char_vocab_file):
    char_vocab = Vocabulary(char_vocab_file)
    print('char vocab size:', char_vocab.size())
  
  mode_ = 'train'
  if 'valid' in FLAGS.input:
    mode_ = 'valid'
  elif 'test' in FLAGS.input:
    mode_ = 'test'
  else:
    assert 'train' in FLAGS.input

  if FLAGS.augument:
    mode_ = 'aug.' + mode_

  if FLAGS.mode_:
    mode_ = FLAGS.mode_

  global df
  df = []
  for line in open(FLAGS.input):
    df.append(line.strip().split('\t', 3))
  
  pool = multiprocessing.Pool()

  if not FLAGS.num_records_:
    if mode.split('.')[-1] in ['valid', 'test', 'dev', 'pm'] or 'valid' in FLAGS.input:
      FLAGS.num_records_ = 1
    else:
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
