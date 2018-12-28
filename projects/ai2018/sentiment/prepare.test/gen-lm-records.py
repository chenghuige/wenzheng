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

flags.DEFINE_string('input', './mount/data/my-embedding/Glove-sentiment-jieba/valid', '') 
flags.DEFINE_string('vocab_', './mount/temp/ai2018/sentiment/tfrecords/word.jieba.ft/vocab.txt', 'vocabulary txt file')
flags.DEFINE_integer('threads', None, '')
flags.DEFINE_string('source', 'dianping', 'dianping or baike or zhidao or zhihu')
flags.DEFINE_integer('max_sentence_len', 20, '')
flags.DEFINE_string('tfrecord_dir', 'tfrecord', '')

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

vocab = None
char_vocab = None

def build_features(file_):
  if not os.path.isfile(file_):
    return 

  file_name = os.path.basename(file_)
  assert os.path.isdir(FLAGS.input)
  mode = 'train' if 'train' in FLAGS.input else 'valid'
  dir_ = os.path.dirname(os.path.dirname(FLAGS.input))
  out_file = os.path.join(dir_ , '{}/{}/{}.record'.format(FLAGS.tfrecord_dir, mode, file_name))
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  
  print('infile', file_, 'out_file', out_file)

  # if os.path.exists(out_file):
  #   return

  max_len = 0
  max_num_ids = 0
  num = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for line in tqdm(open(file_), total=1e6, ascii=True):
      try:
        line = line.rstrip('\n')
        line = filter.filter(line)
        words = line.split(' ')
        words = gezi.add_start_end(words)
        words_list = gezi.break_sentence(words, FLAGS.max_sentence_len)
        for words in words_list:
          content = ' '.join(words)
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
          else:
            char_ids = [0]

          feature = {
                      'content':  melt.int64_feature(content_ids),
                      'content_str': melt.bytes_feature(content), 
                      'char': melt.int64_feature(char_ids),
                      'source': melt.bytes_feature(FLAGS.source), 
                    }

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
  FLAGS.word_limit = 2000
  global vocab, char_vocab
  vocab = gezi.Vocabulary(FLAGS.vocab_)
  print('vocab file', FLAGS.vocab_, 'vocab size', vocab.size())
  if FLAGS.use_char:
    char_vocab = gezi.Vocabulary(FLAGS.vocab_.replace('vocab.txt', 'char_vocab.txt'))

  files = glob.glob(FLAGS.input + '/*') 
  pool = multiprocessing.Pool(multiprocessing.cpu_count())
  pool.map(build_features, files)
  pool.close()
  pool.join()

  # for safe some machine might not use cpu count as default ...
  print('num_records:', counter.value)

  mode = 'train' if 'train' in FLAGS.input else 'valid'
  dir_ = os.path.dirname(os.path.dirname(FLAGS.input))
  os.system('mkdir -p %s/%s/%s' % (dir_, FLAGS.tfrecord_dir, mode))
  out_file = os.path.join(dir_, '{}/{}/num_records.txt'.format(FLAGS.tfrecord_dir, mode))
  gezi.write_to_txt(counter.value, out_file)

  print('mean words:', total_words.value / counter.value)

if __name__ == '__main__':
  tf.app.run()
