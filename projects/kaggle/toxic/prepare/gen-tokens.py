#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-vocab.py
#        \author   chenghuige  
#          \date   2018-01-13 17:50:26.382970
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("most_common", 100000000, "if > 0 then get vocab with most_common words")
flags.DEFINE_integer("min_count",  None, "if > 0 then cut by min_count")
flags.DEFINE_integer("max_lines", 0, "")
flags.DEFINE_boolean("write_unknown", True, "treat ignored words as unknow")
flags.DEFINE_string("out_dir", './mount/data/kaggle/toxic/tokens', "save count info to bin")
flags.DEFINE_string("vocab_name", None, "")

flags.DEFINE_string('input', '/home/gezi/data2/data/kaggle/toxic/train.csv', '')
flags.DEFINE_string('test_input', '/home/gezi/data2/data/kaggle/toxic/test.csv', '')

flags.DEFINE_integer('threads', 12, '')

flags.DEFINE_string('tokenizer_vocab', '/home/gezi/data/glove/glove-vocab.txt', '')
flags.DEFINE_integer('test_count', 0, '')
flags.DEFINE_bool('full_tokenizer', False, '')
flags.DEFINE_integer('limit', None, '')
flags.DEFINE_bool('modify_attribute', False, '')

flags.DEFINE_string('name', None, '')

import pandas as pd
import numpy as np

from gezi import WordCounter
import json
from tqdm import tqdm

import gezi

import tokenizer
import multiprocessing as mp
import six 
assert six.PY3

from tokenizer import attribute_names
from tokenizer import is_toxic

from textblob import Word

df = None
man = mp.Manager() 

name = 'train'

def tokenize(index):
  comments = df['comment_text']
  start, end = gezi.get_fold(len(comments), FLAGS.threads, index)
  
  #for i in tqdm(range(start, end)):
  with open('%s/%s_%d.txt' % (FLAGS.out_dir, name, index), 'w') as out:
    for i in range(start, end):
      if i % 1000 == 0:
        print(i, file=sys.stderr)
      sent = gezi.segment.tokenize_filter_empty(comments[i].replace('\n', ' '))
      print(' '.join(sent), file=out)
      

def run(input):
  global df 
  df = pd.read_csv(input)
  #df = df[:100]
  if FLAGS.limit:
    df = df[:FLAGS.limit]

  pool = mp.Pool()
  pool.map(tokenize, range(FLAGS.threads))
  pool.close()
  pool.join()

  #tokenize(0)

def main(_):
  tokenizer.init(FLAGS.tokenizer_vocab)
  os.system('mkdir -p %s' % FLAGS.out_dir)

  print('name', FLAGS.name, 'out_dir', FLAGS.out_dir)

  run(FLAGS.input)

  global name 
  name = 'test'
  if FLAGS.test_input and not FLAGS.name:
    run(FLAGS.test_input)

if __name__ == '__main__':
  tf.app.run()
