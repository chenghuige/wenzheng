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
flags.DEFINE_string("out_dir", './mount/temp/toxic/tfrecords/', "save count info to bin")
flags.DEFINE_string("vocab_name", None, "")

flags.DEFINE_string('input', '/home/gezi/data2/data/kaggle/toxic/train.csv', '')
flags.DEFINE_string('test_input', '/home/gezi/data2/data/kaggle/toxic/test.csv', '')

flags.DEFINE_integer('threads', 12, '')

flags.DEFINE_string('tokenizer_vocab', '/home/gezi/data/glove/glove-vocab.txt', '')
flags.DEFINE_integer('test_count', 0, '')
flags.DEFINE_bool('special_tokenizer', False, '')

import pandas as pd

from gezi import WordCounter
import json
from tqdm import tqdm

import gezi

import tokenizer
import multiprocessing as mp
import six 
from preprocess import *

assert six.PY3

counter = None
char_counter = None

START_WORD = '<S>'
END_WORD = '</S>'

df = None
man = mp.Manager()
context_tokens_list = None

def tokenize(index):
  global context_tokens_list
  comments = df['comment_text']
  start, end = gezi.get_fold(len(comments), FLAGS.threads, index)
  for i in tqdm(range(start, end), ascii=True):
    comment = comments[i]
    if FLAGS.special_tokenizer:
      context_tokens_list[i] = tokenizer.tokenize(comment).tokens
    else:
      if FLAGS.is_twitter:
        comment = glove_twitter_preprocess(comment)
      context_tokens_list[i] = [x.lower() for x in gezi.segment.tokenize_filter_empty(comment)]

def run(input, count=1):
  global df, context_tokens_list
  df = pd.read_csv(input)
  #df = df[:100]
  context_tokens_list = man.list([None] * len(df['comment_text']))

  timer = gezi.Timer('tokenize')

  pool = mp.Pool()
  pool.map(tokenize, range(FLAGS.threads))
  pool.close()
  pool.join()

  timer.print_elapsed()

  # for context in tqdm(df['comment_text']):
    #context_tokens, _ = tokenizer.tokenize(context)
    #context_tokens = gezi.segment.tokenize_filter_empty(context)
  for context_tokens in context_tokens_list:
    counter.add(START_WORD, count)
    # tokens in one comment treat as 1
    for token in set(context_tokens):
      counter.add(token, count)
      for ch in token:
        char_counter.add(ch, count)
    counter.add(END_WORD, count)

def main(_):
  tokenizer.init(FLAGS.tokenizer_vocab)
  global counter
  counter = WordCounter(
    write_unknown=FLAGS.write_unknown,
    most_common=FLAGS.most_common,
    min_count=FLAGS.min_count)

  global char_counter
  char_counter = WordCounter(
    write_unknown=FLAGS.write_unknown,
    most_common=FLAGS.most_common,
    min_count=FLAGS.min_count)

  run(FLAGS.input)
  if FLAGS.test_input:
    run(FLAGS.test_input, count=FLAGS.test_count)
  
  vocab_name = FLAGS.vocab_name or 'vocab'
  os.system('mkdir -p %s' % FLAGS.out_dir)
  out_txt = os.path.join(FLAGS.out_dir, '%s.txt' % vocab_name)
  counter.save(out_txt)

  out_txt = os.path.join(FLAGS.out_dir, 'char_%s.txt' % vocab_name)
  char_counter.save(out_txt)


if __name__ == '__main__':
  tf.app.run()
