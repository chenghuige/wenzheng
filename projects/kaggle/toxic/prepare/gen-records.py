#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2018-01-14 11:50:06.092416
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import json 
import random

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', './mount/data/kaggle/toxic/train.csv', '') 
flags.DEFINE_string('vocab', './mount/temp/toxic/tfrecords/glove/vocab.txt', 'vocabulary txt file')
flags.DEFINE_integer('num_records', 10, '10 or 5?')
flags.DEFINE_string('tokenizer_vocab', '/home/gezi/data/glove/glove-vocab.txt', '')

import config

from gezi import Vocabulary
import gezi
import melt
import tokenizer

import multiprocessing
import pandas as pd 
from sklearn.utils import shuffle
import numpy as np

#import six
#assert six.PY3

from tqdm import tqdm

from tokenizer import attribute_names

from multiprocessing import Value
counter = Value('i', 0)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
examples = None
vocab = None
char_vocab = None

def get_id(word, vocab):
  for item in (word, word.lower(), word.capitalize(), word.upper()):
    if vocab.has(item):
      return vocab.id(item)
  return vocab.unk_id()

def get_char_id(ch, vocab):
  if vocab.has(ch):
    return vocab.id(ch)
  return vocab.unk_id()

def build_features(index):
  mode = 'train' if 'train' in FLAGS.input else 'test'
  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/{1}.record'.format(mode, index)
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('---out_file', out_file)
  # TODO now only gen one tfrecord file 

  total = len(examples)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)

  ids = examples['id'].values[start: end]
  comments = examples['comment_text'].values[start: end]
  
  try:
    labels = examples[CLASSES].values[start: end]
  except Exception:
    labels = [[0.] * len(CLASSES)] * len(ids)

  with melt.tfrecords.Writer(out_file) as writer:
    for id, comment, label in tqdm(zip(ids, comments, labels)):
      comment_str = comment
      # TODO use info
      doc = tokenizer.tokenize(comment)
      comment_tokens, tokens_info = doc.tokens, doc.attributes
      
      for i in range(len(tokens_info)):
        tokens_info[i] = list(map(float, tokens_info[i]))

      if FLAGS.comment_limit:
        comment_tokens = comment_tokens[: FLAGS.comment_limit]
        tokens_info = tokens_info[: FLAGS.comment_limit]

      tokens_info = np.array(tokens_info)
      tokens_info = tokens_info.reshape(-1)
      tokens_info = list(tokens_info)

      assert len(tokens_info) == len(comment_tokens) * len(attribute_names)

      comment_ids = [get_id(token, vocab) for token in comment_tokens]
      comment_tokens_str = '|'.join([vocab.key(id) for id in comment_ids])
      label = list(map(float, label))

      comment_chars = [list(token) for token in comment_tokens]

      char_ids = np.zeros([len(comment_ids), FLAGS.char_limit], dtype=np.int32)
      
      for i, token in enumerate(comment_chars):
        for j, ch in enumerate(token):
          if j == FLAGS.char_limit:
            break
          char_ids[i, j] = get_char_id(ch, char_vocab)

      char_ids = list(char_ids.reshape(-1))

      #print(char_ids)

      simple_char_ids = []
      num_chs = 0
      for ch in list(comment):
        id_ = get_char_id(ch, char_vocab)
        #if id_ == char_vocab.unk_id():
        #  continue
        simple_char_ids.append(id_)
        if len(simple_char_ids) == FLAGS.simple_char_limit:
          break

      simple_chars_str = ''.join([char_vocab.key(id) for id in simple_char_ids])

      #print(simple_char_ids, simple_chars_str)

      record = tf.train.Example(features=tf.train.Features(feature={
                                "comment": melt.int64_feature(comment_ids),
                                "tokens_info": melt.float_feature(tokens_info),
                                "comment_chars": melt.int64_feature(char_ids),
                                "simple_chars": melt.int64_feature(simple_char_ids),
                                "simple_chars_str": melt.bytes_feature(simple_chars_str),
                                "classes": melt.float_feature(label),
                                "id": melt.bytes_feature(id),
                                "comment_str": melt.bytes_feature(comment_str),
                                "comment_tokens_str": melt.bytes_feature(comment_tokens_str)
                                }))
      
      writer.write(record)
      global counter
      with counter.get_lock():
        counter.value += 1

    print("Build {} instances of features in total".format(writer.size()))
    writer.close()

def main(_):  
  tokenizer.init(FLAGS.tokenizer_vocab)
  global examples, vocab, char_vocab
  examples = pd.read_csv(FLAGS.input)
  #if 'train' in FLAGS.input:
  #  examples = shuffle(examples, random_state=1024)
  vocab = Vocabulary(FLAGS.vocab)
  char_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt'))

  pool = multiprocessing.Pool()
  pool.map(build_features, range(FLAGS.num_records))
  pool.close()
  pool.join()

  # build_features(0)

  print('num_records:', counter.value)
  mode = 'train' if 'train' in FLAGS.input else 'test'
  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  tf.app.run()

