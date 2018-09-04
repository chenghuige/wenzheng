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

flags.DEFINE_string('dir', None, '') 
flags.DEFINE_string('input', None, '') 
flags.DEFINE_string('vocab', None, 'vocabulary txt file')
flags.DEFINE_integer('num_records', 10, '10 or 5?')
flags.DEFINE_string('tokenizer_vocab', '/home/gezi/data/glove/glove-vocab.txt', '')
flags.DEFINE_string('mode_', None, '')

flags.DEFINE_bool('lower', False, 'if lower then word lower')
flags.DEFINE_bool('ngram_lower', False, 'if lower then ngram lower')
flags.DEFINE_integer('ngram_min', 3, '')
flags.DEFINE_integer('ngram_max', 3, '')

flags.DEFINE_float('weight', 1., '')

flags.DEFINE_bool('has_dup', False, '')

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

from multiprocessing import Value

from tokenizer import attribute_names

counter = Value('i', 0)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
examples = None
vocab = None
unk_vocab = None
char_vocab = None

pos_vocab = None 
tag_vocab = None 
ner_vocab = None

ngram_vocab = None

enprob_dict = None

import copy

def get_id(word, vocab):
  for item in (word, word.lower(), word.capitalize(), word.upper()):
    if vocab.has(item):
      return vocab.id(item)
  return vocab.unk_id()

def get_char_id(ch, vocab):
  if vocab.has(ch):
    return vocab.id(ch)
  return vocab.unk_id()

def get_ngram_id(ngram, vocab):
  for item in (ngram, ngram.lower(), ngram.capitalize(), ngram.upper()):
    if vocab.has(item):
      return vocab.id(item)
  return vocab.unk_id()  

def get_mode():
  if FLAGS.mode_:
    return FLAGS.mode_ 
  if not FLAGS.has_dup:
    return 'train' if 'train' in FLAGS.input else 'test'
  else:
    return 'train.sents' if 'train' in FLAGS.input else 'test.sents'


def get_fold(ids, index):
  ids_ = []
  ids = list(ids)
  ids_set = set()
  for id in ids:
    if id not in ids_set:
      ids_.append(id)
      ids_set.add(id)
  start_, end_ = gezi.get_fold(len(ids_), FLAGS.num_records, index)
  
  ids.append('END')
  ids_.append('END')
    
  start = None 
  end = None 
  for i in range(len(ids)):
    if ids[i] == ids_[start_]:
      start = i
    elif ids[i] == ids_[end_]:
      end = i
      return start, end

def build_features(index):
  mode = get_mode()
  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/{1}.record'.format(mode, index)
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('---out_file', out_file)
  # TODO now only gen one tfrecord file 

  total = len(examples)
  if not FLAGS.has_dup:
    start, end = gezi.get_fold(total, FLAGS.num_records, index)
  else:
    start, end = get_fold(examples['id'].values, index)

  ids = examples['id'].values[start: end]
  ids = list(map(str, ids))
  comments = examples['comment_text'].values[start: end]
  tokens_list = examples['tokens'].values[start: end]
  tokens_infos = examples['attributes'].values[start: end]
  # TODO change to poses
  poses = examples['poses'].values[start: end]
  tags = examples['tags'].values[start: end]
  ners = examples['ners'].values[start: end]
  ori_tokens_list = examples['ori_tokens'].values[start: end]
  
  try:
    labels = examples[CLASSES].values[start: end]
  except Exception:
    labels = [[0.] * len(CLASSES)] * len(ids)

  with melt.tfrecords.Writer(out_file) as writer:
    for id, comment, label, comment_tokens, ori_tokens, tokens_info, pos, tag, ner in tqdm(zip(ids, comments, labels, tokens_list, ori_tokens_list, tokens_infos, poses, tags, ners)):
      if not isinstance(comment, str):
        comment = 'ok'
      comment_str = comment

      comment_tokens = comment_tokens.split(' ')
      tokens_info = tokens_info.split(' ')
      pos = pos.split(' ')
      tag = tag.split(' ')
      ner = ner.split(' ')
      ori_tokens = ori_tokens.split(' ')

      if FLAGS.comment_limit:
        comment_tokens = comment_tokens[:FLAGS.comment_limit]
        ori_tokens = ori_tokens[:FLAGS.comment_limit]
        tokens_info = tokens_info[:len(attribute_names) * FLAGS.comment_limit]

      pos_ids = [get_char_id(x, pos_vocab) for x in pos]
      tag_ids = [get_char_id(x, tag_vocab) for x in tag]
      ner_ids = [get_char_id(x, ner_vocab) for x in ner]

      # NOTICE comment_ids with vocab(all train + test word so no unk)
      if not FLAGS.lower:
        comment_ids = [get_id(token, vocab) for token in comment_tokens]
        #comment_ids_withunk = [get_id(token, unk_vocab) for token in comment_tokens]
      else:
        comment_ids = [get_id(token.lower(), vocab) for token in comment_tokens]
        #comment_ids_withunk = [get_id(token.lower(), unk_vocab) for token in comment_tokens]

      comment_tokens_str = '|'.join([vocab.key(id) for id in comment_ids])
      label = list(map(float, label))

      tokens_info = list(map(float, tokens_info))

      #print(len(comment_ids), len(tokens_info) / len(attribute_names), len(tokens_info) / len(comment_ids))
      assert len(tokens_info) == len(attribute_names) * len(comment_ids), '%d %f' %(len(comment_ids), len(tokens_info) / len(attribute_names))


      #comment_chars = [list(token) for token in comment_tokens]
      ## CHANGE to use ori token so fu**ck will encode ** but  NiggerMan to Nigger Man will all encode NiggerMan NiggerMan twice
      chars_list = [list(token) for token in ori_tokens]
      char_ids = np.zeros([len(comment_ids), FLAGS.char_limit], dtype=np.int32)
      assert len(comment_ids) == len(chars_list), '{} {} {} {} {}'.format((len(comment_ids), len(chars_list), comment), tokens, ori_tokens)
      
      for i, chars in enumerate(chars_list):
        for j, ch in enumerate(chars):
          if j == FLAGS.char_limit:
            break
          char_ids[i, j] = get_char_id(ch, char_vocab)

      char_ids = list(char_ids.reshape(-1))

      #print(char_ids)

      # --------------simple char
      simple_char_ids = []
      for ch in list(comment):
        id_ = get_char_id(ch, char_vocab)
        #if id_ == char_vocab.unk_id():
        #  continue
        simple_char_ids.append(id_)
        if len(simple_char_ids) == FLAGS.simple_char_limit:
          break

      simple_chars_str = ''.join([char_vocab.key(id) for id in simple_char_ids])
      #print(simple_char_ids, simple_chars_str)

      # # --------------simple ngram
      # simple_ngrams = gezi.get_ngrams(comment)
      # simple_ngrams = simple_ngrams[:FLAGS.simple_char_limit * 5]
      # simple_ngram_ids = [get_ngram_id(ngram, ngram_vocab) for ngram in simple_ngrams]

      # --------------ngram
      ngram_ids_list = np.zeros([len(comment_ids), FLAGS.char_limit], dtype=np.int32)
      if not FLAGS.ftngram:
        #ngrams_list = [gezi.get_ngrams(token) for token in ori_tokens]
        if not FLAGS.ngram_lower:
          ngrams_list = [gezi.get_ngrams(token, FLAGS.ngram_min, FLAGS.ngram_max) for token in comment_tokens]
        else:
          ngrams_list = [gezi.get_ngrams(token.lower(), FLAGS.ngram_min, FLAGS.ngram_max) for token in comment_tokens]

        for i, ngrams in enumerate(ngrams_list):
          for j, ngram in enumerate(ngrams):
            if j == FLAGS.char_limit:
              break
            #assert get_ngram_id(ngram, ngram_vocab) < 20003
            ngram_ids_list[i, j] = get_ngram_id(ngram, ngram_vocab)
      else:
        #for i, (token, ori_token) in enumerate(zip(comment_tokens, ori_tokens)):
        for i, (token, ori_token) in enumerate(zip(comment_tokens, comment_tokens)):
          ngram_ids = gezi.fasttext_ids(ori_token, vocab, FLAGS.ngram_buckets, FLAGS.ngram_min, FLAGS.ngram_max)
          if len(ngram_ids) >= FLAGS.char_limit:
            ngram_ids = gezi.fasttext_ids(token, vocab, FLAGS.ngram_buckets, FLAGS.ngram_min, FALGS.ngram_max)
          ngram_ids = ngram_ids[:FLAGS.char_limit]
          for j, ngram_id in enumerate(ngram_ids):
            ngram_ids_list[i, j] = ngram_id

      ngram_ids = list(ngram_ids_list.reshape(-1))

      # # ---------------fngrams(full ngrams)
      # fngrams_list = [gezi.get_ngrams_hash(token, FLAGS.ngram_buckets, 3, 6, reserve=3) for token in ori_tokens]
      # fngram_ids =  np.zeros([len(comment_ids), FLAGS.ngram_limit], dtype=np.int32)
      # for i, fngrams in enumerate(fngrams_list):
      #   for j, fngram in enumerate(fngrams):
      #     if j == FLAGS.ngram_limit:
      #       break
      #     fngram_ids[i, j] = fngram
      # fngram_ids = list(fngram_ids.reshape(-1))

      # global info per comment  7 features
      comment_info = []
      comment_info.append(len(ori_tokens))
      comment_info.append(len(comment_tokens))
      #comment_len = sum[len(x) for x in ori_tokens]
      comment_len = len(comment_str)
      comment_info.append(comment_len)
      comment_info.append(comment_len / (len(ori_tokens) + 1))
      num_unks = len([x for x in comment_ids if x == vocab.unk_id()])
      comment_info.append(num_unks)
      comment_info.append(num_unks / len(comment_tokens))
      comment_info.append(enprob_dict[id])

      record = tf.train.Example(features=tf.train.Features(feature={
                                "comment": melt.int64_feature(comment_ids),
                                #"comment_withunk": melt.int64_feature(comment_ids_withunk),
                                "tokens_info": melt.float_feature(tokens_info),
                                "comment_info": melt.float_feature(comment_info),
                                "pos": melt.int64_feature(pos_ids),
                                "tag": melt.int64_feature(tag_ids),
                                "ner": melt.int64_feature(ner_ids),
                                "comment_chars": melt.int64_feature(char_ids),
                                "comment_ngrams": melt.int64_feature(ngram_ids),
                                "simple_chars": melt.int64_feature(simple_char_ids),
                                #"simple_ngrams": melt.int64_feature(simple_ngram_ids),
                                #"comment_fngrams": melt.int64_feature(fngram_ids),
                                #"simple_chars_str": melt.bytes_feature(simple_chars_str),
                                "classes": melt.float_feature(label),
                                "id": melt.bytes_feature(id),
                                "weight": melt.float_feature([FLAGS.weight]),
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
  os.system('mkdir -p %s' % FLAGS.dir)
  tokenizer.init(FLAGS.tokenizer_vocab)
  global examples, vocab, unk_vocab, char_vocab, pos_vocab, tag_vocab, ner_vocab, ngram_vocab
  examples = pd.read_csv(FLAGS.input)
  #if 'train' in FLAGS.input:
  #  examples = shuffle(examples, random_state=1024)
  vocab = Vocabulary(FLAGS.vocab)
  # unk_vocab is actually a small vocab so will genearte unk for training
  #unk_vocab =  Vocabulary(FLAGS.vocab.replace('vocab.txt', 'unk_vocab.txt'))
  char_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt'))
  pos_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'pos_vocab.txt'))
  tag_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'tag_vocab.txt'))
  ner_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'ner_vocab.txt'))
  ngram_vocab = Vocabulary(FLAGS.vocab.replace('vocab.txt', 'ngram_vocab.txt'))

  global enprob_dict
  enprob_dict = {}
  enprob_file = '~/data/kaggle/toxic/train.enprob.csv' if 'train' in FLAGS.input else '~/data/kaggle/toxic/test.enprob.csv'
  enprob_df = pd.read_csv(enprob_file)
  for id, enprob in zip(enprob_df['id'].values, enprob_df['enprob'].values):
    enprob_dict[id] = enprob
  enprob_dict['0'] = 1.

  pool = multiprocessing.Pool()
  pool.map(build_features, range(FLAGS.num_records))
  pool.close()
  pool.join()

  #build_features(0)

  print('num_records:', counter.value)
  mode = get_mode()
  out_file = os.path.dirname(FLAGS.vocab) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  tf.app.run()

