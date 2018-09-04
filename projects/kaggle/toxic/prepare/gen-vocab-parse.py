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
flags.DEFINE_integer("max_ngrams", None, "")
flags.DEFINE_integer("max_lines", 0, "")
flags.DEFINE_boolean("write_unknown", True, "treat ignored words as unknow")
flags.DEFINE_string("out_dir", './mount/temp/toxic/tfrecords/', "save count info to bin")
flags.DEFINE_string("vocab_name", None, "")

flags.DEFINE_string('input', '/home/gezi/data2/data/kaggle/toxic/train.csv', '')
flags.DEFINE_string('test_input', '/home/gezi/data2/data/kaggle/toxic/test.csv', '')

flags.DEFINE_integer('threads', 12, '')

flags.DEFINE_string('tokenizer_vocab', '/home/gezi/data/glove/glove-vocab.txt', '')
flags.DEFINE_integer('test_count', 0, '')
flags.DEFINE_bool('full_tokenizer', False, '')
# flags.DEFINE_bool('simple_tokenizer', False, '')
flags.DEFINE_integer('limit', None, '')
flags.DEFINE_bool('modify_attribute', False, '')

flags.DEFINE_bool('lower', False, 'if lower then word lower')
flags.DEFINE_bool('ngram_lower', False, 'if lower then ngram lower')
flags.DEFINE_integer('ngram_min', 3, '')
flags.DEFINE_integer('ngram_max', 3, '')

flags.DEFINE_bool('write_tokens', True, '')
flags.DEFINE_bool('write_csv', True, '')

flags.DEFINE_bool('lemmatization', False, '')

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

counter = None
char_counter = None 
ngram_counter = None

pos_counter = None
tag_counter = None 
ner_counter = None

START_WORD = '<S>'
END_WORD = '</S>'

df = None
man = mp.Manager()
tokens_list = None
attributes_list = None
ori_tokens_list = None
poses_list = None
tags_list = None
ners_list = None

def tokenize(index):
  global tokens_list
  comments = df['comment_text']
  start, end = gezi.get_fold(len(comments), FLAGS.threads, index)

  if 'tokens' in df.columns:
    for i in range(start, end):
      # if df['id'][i] == '5bbabc3b14cc1f7f':
      #   sent = tokenizer.full_tokenize(comments[i])
      #   tokens_list[i] = sent.tokens
      #   attributes_list[i] = np.reshape(np.array([list(map(float, x)) for  x in sent.attributes]), -1)
      #   poses_list[i] = sent.poses
      #   tags_list[i] = sent.tags
      #   ners_list[i] = sent.ners
      # else:
      tokens_list[i] = df['tokens'][i].split(' ')
      attributes_list[i] = df['attributes'][i].split(' ')
      # if len(attributes_list[i]) != len(attribute_names) * len(tokens_list[i]) or FLAGS.modify_attribute:
      #   sent = tokenizer.tokenize(comments[i])
      #   attributes_list[i] = np.reshape(np.array([list(map(float, x)) for  x in sent.attributes]), -1)
      #   assert len(attributes_list[i]) == len(attribute_names) * len(tokens_list[i]), '{} {} {} {}'.format(len(attributes_list[i]) / len(attribute_names), len(tokens_list[i]), i, df['id'][i])
      poses_list[i] = df['poses'][i].split(' ')
      tags_list[i] = df['tags'][i].split(' ')
      ners_list[i] = df['ners'][i].split(' ') 
      ori_tokens_list[i] = df['ori_tokens'][i].split(' ')    
  else:
    for i in tqdm(range(start, end)):
    # for i in range(start, end):
    #   if i % 1000 == 0:
    #     print(i, file=sys.stderr)
      if FLAGS.full_tokenizer:
        #if FLAGS.simple_tokenizer:
        sent = tokenizer.full_tokenize(comments[i], lemmatization=FLAGS.lemmatization)
        # else:
        #   sent = gezi.segment.tokenize_filter_empty(comments[i])
        # if FLAGS.lower:
        #   sent.tokens = [w.lower() for w in sent.tokens]
        tokens_list[i] = sent.tokens
        ori_tokens_list[i] = sent.ori_tokens
        attributes_list[i] = np.reshape(np.array([list(map(float, x)) for  x in sent.attributes]), -1)
        poses_list[i] = sent.poses
        tags_list[i] = sent.tags
        ners_list[i] = sent.ners
      else:
        sent = tokenizer.tokenize(comments[i], lemmatization=FLAGS.lemmatization)
        # if FLAGS.lower:
        #   sent.tokens = [w.lower() for w in sent.tokens]
        tokens_list[i] = sent.tokens
        ori_tokens_list[i] = sent.ori_tokens       

        #print('----------', sent.attributes)
        try:
          attributes_list[i] = np.reshape(np.array([list(map(float, x)) for  x in sent.attributes]), -1)
        except Exception:
          print(sent.attributes)
          raise ValueError()
        poses_list[i] = ['NONE'] * len(tokens_list[i])
        tags_list[i] = ['NONE'] * len(tokens_list[i])
        ners_list[i] = ['NONE'] * len(tokens_list[i])
        

def run(input, count=1):
  global df, tokens_list, poses_list, tags_list, ners_list 
  global attributes_list, ori_tokens_list 
  df = pd.read_csv(input)
  #df = df[:100]
  if FLAGS.limit:
    df = df[:FLAGS.limit]
  tokens_list = man.list([None] * len(df['comment_text']))
  attributes_list = man.list([None] * len(df['comment_text']))
  poses_list = man.list([None] * len(df['comment_text']))
  tags_list = man.list([None] * len(df['comment_text']))
  ners_list = man.list([None] * len(df['comment_text']))
  ori_tokens_list = man.list([None] * len(df['comment_text']))

  timer = gezi.Timer('tokenize')

  pool = mp.Pool()
  pool.map(tokenize, range(FLAGS.threads))
  pool.close()
  pool.join()

  #tokenize(0)

  timer.print_elapsed()

  # for context in tqdm(df['comment_text']):
    #context_tokens, _ = tokenizer.tokenize(context)
    #context_tokens = gezi.segment.tokenize_filter_empty(context)
  for context_tokens in tokens_list:
    counter.add(START_WORD, count)
    # tokens in one comment treat as 1
    for token in set(context_tokens):
      if FLAGS.lower:
        token = token.lower()
      counter.add(token, count)
    counter.add(END_WORD, count)
  
  for context_tokens in ori_tokens_list:
    # tokens in one comment treat as 1
    for token in set(context_tokens):
      for ch in token:
        char_counter.add(ch, count)
      if FLAGS.ngram_lower:
        token = token.lower()
      ngrams = gezi.get_ngrams(token, FLAGS.ngram_min, FLAGS.ngram_max)
      for ngram in ngrams:
        ngram_counter.add(ngram, count)
  
  for poses in poses_list:
    for pos in set(poses):
      pos_counter.add(pos, count)
  
  for tags in tags_list:
    for tag in set(tags):
      tag_counter.add(tag, count)

  for ners in ners_list:
    for ner in set(ners):
      ner_counter.add(ner, count)

  tokens = [' '.join(x) for x in tokens_list]
  attributes = [' '.join(map(str, x)) for x in attributes_list]
  poses = [' '.join(x) for x in poses_list]
  tags = [' '.join(x) for x in tags_list]
  ners = [' '.join(x) for x in ners_list]
  ori_tokens = [' '.join(x) for x in ori_tokens_list]

  df['tokens'] = tokens
  df['attributes'] = attributes
  # TODO change to poses..
  df['poses'] = poses
  df['tags'] = tags
  df['ners'] = ners
  df['ori_tokens'] = ori_tokens

  if FLAGS.write_csv:
    if FLAGS.name:
      name = FLAGS.name 
    else:
      name = 'train.csv' if 'train' in input else 'test.csv'
    out_csv = os.path.join(FLAGS.out_dir, name)
    print('save csv to', out_csv)
    df.to_csv(out_csv, index=False)

  if FLAGS.write_tokens:
    if FLAGS.name:
      name = FLAGS.name.replace('.csv', '_tokens.txt') 
    else:
      name ='train_tokens.txt' if 'train' in input else 'test_tokens.txt'

    out_txt = os.path.join(FLAGS.out_dir, name)
    df.sort_values(['id'], inplace=True)
    df = df.reset_index(drop=True)
    with open(out_txt, 'w') as out:
      for id, comment, tokens in zip(df['id'].values, df['comment_text'].values, df['tokens'].values):
        print(id, comment, tokens, sep='\t', file=out)

def main(_):
  tokenizer.init(FLAGS.tokenizer_vocab)
  if FLAGS.full_tokenizer:
    gezi.segment.init_spacy_full()

  os.system('mkdir -p %s' % FLAGS.out_dir)

  print('name', FLAGS.name, 'out_dir', FLAGS.out_dir)

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

  global ngram_counter 
  ngram_counter = WordCounter(
     write_unknown=True,
     min_count=FLAGS.min_count)

  global pos_counter, tag_counter, ner_counter
  pos_counter = WordCounter(
    write_unknown=True,
    min_count=1)
  tag_counter = WordCounter(
    write_unknown=True,
    min_count=1)
  ner_counter = WordCounter(
    write_unknown=True,
    min_count=1) 

  run(FLAGS.input)

  if FLAGS.test_input and not FLAGS.name:
    run(FLAGS.test_input, count=FLAGS.test_count)
  
  if not FLAGS.name:
    vocab_name = FLAGS.vocab_name or 'vocab'
    os.system('mkdir -p %s' % FLAGS.out_dir)
    out_txt = os.path.join(FLAGS.out_dir, '%s.txt' % vocab_name)
    counter.save(out_txt)

    out_txt = os.path.join(FLAGS.out_dir, 'char_%s.txt' % vocab_name)
    char_counter.save(out_txt)

    out_txt = os.path.join(FLAGS.out_dir, 'pos_vocab.txt')
    pos_counter.save(out_txt)

    out_txt = os.path.join(FLAGS.out_dir, 'tag_vocab.txt')
    tag_counter.save(out_txt)

    out_txt = os.path.join(FLAGS.out_dir, 'ner_vocab.txt')
    ner_counter.save(out_txt)

    out_txt = os.path.join(FLAGS.out_dir, 'ngram_vocab.txt')
    if not FLAGS.max_ngrams:
      ngram_counter.save(out_txt)
    else:
      # if later need most 2w ngram head -200000 ngram_vocab.full.txt > ngram_vocab.txt
      out_full_txt = os.path.join(FLAGS.out_dir, 'ngram_vocab.full.txt')
      ngram_counter.save(out_full_txt)
      os.system('head -n %d %s > %s' % (FLAGS.max_ngrams, out_full_txt, out_txt))

if __name__ == '__main__':
  tf.app.run()
