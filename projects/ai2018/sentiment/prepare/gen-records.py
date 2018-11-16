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
      num_records = 7
  #print('------------------', num_records, FLAGS.num_records_)
  start, end = gezi.get_fold(total, num_records, index)

  print('total', total, 'infile', FLAGS.input, 'out_file', out_file)

  max_len = 0
  max_num_ids = 0
  num = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for i in tqdm(range(start, end), ascii=True):
      try:
        row = df.iloc[i]
        id = str(row[0])

        if seg_result:
          if id not in seg_result:
            print('id %s ot found in seg_result' % id)
            continue
          words = seg_result[id]

          if FLAGS.content_limit_:
            # NOW only for bert!
            if len(words) + 2 > FLAGS.content_limit_:
              words = words[:FLAGS.content_limit_ - 3 - 50] + ['[MASK]'] + words[-50:]
              #print(words)
          if FLAGS.add_start_end_:
            words = gezi.add_start_end(words, FLAGS.start_mark, FLAGS.end_mark)
        if pos_result:
          pos = pos_result[id]
          if FLAGS.add_start_end_:
            pos = gezi.add_start_end(pos)
        if ner_result:
          ner = ner_result[id]
          if FLAGS.add_start_end_:
            ner = gezi.add_start_end(ner)

        if start_index > 0:
          id == 't' + id
  
        content = row[1] 
        content_ori = content
        content = filter.filter(content)

        if not FLAGS.use_soft_label_:
          if 'test' in mode:
            label = [-2] * 20
          else:
            label = list(row[2:])
          
          #label = [x + 2 for x in label]
          #num_labels = len(label)
        else:
          label = [0.] * 80
          if not FLAGS.is_soft_label:
            for idx, val in enumerate(row[2:]):
              label[idx * 4 + val] = 1.
          else:
            logits = np.array(gezi.str2scores(row['score']))
            logits = np.reshape(logits, [20, 4])
            probs = gezi.softmax(logits)
            label = list(np.reshape(probs, [-1]))

        if not seg_result:
          content_ids, words = text2ids_(content, preprocess=False, return_words=True)
          assert len(content_ids) == len(words)
        else:
          content_ids = [vocab.id(x) for x in words]
          #print(words, content_ids)
          #exit(0)

        if len(content_ids) > max_len:
          max_len = len(content_ids)
          print('max_len', max_len)

        if len(content_ids) > FLAGS.word_limit and len(content_ids) < 5:
          print('{} {} {}'.format(id, len(content_ids), content_ori))
        #if len(content_ids) > FLAGS.word_limit:
        #  print(id, content)
        #  if mode not in ['test', 'valid']:
        #    continue 

        #if len(content_ids) < 5 and mode not in ['test', 'valid']:
        #  continue

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

        if pos_vocab:
          assert pos
          pos = pos[:FLAGS.word_limit]
          pos_ids = [pos_vocab.id(x) for x in pos]
        else:
          pos_ids = [0]

        if ner_vocab:
          assert ner 
          if pos_vocab:
            assert len(pos) == len(ner)         
          ner = ner[:FLAGS.word_limit]

          ner_ids = [ner_vocab.id(x) for x in ner]
        else:
          ner_ids = [0]

        wlen = [len(word) for word in words]

        feature = {
                    'id': melt.bytes_feature(id),
                    'content':  melt.int64_feature(content_ids),
                    'content_str': melt.bytes_feature(content_ori), 
                    'char': melt.int64_feature(char_ids),
                    'pos': melt.int64_feature(pos_ids), # might also be postion info for mix seg
                    'ner': melt.int64_feature(ner_ids),
                    'wlen': melt.int64_feature(wlen),
                    'source': melt.bytes_feature(mode), 
                  }
        feature['label'] = melt.int64_feature(label) if not FLAGS.use_soft_label_ else melt.float_feature(label)

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

  assert FLAGS.use_fold
  #text2ids.init(FLAGS.vocab_)
  global vocab, char_vocab, pos_vocab, ner_vocab, seg_result, pos_result, ner_result
  #vocab = text2ids.vocab
  vocab = gezi.Vocabulary(FLAGS.vocab_, fixed=FLAGS.fixed_vocab, unk_word=FLAGS.unk_word)
  print('vocab size:', vocab.size())
  char_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'char_vocab.txt')
  if os.path.exists(char_vocab_file):
    char_vocab = Vocabulary(char_vocab_file)
    print('char vocab size:', char_vocab.size())
  pos_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'pos_vocab.txt')
  if os.path.exists(pos_vocab_file):
    pos_vocab = Vocabulary(pos_vocab_file)
    print('pos vocab size:', pos_vocab.size())
  ner_vocab_file = FLAGS.vocab_.replace('vocab.txt', 'ner_vocab.txt')
  if os.path.exists(ner_vocab_file):
    ner_vocab = Vocabulary(ner_vocab_file)
    print('ner vocab size:', ner_vocab.size())
  
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

  seg_file = FLAGS.vocab_.replace('vocab.txt', '%s.seg.txt' % mode_)
  seg_result = {}
  if os.path.exists(seg_file):
    print('seg or seg_pos exits:', seg_file)
    pos_result = {}
    for line in open(seg_file):
      id, segs = line.rstrip('\n').split('\t', 1)
      segs = segs.split('\x09')
      if FLAGS.ignore_start_end:
        segs = segs[1:-1]
      if '|' in segs[0] and not FLAGS.word_only:
        l = [x.rsplit('|', 1) for x in segs]
        segs, pos = list(zip(*l))
        pos_result[id] = pos
      seg_result[id] = segs

  seg_done = True if seg_result else False
  ner_file = FLAGS.vocab_.replace('vocab.txt', '%s.ner.txt' % mode_)
  ner_result = {}
  if os.path.exists(ner_file):
    print('seg_ner exists:', ner_file)
    for line in open(ner_file):
      id, segs = line.rstrip('\n').split('\t', 1)
      segs = segs.split('\x09')
      if FLAGS.ignore_start_end:
        segs = segs[1:-1]
      if '|' in segs[0]:
        l = [x.rsplit('|', 1) for x in segs]
        segs, ner = list(zip(*l))
      if not seg_done:      
        seg_result[id] = segs
      ner_result[id] = ner

  print('len(seg_result)', len(seg_result))
  print('len(ner_result)', len(ner_result))

  # print('to_lower:', FLAGS.to_lower, 'feed_single:', FLAGS.feed_single, 'feed_single_en:', FLAGS.feed_single_en, 'seg_method', FLAGS.seg_method)
  # print(text2ids.ids2text(text2ids_('傻逼脑残B')))
  # print(text2ids.ids2text(text2ids_('喜欢玩孙尚香的加我好友：2948291976')))

  global df
  df = pd.read_csv(FLAGS.input, lineterminator='\n')
  
  pool = multiprocessing.Pool()

  if not FLAGS.num_records_:
    if mode.split('.')[-1] in ['valid', 'test', 'dev', 'pm'] or 'valid' in FLAGS.input:
      FLAGS.num_records_ = 1
    else:
      FLAGS.num_records_ = 7

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
