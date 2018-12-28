#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2018-02-05 20:05:25.123740
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', './mount/temp/ai2018/sentiment/tfrecord/valid/*record,', '')
flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('type', 'debug', '')
flags.DEFINE_string('base', './mount/temp/ai2018/sentiment/tfrecord/', '')
#flags.DEFINE_integer('unk_id', 1, '')
#flags.DEFINE_integer('fold', None, '') 
flags.DEFINE_bool('dump_valid', True, '')
flags.DEFINE_bool('dump_test', True, '')

import tensorflow as tf
tf.enable_eager_execution()

import sys, os
from sklearn import metrics
import pandas as pd 
import numpy as np
import gezi


import pickle

from wenzheng.utils import ids2text

import melt
logging = melt.logging
from dataset import Dataset

from tqdm import tqdm

# TODO by default save all ? so do not need to change the code ? 
# _asdict() https://stackoverflow.com/questions/26180528/python-named-tuple-to-dictionary
# err... valid and test data share same id...
def deal(dataset, infos):
  for x, _ in tqdm(dataset, ascii=True):
    for key in x:
      x[key] = x[key].numpy()
      if type(x[key][0]) == bytes:
        x[key] = gezi.decode(x[key])
    ids = x['id']
    for j in range(len(ids)):
      infos[ids[j]] = {}
      for key in x:
        infos[ids[j]][key] = x[key][j]

def main(_):

  base = FLAGS.base
  logging.set_logging_path('./mount/tmp/')
  vocab_path = f'{base}/vocab.txt'
  ids2text.init(vocab_path)
  FLAGS.vocab = f'{base}/vocab.txt'

  # FLAGS.length_index = 2
  # FLAGS.buckets = '100,400'
  # FLAGS.batch_sizes = '64,64,32'

  input_ = FLAGS.input 
  if FLAGS.type == 'test':
    input_ = input_.replace('valid', 'test')

  inputs = gezi.list_files(input_)
  inputs.sort()
  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  if FLAGS.type == 'debug':
    print('type', FLAGS.type, 'inputs', inputs, file=sys.stderr)

    dataset = Dataset('valid')
    dataset = dataset.make_batch(FLAGS.batch_size_, inputs)

    print('dataset', dataset)

    timer = gezi.Timer('read record')
    for i, (x, y) in enumerate(dataset):
      # if i % 10 == 1:
      #   print(x['id'])
      #   print(x['content'][0])
      #   print(ids2text.ids2text(x['content'][0], sep='|'))
      #   print(x['content'])
      #   print(type(x['id'].numpy()[0]) == bytes)
      #   break
      x['id'] = gezi.decode(x['id'].numpy())
      x['content_str'] = gezi.decode(x['content_str'].numpy())
      for j, id in enumerate(x['id']):
        if id == '573':
          print(id, x['content_str'][j])
  elif FLAGS.type == 'dump':
    valid_infos = {}
    test_infos = {}
    # TODO notice train and valid also share ids.. so valid only save 0 is ok...
    # 120000 doc but first 15000 train duplicate id with valid so only save valid result for those ids currently 
    if FLAGS.dump_valid:
      inputs = gezi.list_files(f'{base}/train/*record')
      dataset = Dataset('valid')
      dataset = dataset.make_batch(1, inputs)
      deal(dataset, valid_infos)
      print('after valid', len(valid_infos))

      for key in valid_infos:
        print(valid_infos[key])
        print(ids2text.ids2text(valid_infos[key]['content']))
        break

      ofile = f'{base}/info.pkl'
      with open(ofile, 'wb') as out:
        pickle.dump(valid_infos, out)  

      del valid_infos

    if FLAGS.dump_test:
      inputs = gezi.list_files(f'{base}/test/*record')
      dataset = Dataset('test')
      dataset = dataset.make_batch(1, inputs)
      deal(dataset, test_infos)
      print('after test', len(test_infos))

      ofile = ofile.replace('.pkl', '.test.pkl')  
      with open(ofile, 'wb') as out:
        pickle.dump(test_infos, out)
      for key in test_infos:
        print(test_infos[key])
        print(ids2text.ids2text(test_infos[key]['content']))
        break
  elif FLAGS.type == 'show_info':
    valid_infos = pickle.load(open(f'{base}/info.pkl', 'rb'))
    lens = [len(valid_infos[key]['content']) for key in valid_infos]
    unks = [list(valid_infos[key]['content']).count(FLAGS.unk_id) for key in valid_infos]
    print('num unks per doc:', sum(unks) / len(valid_infos))
    print('num doc with unk ratio:', len([x for x in unks if x != 0]) / len(unks)) 
    print('un unk tokens ratio:', sum(unks) / sum(lens))
    print('len max:', np.max(lens))
    print('len min:', np.min(lens))
    print('len mean:', np.mean(lens)) 
    print('num docs:', len(valid_infos))

    num_show = 0
    for key in valid_infos:
      if list(valid_infos[key]['content']).count(FLAGS.unk_id) > 0:
        print(valid_infos[key])
        print(ids2text.ids2text(valid_infos[key]['content']))
        num_show += 1
        if num_show > 5:
          break

    del valid_infos
    print('--------------for test info:')
    test_infos = pickle.load(open(f'{base}/info.test.pkl', 'rb'))
    lens = [len(test_infos[key]['content']) for key in test_infos]
    unks = [list(test_infos[key]['content']).count(FLAGS.unk_id) for key in test_infos]
    print('num unks per doc:', sum(unks) / len(test_infos))
    print('num doc with unk ratio:', len([x for x in unks if x != 0]) / len(test_infos)) 
    print('un unk tokens ratio:', sum(unks) / sum(lens))
    print('len max:', np.max(lens))
    print('len min:', np.min(lens))
    print('len mean:', np.mean(lens))
    print('num docs:', len(test_infos))

    num_show = 0
    for key in test_infos:
      if list(test_infos[key]['content']).count(FLAGS.unk_id) > 0:
        print(test_infos[key])
        print(ids2text.ids2text(test_infos[key]['content']))
        num_show += 1
        if num_show > 5:
          break
  else:
    raise ValueError(FLAGS.type)


if __name__ == '__main__':
  tf.app.run()
