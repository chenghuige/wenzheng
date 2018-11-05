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
#flags.DEFINE_integer('fold', None, '')

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
    inputs = gezi.list_files(f'{base}/train/*record')
    dataset = Dataset('valid')
    dataset = dataset.make_batch(1, inputs)
    deal(dataset, valid_infos)
    print('after valid', len(valid_infos))
    inputs = gezi.list_files(f'{base}/test/*record')
    dataset = Dataset('test')
    dataset = dataset.make_batch(1, inputs)
    deal(dataset, test_infos)
    print('after test', len(test_infos))

    for key in valid_infos:
      print(valid_infos[key])
      print(ids2text.ids2text(valid_infos[key]['content']))
      break

    ofile = f'{base}/info.pkl'
    with open(ofile, 'wb') as out:
      pickle.dump(valid_infos, out)  

    ofile = ofile.replace('.pkl', '.test.pkl')  
    with open(ofile, 'wb') as out:
      pickle.dump(test_infos, out)
  elif FLAGS.type == 'show_info':
    valid_infos = pickle.load(open(f'{base}/info.pkl', 'rb'))
    lens = [len(valid_infos[key]['content']) for key in valid_infos]
    unks = [list(valid_infos[key]['content']).count(1) for key in valid_infos]
    print('num unks per doc:', sum(unks) / len(unks))
    print('num doc with unk ratio:', len([x for x in unks if x != 0]) / len(unks)) 
    print('un unk tokens ratio:', sum(unks) / sum(lens))
    print('len max:', np.max(lens))
    print('len min:', np.min(lens))
    print('len mean:', np.mean(lens))
  else:
    raise ValueError(FLAGS.type)


if __name__ == '__main__':
  tf.app.run()
