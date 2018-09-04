#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-glove.py
#        \author   chenghuige  
#          \date   2018-01-15 23:52:08.616633
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dir', './mount/temp/toxic/tfrecords/glove/', '')
flags.DEFINE_string('emb', './mount/data/kaggle/toxic/talk_corpus/fastText/result.3gram.5epoch/toxic.ngram', '')
flags.DEFINE_integer('min_count', -1, '')
flags.DEFINE_integer('emb_dim', 300, '')
flags.DEFINE_string('out_name', 'emb.npy', '')
flags.DEFINE_string('type', 'normal', '''normal try merge all in glove, and add not in glove ones which with min train count,
                                       scratch add all with min train count and try merge glove,
                                       only only merge all glove and not consider train count''')
#flags.DEFINE_integer('ngram_buckets', 200000, '')
flags.DEFINE_string('ngram_vocab', None, '')
flags.DEFINE_string('ngram_output', None, '')

from tqdm import tqdm
import numpy as np
import gezi
from gezi import Vocabulary
import config

def main(_):
  num_conflicts = 0
  visited = {}
  visited_ngram = {}
  ngram_vocab_path = FLAGS.ngram_vocab or os.path.join(FLAGS.dir, 'ngram_vocab.txt')
  ngram_vocab = Vocabulary(ngram_vocab_path)
  print('ngram_vocab size', ngram_vocab.size())
  print('num ngram buckets', FLAGS.ngram_buckets)
  if FLAGS.emb.endswith('.npy'):
    ngram_emb = np.load(FLAGS.emb)
    assert len(ngram_emb) > 100000
  else:
    ngram_emb = []
    for line in open(FLAGS.emb):
      ngram_emb.append([float(x) for x in line.strip().split()])
  print('len ngram emb', len(ngram_emb))
  emb_mat = []
  vec_size = FLAGS.emb_dim
  # for padding zero
  emb_mat.append(np.array([0.] * vec_size))
  # exclude first pad and last 3 unk <s> </s>
  # unk, <s>, </s>, sincie ngram vocab txt not include these will append 
  for i in range(3):
    emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])

  for i in range(4, ngram_vocab.size()):
    ngram = ngram_vocab.key(i)
    ngram_hash = gezi.hash(ngram)
    ngram_id = ngram_hash % FLAGS.ngram_buckets
    if ngram_id not in visited:
      visited[ngram_id] = 1
      visited_ngram[ngram_id] = [ngram]
    else:
      visited[ngram_id] += 1
      visited_ngram[ngram_id].append(ngram)
      num_conflicts += 1
      #print('Conflict', visited_ngram[ngram_id], 'Num conflicts', num_conflicts)
    emb_mat.append(ngram_emb[ngram_id])
  print('Num conflicts', num_conflicts)

  print('len(emb_mat)', len(emb_mat))
  ngram_output = FLAGS.ngram_output or 'ngram.npy'
  out_mat = os.path.join(FLAGS.dir, ngram_output)
  print('out mat', out_mat)
  np.save(out_mat, np.array(emb_mat))
    
if __name__ == '__main__':
  tf.app.run()
