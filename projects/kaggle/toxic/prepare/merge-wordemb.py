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
flags.DEFINE_string('emb', './mount/data/glove/glove.840B.300d.txt', '')
flags.DEFINE_integer('min_count', 4, '')
flags.DEFINE_bool('include_non_match', True, '')

from tqdm import tqdm
import numpy as np

def main(_):
  input_vocab = os.path.join(FLAGS.dir, 'vocab.full.txt')
  lines = open(input_vocab).readlines()

  ori_words_counts = [x.strip().split('\t') for x in lines]
  ori_words, counts = zip(*ori_words_counts)
  counts = map(int, counts)

  ori_set = set(ori_words)
  
  embedding_dict = {}

  vec_size = 300
  with open(FLAGS.emb, 'r', encoding='utf-8') as fh:
    for line in tqdm(fh, total=2196017):
    #for i, line in enumerate(fh):
      array = line.split()
      word = "".join(array[0: -vec_size])
      vector = list(map(float, array[-vec_size:]))
      if word in ori_set:
        embedding_dict[word] = vector
      #if i == 1000:
      #  print(i)
  print("{} / {} tokens have corresponding word embedding vector".format(len(embedding_dict), len(ori_words)))
  
  words = []   
  emb_mat = []
  emb_mat.append(np.array([0.] * vec_size))
  if not '<UNK>' in ori_set:
    #change from all 0 to random normal for unk
    emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
    words.append('<UNK>')

  with open('/home/gezi/tmp/rare_words.txt', 'w') as rare_out:
    for word, count in zip(ori_words, counts):
      if word in embedding_dict:
        emb_mat.append(np.array(embedding_dict[word]))
        words.append(word)  
      else:
        if FLAGS.include_non_match and count >= FLAGS.min_count:
          print('%s %d' % (word, count), file=rare_out)
          emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
          words.append(word)  

  out_vocab = os.path.join(FLAGS.dir, 'vocab.txt')
  with open(out_vocab, 'w') as out:
    for word in words:
      print(word, file=out)

  out_mat = os.path.join(FLAGS.dir, 'word_emb.npy')
  print('len(emb_mat)', len(emb_mat))
  np.save(out_mat, np.array(emb_mat))
    
if __name__ == '__main__':
  tf.app.run()
