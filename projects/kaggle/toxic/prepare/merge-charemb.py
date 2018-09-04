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
flags.DEFINE_string('emb', './mount/data/glove/glove.840B.300d-char.txt', '')
flags.DEFINE_integer('min_count', -1, '')
flags.DEFINE_bool('include_non_match', True, '')

from tqdm import tqdm
import numpy as np

def main(_):
  input_vocab = os.path.join(FLAGS.dir, 'char_vocab.full.txt')
  lines = open(input_vocab).readlines()

  ori_words_counts = [x.rstrip('\n').split('\t') for x in lines]
  # TODO FIXME why must remove? other wise when in  for word, count in zip(ori_words, counts): will ValueError: invalid literal for int() with base 10: '    '
  ori_words_counts = filter(lambda x: x[0].strip(), ori_words_counts)
  ori_words, counts = zip(*ori_words_counts)
  counts = map(int, counts)


  ori_set = set(ori_words)
  
  embedding_dict = {}

  vec_size = 300
  with open(FLAGS.emb, 'r', encoding='utf-8', errors='ignore') as fh:
    #for line in tqdm(fh, total=2196017):
    for i, line in enumerate(fh):
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
    #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
    emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
    words.append('<UNK>')

  emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
  words.append(' ')
  
  emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
  # TODO \x01 not used any more since, ignore \n ..
  words.append('\x01')
  
  emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
  words.append('<S>')
  
  emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
  words.append('</S>')

  with open('/home/gezi/tmp/rare_words.txt', 'w') as rare_out:
    for word, count in zip(ori_words, counts):
      if word in embedding_dict:
        emb_mat.append(np.array(embedding_dict[word]))
        words.append(word)  
      else:
        if FLAGS.include_non_match and count >= FLAGS.min_count:
          print('%s %d' % (word, count), file=rare_out)
          #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
          emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
          words.append(word)  

  out_vocab = os.path.join(FLAGS.dir, 'char_vocab.txt')
  with open(out_vocab, 'w') as out:
    for word in words:
      print(word, file=out)

  out_mat = os.path.join(FLAGS.dir, 'char_emb.npy')
  print('len(emb_mat)', len(emb_mat))
  np.save(out_mat, np.array(emb_mat))
    
if __name__ == '__main__':
  tf.app.run()
