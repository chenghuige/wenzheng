#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-ids.py
#        \author   chenghuige  
#          \date   2018-10-11 11:58:46.615350
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

import gezi
import numpy as np

from pathlib import Path

flags.DEFINE_string('vocab', '/home/gezi/mount/temp/ai2018/sentiment/tfrecords/char.glove/vocab.txt', '')
flags.DEFINE_string('idir', '/home/gezi/other/tools/GloVe-sentiment-char/', '')
flags.DEFINE_string('odir', '/home/gezi/mount/temp/lm/corpus/sentiment/', '')


def main(_):
  vocab = gezi.Vocabulary(FLAGS.vocab)  
  command = 'mkdir -p %s/valid' % FLAGS.odir
  print('command', command)
  os.system(command)

  command = 'mkdir -p %s/train' % FLAGS.odir
  print('command', command)
  os.system(command)

  # TODO Path will turn ./mount to mount...
  odir = Path(FLAGS.odir)
  print('odir', odir)
  def deal(file_, type):
    print(file_, type)
    ids_list = []
    for i, line in enumerate(open(file_)):
      if i % 100000 == 0:
        print(i)
      line = line.rstrip()
      line = line.strip('"') 
      line = line.strip()
      l = line.split(' ')
      l.insert(0, '<S>')
      ids = [vocab.id(x) for x in l]
      #print(' '.join(map(str, ids)), file=out)
      ids_list.append(np.array(ids))
    ids_list = np.array(ids_list) 
    out_file = odir / type / 'ids.npy'
    print('out_file', out_file)
    np.save(out_file, ids_list)

  idir = Path(FLAGS.idir)
  deal(idir / 'text.valid', 'valid')
  deal(idir / 'text.train', 'train')


if __name__ == '__main__':
  tf.app.run()  
  