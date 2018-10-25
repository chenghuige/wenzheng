#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description   mix seg must have input vocab
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, '')
flags.DEFINE_string('vocab', None, '')

import sys,os
os.environ['BSEG'] = '1'
import numpy as np

import gezi
from gezi import WordCounter 

import pandas as pd

from projects.ai2018.sentiment.prepare import filter

from tqdm import tqdm
import traceback 

assert gezi.env_has('BSEG')

vocab = None

def seg(id, text, out):
  text = filter.filter(text)
  words = []
  for word in gezi.cut(text):
    if vocab.has(word):
      words.append(word)
    else:
      for ch in word.decode('utf8'):
        words.append(ch.encode('utf8'))
  print(id, '\x09'.join(words), sep='\t', file=out)

def main(_):  
  # FLAGS.seg_method = 'basic_digit'
  # FLAGS.feed_single = True
  # FLAGS.feed_single_en = True
  # print('seg_method:', FLAGS.seg_method, file=sys.stderr)
  # print('feed_single:', FLAGS.feed_single, file=sys.stderr)
  # print('feed_single_en:', FLAGS.feed_single_en, file=sys.stderr)

  #assert FLAGS.vocab 

  global vocab 
  vocab = gezi.Vocabulary(FLAGS.vocab)

  ifile = sys.argv[1]
  if not gezi.env_has('BSEG'):
    ofile = ifile.replace('.csv', '.seg.mix.txt')
  else:
    ofile = ifile.replace('.csv', '.seg.bseg.mix.txt')

  ids_set = set()
  fm = 'w'
  if os.path.exists(ofile):
    fm = 'a'
    for line in open(ofile):
      ids_set.add(line.split('\t')[0])

  print('%s already done %d' % (ofile, len(ids_set)))

  num_errs = 0
  with open(ofile, fm) as out:
    df = pd.read_csv(ifile, lineterminator='\n')
    contents = df['content'].values 
    ids = df['id'].values
    for i in tqdm(range(len(df)), ascii=True):
      if str(ids[i]) in ids_set:
        continue
      #if i != 2333:
      #  continue
      #print(gezi.cut(filter.filter(contents[i]), type_))
      try:
        seg(ids[i], contents[i], out)
      except Exception:
        #print(traceback.format_exc())
        num_errs += 1
        continue
      #exit(0)

  print('num_errs:', num_errs, 'ratio:', num_errs / len(df))

if __name__ == '__main__':
  tf.app.run()
