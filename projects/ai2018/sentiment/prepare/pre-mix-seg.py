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
flags.DEFINE_bool('for_pretrain', False, '')

import sys,os
#os.environ['BSEG'] = '1'

import numpy as np

import gezi
from gezi import WordCounter 

import pandas as pd

from projects.ai2018.sentiment.prepare import filter

from tqdm import tqdm
import traceback 

#assert gezi.env_has('BSEG')

import six 
if gezi.env_has('BSEG'):
  assert six.PY2

vocab = None

def seg(id, text, out, counter):
  text = filter.filter(text)
  words = []
  for i, word in enumerate(gezi.cut(text)):
    counter.add(str(i))
    if vocab.has(word) and not word.isdigit():
      words.append('%s|%d' % (word, i))
    else:
      if six.PY2:
        for ch in word.decode('utf8'):
          words.append('%s|%d' % (ch.encode('utf8'), i))
      else:
        for ch in word:
          words.append('%s|%d' % (ch, i))

  if not FLAGS.for_pretrain:
    print(id, '\x09'.join(words), sep='\t', file=out)
  else:
    print(' '.join([x.split('|')[0] for x in words]), file=out)

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
    ofile = ifile.replace('.csv', '.seg.jieba.mix.txt')
  else:
    ofile = ifile.replace('.csv', '.seg.bseg.mix.txt')


  counter = WordCounter(most_common=0, min_count=1)
  vocab2 = ifile.replace('.csv', '.pos.mix.vocab')


  num_errs = 0
  with open(ofile, 'w') as out:
    df = pd.read_csv(ifile, lineterminator='\n')
    contents = df['content'].values 
    ids = df['id'].values
    for i in tqdm(range(len(df)), ascii=True):
      #if str(ids[i]) in ids_set:
      #  continue
      #if i != 2333:
      #  continue
      #print(gezi.cut(filter.filter(contents[i]), type_))
      try:
        seg(ids[i], contents[i], out, counter)
      except Exception:
        if num_errs == 0:
          print(traceback.format_exc())
        num_errs += 1
        continue
      #exit(0)

  counter.save(vocab2)
  print('num_errs:', num_errs, 'ratio:', num_errs / len(df))

if __name__ == '__main__':
  tf.app.run()
