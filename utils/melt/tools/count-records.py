#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', './*', '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_integer('threads', 12, '') 
flags.DEFINE_bool('write_count', False, '')

import sys, os, time
import gezi
import melt 

import  multiprocessing
from multiprocessing import Process, Manager, Value

counter = Value('i', 0)
def deal_file(file):
  try:
    count = melt.get_num_records_single(file)
  except Exception:
    print('bad file:', file)
  global counter
  with counter.get_lock():
    counter.value += count 
  print(file, count)

def main(_):
  timer = gezi.Timer()
  input = FLAGS.input 
  
  if FLAGS.threads == 1:
    num_records = melt.get_num_records_print(input)
    print(timer.elapsed())
  else:
    files = gezi.list_files(input)
    print(files)
    pool = multiprocessing.Pool(processes = FLAGS.threads)
    pool.map(deal_file, files)
    pool.close()
    pool.join()
    
    num_records = counter.value 
    print('num_records:', num_records)

  if FLAGS.write_count:
    outdir = os.path.dirname(input)  
    output = '%s/num_records.txt' % outdir
    print('write to %s'%output)
    out = open(output, 'w')
    out.write(str(num_records))


if __name__ == '__main__':
  tf.app.run()
