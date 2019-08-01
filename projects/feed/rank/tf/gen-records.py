#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2019-07-27 22:33:36.314010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app as absl_app
from absl import flags
FLAGS = flags.FLAGS

import glob
from tqdm import tqdm
import multiprocessing
from multiprocessing import Value, Manager
counter = Value('i', 0)

import gezi
import melt
from text_dataset import Dataset

import tensorflow as tf

dataset = None

def get_out_file(infile):
  infile_ = os.path.basename(infile)
  ofile_ = infile_ + '.record'
  ofile = os.path.join(FLAGS.out_dir, ofile_)
  return ofile

def build_features(infile):
  ofile = get_out_file(infile)
  print('----------writing to', ofile)
  with melt.tfrecords.Writer(ofile) as writer:
    for line in tqdm(open(infile)):
      fields = line.rstrip().split('\t')
      if len(fields) > 4:
        label = int(fields[0])
        id = '{}\t{}'.format(fields[2], fields[3])
        feat_id, feat_field, feat_value = dataset.get_feat(fields[4:])
        assert len(feat_id) == len(feat_value), "len(feat_id) == len(feat_value) -----------------"
        assert len(feat_id) == len(feat_field)

        feature = {
                    'label': melt.int64_feature(label),
                    'id': melt.bytes_feature(id),
                    'index': melt.int64_feature(feat_id),
                    'field': melt.int64_feature(feat_field),
                    'value': melt.float_feature(feat_value)
                  }
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(record)
        global counter
        with counter.get_lock():
          counter.value += 1


def main(_):
  global dataset
  dataset = Dataset()
  
  pool = multiprocessing.Pool()

  files = glob.glob(FLAGS.input)
  print('input', FLAGS.input)
  
  
  pool.map(build_features, files)
  pool.close()
  pool.join()

  print('num_records:', counter.value)

  out_file = '{}/num_records.txt'.format(FLAGS.out_dir)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  flags.DEFINE_string('input', None, '')
  flags.DEFINE_string('out_dir', None, '')
  flags.DEFINE_string('mode', None, '')

  absl_app.run(main) 
