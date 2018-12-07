# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import glob
import cv2
import melt
import gezi

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

NUM_FOLDS = 10

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
classes_map = dict(zip(classes, range(len(classes))))

print('----------', classes_map)

import pandas as pd
l = pd.read_csv('./mount/data/kaggle/cifar-10/trainLabels.csv')
m = {}
for i in range(len(l)):
  m[l.id[i]] = l.label[i]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      id = int(os.path.basename(input_file)[:-4])
      img = cv2.imread(input_file)
      # turn to channel first
      #img = img.transpose(2,0,1)
      if 'test' not in output_file:  
        label = classes_map[m[id]]
      else:
        label = -1
      example = tf.train.Example(features=tf.train.Features(
          feature={
              'id': _int64_feature(id),
              'image': _bytes_feature(img.tobytes()),
              'label': _int64_feature(label)
          }))
      record_writer.write(example.SerializeToString())

def main(data_dir):
  input_dir = './mount/data/kaggle/cifar-10'
  
  input_files = [f for f in glob.glob('%s/%s/*.png' % (input_dir, 'train')) if int(os.path.basename(f)[:-4]) % NUM_FOLDS != 0]
  print('train:', len(input_files))
  output_file = os.path.join(data_dir, 'train.tfrecords')
  convert_to_tfrecord(input_files, output_file)

  input_files = [f for f in glob.glob('%s/%s/*.png' % (input_dir, 'train')) if int(os.path.basename(f)[:-4]) % NUM_FOLDS == 0]
  input_files = sorted(input_files, key=lambda f:int(os.path.basename(f)[:-4]))
  print('valid:', len(input_files))
  output_file = os.path.join(data_dir, 'valid.tfrecords')
  convert_to_tfrecord(input_files, output_file) 

  input_files = glob.glob('%s/%s/*.png' % (input_dir, 'test'))
  input_files = sorted(input_files, key=lambda f:int(os.path.basename(f)[:-4]))
  print('test:', len(input_files))
  output_file = os.path.join(data_dir, 'test.tfrecords')
  convert_to_tfrecord(input_files, output_file) 

  print('Done!')


if __name__ == '__main__':
  data_dir = './mount/data/cifar10'
  main(data_dir)
