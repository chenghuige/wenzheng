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

import tensorflow as tf
import glob
import cv2
from tqdm import tqdm
import melt
import gezi

NUM_FOLDS = 10

import pandas as pd
m = {}
l = pd.read_csv('../input/train.csv')
for i in range(len(l)):
  m[l.id_code[i]] = l.diagnosis[i]

def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in tqdm(input_files, ascii=True):
      id = os.path.basename(input_file)[:-4]
      #img = cv2.imread(input_file)
      img = melt.read_image(input_file)
      # turn to channel first
      #img = img.transpose(2,0,1)
      if 'test' not in output_file:  
        label = m[id]
      else:
        label = -1
      example = tf.train.Example(features=tf.train.Features(
          feature={
              'id': melt.bytes_feature(id),
              #'image': melt.bytes_feature(img.tobytes()),
              'image': melt.bytes_feature(img),
              'label': melt.int64_feature(label)
          }))
      record_writer.write(example.SerializeToString())

def main(data_dir):
  input_dir = '../input'
  
  input_files = [f for f in glob.glob('%s/%s/*.png' % (input_dir, 'train_images')) if int(gezi.hash(os.path.basename(f)[:-4])) % NUM_FOLDS != 0]
  print('train:', len(input_files))
  output_file = os.path.join(data_dir, 'train.tfrecords')
  convert_to_tfrecord(input_files, output_file)

  input_files = [f for f in glob.glob('%s/%s/*.png' % (input_dir, 'train_images')) if int(gezi.hash(os.path.basename(f)[:-4])) % NUM_FOLDS == 0]
  print('valid:', len(input_files))
  output_file = os.path.join(data_dir, 'valid.tfrecords')
  convert_to_tfrecord(input_files, output_file) 

  input_files = glob.glob('%s/%s/*.png' % (input_dir, 'test_images'))
  output_file = os.path.join(data_dir, 'test.tfrecords')
  convert_to_tfrecord(input_files, output_file) 


if __name__ == '__main__':
  data_dir = '../input/tfrecords'
  main(data_dir)
