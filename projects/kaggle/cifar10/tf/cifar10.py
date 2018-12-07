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
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('random_brightness', False, '')
flags.DEFINE_bool('random_contrast', False, '')
flags.DEFINE_bool('return_dict', False, '')

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion

  def get_filenames(self):
    if self.subset in ['train', 'valid', 'test']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'id': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([HEIGHT * WIDTH * DEPTH])

    image = tf.cast(
        tf.reshape(image, [HEIGHT, WIDTH, DEPTH]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    id = tf.cast(features['id'], tf.int32)

    # Custom preprocessing.
    image = self.preprocess(image)
    if not FLAGS.return_dict:
      return id, image, label
    else:
      return {'id': id, 'image': image}, label


  def make_batch(self, batch_size, filenames=None, repeat=None, initializable=None):
    """Read the images and labels from 'filenames'."""
    filenames = filenames or self.get_filenames()
    
    if initializable is None:
      initializable = self.subset != 'train'
    
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)

    if not initializable:
      iterator = dataset.make_one_shot_iterator()
    else:
      iterator = dataset.make_initializable_iterator()
    self.iterator = iterator
    return iterator

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      #... yes should do something like below.. but you will see with dataset.map.. not ok as summary without scope and finally graph has no these summaries
      # refer to https://stackoverflow.com/questions/47345394/image-summaries-with-tensorflows-dataset-api  TODO FIXME
      tf.summary.image('image', image)
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
      if FLAGS.random_brightness:
        image = tf.image.random_brightness(image,
                                          max_delta=63)
      if FLAGS.random_contrast:
        distorted_image = tf.image.random_contrast(image,
                                                  lower=0.2, 
                                                  upper=1.8)
      tf.summary.image('image/distort', image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 45000
    elif subset == 'valid':
      return 5000
    elif subset == 'test':
      #return 10000
      return 300000 # for kaggle cifar10 https://www.kaggle.com/c/cifar-10/submit
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
