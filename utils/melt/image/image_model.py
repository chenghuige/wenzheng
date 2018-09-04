#!/usr/bin/env python
# ==============================================================================
#          \file   image_model.py
#        \author   chenghuige  
#          \date   2017-04-10 19:58:46.031602
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
import sys
#import math

import tensorflow as tf

import numpy as np
  
import melt
import gezi

try:
  from nets import nets_factory
except Exception:
  pass

slim = tf.contrib.slim

class ImageModel(object):
  def __init__(self, 
               image_checkpoint_file=None,
               model_name=None, 
               height=None, 
               width=None,
               feature_name=None,
               image_format='jpeg',
               moving_average_decay=None,
               num_classes=None,
               top_k=None,
               sess=None,
               graph=None):
    assert image_checkpoint_file or model_name, 'need model_name if train from scratch otherwise need image_checkpoint_file'
    self.graph = tf.Graph() if graph is None else graph
    self.sess = melt.gen_session(graph=self.graph) if sess is None else sess
    self.feature_name = feature_name

    if image_checkpoint_file:
      net = melt.image.get_imagenet_from_checkpoint(image_checkpoint_file)
      assert net is not None, image_checkpoint_file
      model_name = model_name or net.name
      height = height or net.default_image_size
      width = width or net.default_image_size
    else:
      assert model_name is not None
      gnu_name = gezi.to_gnu_name(model_name)
      net = nets_factory.networks_map[gnu_name]
      height = height or net.default_image_size
      width = width or net.default_image_size

    print('checkpoint', image_checkpoint_file, 'model_name', model_name, 
          'height', height, 'width', width, file=sys.stderr)

    self.num_classes = num_classes
    self.model_name = model_name
    with self.sess.graph.as_default():
      self.images_feed = tf.placeholder(tf.string, [None, ], name='images')
      if not self.num_classes:
        print('build graph for final one feature', file=sys.stderr)
        self.feature = self._build_graph(model_name, height, width, image_format=image_format)
        print('build graph for attention features', file=sys.stderr)
        self.features = self._build_graph2(model_name, height, width, image_format=image_format)
      else:
        assert self.num_classes > 1
        if feature_name != 'Logits':
          prelogits_feature = self._build_graph(model_name, height, width, image_format=image_format)
          #with tf.variable_scope('ImageModelLogits'):
          self.logits = slim.fully_connected(prelogits_feature, num_classes, activation_fn=None,
                                      scope='Logits')
        else:
          # directly use slim model
          self.logits = self._build_graph(model_name, height, width, num_classes=num_classes, image_format=image_format)
        if top_k:
          with tf.variable_scope('ImageModelTopN'):
            self.top_logits, self.top_indices = tf.nn.top_k(self.logits, top_k, name='TopK')
        self.predictions = tf.nn.softmax(self.logits, name='Predictions')
        # https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.readme.txt
        self.multi_predictions = tf.nn.sigmoid(self.logits, name='multi_predictions')

      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      self.sess.run(init_op)
      if image_checkpoint_file:
        #---load inception model check point file
        init_fn = melt.image.image_processing.create_image_model_init_fn(model_name, 
                                                                         image_checkpoint_file,
                                                                         moving_average_decay=moving_average_decay)
        init_fn(self.sess)

  def _build_graph(self, model_name, height, width, num_classes=None, image_format=None):
    melt.apps.image_processing.init(model_name, self.feature_name)
    return melt.apps.image_processing.image_processing_fn(self.images_feed,  
                                                          height=height, 
                                                          width=width,
                                                          image_format=image_format,
                                                          feature_name=self.feature_name,
                                                          num_classes=num_classes)

  def _build_graph2(self, model_name, height, width, image_format=None):
    features_name = melt.get_features_name(self.model_name)
    melt.apps.image_processing.init(model_name, features_name)
    return melt.apps.image_processing.image_processing_fn(self.images_feed,  
                                                          height=height, 
                                                          width=width,
                                                          image_format=image_format,
                                                          feature_name=features_name)

  def process(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.feature, feed_dict={self.images_feed: images})

  def process2(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.features, feed_dict={self.images_feed: images})

  def gen_logits(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.logits, feed_dict={self.images_feed: images})

  def classify(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.predictions, feed_dict={self.images_feed: images})

  def multi_classify(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.multi_predictions, feed_dict={self.images_feed: images})

  def logits(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.logits, feed_dict={self.images_feed: images})

  def top_k(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run([self.top_logits, self.top_indices], feed_dict={self.images_feed: images})     

  def gen_feature(self, images):
    return self.process(images)

  def gen_features(self, images):
    return self.process2(images)
