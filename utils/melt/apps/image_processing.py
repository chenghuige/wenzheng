#!/usr/bin/env python
# ==============================================================================
#          \file   image_processing.py
#        \author   chenghuige  
#          \date   2017-04-07 08:49:43.118136
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt
# from melt import logging

flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

image_processing_fn = None

def init(image_model_name=None, feature_name=None, num_classes=None, preprocess_image=True, im2text_prcocessing=False):
  global image_processing_fn
  if image_processing_fn is None:
    assert image_model_name is not None
    if im2text_prcocessing:
      raise ValueError('not use im2txt anymore')
      image_processing_fn = melt.image.image_processing.create_image2feature_fn(image_model_name)
    else:
      image_processing_fn = melt.image.image_processing.create_image2feature_slim_fn(image_model_name, 
                                                                                     feature_name=feature_name,
                                                                                     num_classes=num_classes,
                                                                                     preprocess_image=preprocess_image)
  return image_processing_fn
