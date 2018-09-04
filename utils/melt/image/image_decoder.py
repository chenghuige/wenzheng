#!/usr/bin/env python
# ==============================================================================
#          \file   image_decoder.py
#        \author   chenghuige  
#          \date   2017-03-31 12:08:54.242407
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded, channels=3)

    try:
      self._decode_bmp = tf.image.decode_bmp(self._encoded, channels=3)
      self._decode = tf.image.decode_image(self._encoded, channels=3) 
    except Exception:
      pass

    self.num_imgs = 0
    self.num_bad_imgs = 0

  def decode_jpeg(self, encoded_jpeg):
    try:
      image = self._sess.run(self._decode_jpeg,
                             feed_dict={self._encoded: encoded_jpeg})
    except Exception:
      image = None
    return image

  def decode(self, encoded, image_format='jpeg'):
    if image_format == 'jpeg':
      decode_op = self._decode_jpeg
    elif image_format == 'bmp':
      decode_op = self._decode_bmp
    else:
      decode_op = self._decode
    self.num_imgs += 1
    try:
      image = self._sess.run(decode_op,
                             feed_dict={self._encoded: encoded})
    except Exception:
      image = None
      self.num_bad_imgs += 1
    return image