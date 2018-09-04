from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#--------metrics @TODO use contrib ones
def precision_at_k(py_x, y, k=1, name=None):
  with tf.name_scope(name, 'precision_at_%d'%k, [py_x, y]):
    correct_prediction = tf.nn.in_top_k(py_x, y, k)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
