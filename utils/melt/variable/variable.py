import tensorflow as tf
# Model Variables
#weights = slim.model_variable('weights',
#                              shape=[10, 10, 3 , 3],
#                              initializer=tf.truncated_normal_initializer(stddev=0.1),
#                              regularizer=slim.l2_regularizer(0.05),
#                              device='/CPU:0')
#model_variables = slim.get_model_variables()

## Regular variables
#my_var = slim.variable('my_var',
#                       shape=[20, 1],
#                       initializer=tf.zeros_initializer)
#regular_variables_and_model_variables = slim.get_variables()

#@TODO use slime model_variable directly or wrapp it here
def init_weights(shape, stddev=0.01, name=None):
  return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)

def init_weights_truncated(shape, stddev=1.0, name=None):
  return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def init_weights_random(shape, stddev=1.0, name=None):
  return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name) 

def init_weights_uniform(shape, minval=0, maxval=None, name=None):
  return tf.Variable(tf.random_uniform(shape, minval, maxval), name=name)

def init_bias(shape, val=0.1, name=None):
  if not isinstance(shape, (list, tuple)):
    shape = [shape]
  initial = tf.constant(val, shape=shape)
  return tf.Variable(initial, name=name)

#def get_weights(name, shape, stddev=0.01):
#  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=stddev))

def get_weights(name, shape, minval=-0.08, maxval=0.08, trainable=True):
  return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval, maxval), trainable=trainable)

def get_weights_truncated(name, shape, stddev=1.0, trainable=True):
  return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=trainable)

def get_weights_random(name, shape, stddev=1.0, trainable=True):
  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=stddev), trainable=trainable)
 
def get_weights_normal(name, shape, stddev=1.0, trainable=True):
  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=stddev), trainable=trainable)

def get_weights_uniform(name, shape, minval=0, maxval=None, trainable=True):
  return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval, maxval), trainable=trainable)

def get_bias(name, shape, val=0.1, trainable=True):
  if not isinstance(shape, (list, tuple)):
    shape = [shape]
  return tf.get_variable(name, shape, initializer=tf.constant_initializer(val), trainable=trainable)