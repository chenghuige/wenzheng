#!/usr/bin/env python
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-16 19:32:41.443712
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim

import sys, os, glob, traceback
import inspect
import six

import numpy as np

import glob

import gezi
import melt 

logging = melt.logging

keras = tf.keras

# TODO FIXME should use below but now not work
def create_restore_fn(checkpoint, model_name, restore_model_name):
  model_name = gezi.pascal2gnu(model_name)
  restore_model_name = gezi.pascal2gnu(restore_model_name)
  
  variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name)
  assert variables_to_restore, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  prefix  = '%s/%s' % (model_name, restore_model_name)

  # remove model_name
  def name_in_checkpoint(var):
    return var.op.name.replace(prefix, restore_model_name)

  variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore if var.op.name.startswith(prefix)}
  
  varnames_in_checkpoint = melt.get_checkpoint_varnames(checkpoint)
  # FIXME wrong..
  variables_to_restore = {var2:var for var2 in varnames_in_checkpoint}

  saver = tf.train.Saver(variables_to_restore)

  def restore_fn(sess):
    timer = gezi.Timer('restore var from %s %s' % (restore_model_name, checkpoint))
    saver.restore(sess, checkpoint)
    timer.print()

  return restore_fn

class Model(keras.Model):
  def __init__(self,  
               **kwargs):
    super(Model, self).__init__(**kwargs)
    self.debug = False
    self.step = -1
    self.training = False
  
  def train(self):
    self.training = True

  def eval(self):
    self.training = False

# def reset_global_step(global_step):
#   new_global_step = tf.train.get_global_step().assign(global_step)

def exists_model(model_dir):
  return os.path.exists(model_dir) and (not os.path.isdir(model_dir) or glob.glob(model_dir + '/model*ckpt*'))

from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  if ratio is None:
    ratios = tf.get_collection(name)[-1]
    x = x * ratios + tf.stop_gradient(x) * (1 - ratios)
  else:
    x = x * ratio + tf.stop_gradient(x) * (1 - ratio)
  return x

def adjust_weights(x, ratio=None, name='learning_rate_weights'):
  if ratio is None:
    ratios = tf.get_collection(name)[-1]
    x = x * ratios 
  else:
    x = x * ratio 
  return x

def try_convert_images(images):
  if not isinstance(images, (list, tuple, np.ndarray)):
    images = [images]
  if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
    images = [melt.image.read_image(image) for image in images]
  return images

def freeze_graph(sess, model_path, global_step=None, output_collection_names=None, output_node_names=None):
  if output_collection_names is None and output_node_names is None:
    return None  

  graph_def = sess.graph_def
  graph = sess.graph

  if global_step is not None:  # allow 0
    outfile = '%s-%d.pb' % (model_path, global_step)
    outmapfile = '%s-%d.map' % (model_path, global_step)
  else:
    outfile = '%s.pb' % model_path
    outmapfile = '%s.map' % model_path

  #print('outfile', outfile, 'outmap', outmapfile)

  if output_node_names is None:
    output_node_names = []
    outmap = open(outmapfile, 'w')
    for cname in output_collection_names:
      for item in graph.get_collection(cname):
        # [TopKV2(values=<tf.Tensor 'TopKV2_1:0' shape=(1,) dtype=int32>, indices=<tf.Tensor 'TopKV2_1:1' shape=(1,) dtype=int32>)]
        # f.get_collection('y')[0][0].name Out[17]: u'TopKV2_1:0'

        # bypass :0
        # if not hasattr(item, 'name'):
        #   print('item no name for', item, file=sys.stderr)
        #   continue
        # :1 for like top_k, :2 for future usage might length 3 tuple
        if item is None:
          continue
        if not (item.name.endswith(':0') or item.name.endswith(':1') or item.name.endswith(':2')):
          continue
        opname = item.name[:-2]
        output_node_names.append(opname)
        print('%s\t%s' % (cname, item.name), file=outmap)
  frozen_graph_def = convert_variables_to_constants(sess, graph_def, output_node_names)

  #print('outfile')
  with tf.gfile.GFile(outfile, "wb") as f:
    f.write(frozen_graph_def.SerializeToString())

  model_dir = os.path.dirname(outfile)
  #print('model_dir', model_dir)
  maps = glob.glob('%s/*.map' % model_dir)
  #print('maps', maps)
  for map_ in maps:
    index_file = map_.replace('.map', '.index')
    if not os.path.exists(index_file):
      pb_file = map_.replace('.map', '.pb')
      #print('remove %s %s' % (map_, pb_file))
      os.remove(map_)
      os.remove(pb_file)

  return frozen_graph_def


def is_raw_image(image_features):
  return isinstance(image_features[0], np.string_)


def set_learning_rate(lr, sess=None, name='learning_rate'):
  if not sess:
    sess = melt.get_session()
  sess.run(tf.assign(tf.get_collection(name)[-1], lr))

def multiply_learning_rate(lr, sess=None, name='learning_rate'):
  if not sess:
    sess = melt.get_session()
  sess.run(tf.assign(tf.get_collection(name)[-1], tf.get_collection(name)[-1] * lr))

#https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
def init_uninitialized_variables(sess, list_of_variables = None):
  if list_of_variables is None:
    list_of_variables = tf.global_variables()
  uninitialized_variables = list(tf.get_variable(name) for name in
                                 sess.run(tf.report_uninitialized_variables(list_of_variables)))
  uninitialized_variables = tf.group(uninitialized_variables, tf.local_variables_initializer()) 
  sess.run(tf.variables_initializer(uninitialized_variables))
  return uninitialized_variables

def get_global_step(model_dir, num_steps_per_epoch, fix_step=True):
  if not model_dir:
    return 0

  checkpoint_path = get_model_path(model_dir)
  if os.path.isdir(checkpoint_path) or not os.path.exists(checkpoint_path + '.index'):
    return 0
  
  pre_step = get_model_step(checkpoint_path)
  if not num_steps_per_epoch or not fix_step:
    return pre_step

  pre_epoch = melt.get_model_epoch(checkpoint_path)
  if pre_epoch is None:
    return pre_step
  
  fixed_pre_step = pre_step
  if abs(pre_step / num_steps_per_epoch - pre_epoch) > 0.1:
    fixed_pre_step = int(pre_epoch * num_steps_per_epoch)
    return fixed_pre_step
  else:
    return pre_step

def checkpoint_exists(checkpoint_path):
  return not os.path.isdir(checkpoint_path) and \
         os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + '.index') 

def get_checkpoint_varnames(model_dir):
  checkpoint_path = get_model_path(model_dir)
  #if model_dir is dir then checkpoint_path should be model path not model dir like
  #/home/gezi/new/temp/image-caption/makeup/model/bow.inceptionResnetV2.finetune.later/model.ckpt-589.5-1011000
  #if user input model_dir is like above model path we assume it to be correct and exists not check!
  if os.path.isdir(checkpoint_path) or not os.path.exists(checkpoint_path + '.index'):
    return None
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    varnames = [var_name for var_name in var_to_shape_map]
    return varnames
  except Exception:
    print(traceback.format_exc())
    return None

def varname_in_checkpoint(varname_part, model_dir, mode='in'):
  assert varname_part
  varnames = get_checkpoint_varnames(model_dir)
  if not varnames:
    return False
  else:
    varnames_exists = False
    for varname in varnames:
      if mode == 'in':
        if varname_part in varname:
          varnames_exists = True
          break
      elif mode == 'startswith':
        if varname.startswith(varname_part):
          varnames_exists = True
          break
      elif mode == 'exact_match':
        if varname_part == varname:
          varnames_exists = True
          
    return varnames_exists

def has_image_model(model_dir, image_model_name):
  return varname_in_checkpoint(image_model_name, model_dir)

def try_add_to_collection(name, op):
  if not tf.get_collection(name):
    tf.add_to_collection(name, op)

def remove_from_collection(key):
  #must use ref get list and set to empty using [:] = [] or py3 can .clear
  #https://stackoverflow.com/questions/850795/different-ways-of-clearing-lists
  l = tf.get_collection_ref(key)
  l[:] = []  

def rename_from_collection(key, to_key, index=0, scope=None):
  l = tf.get_collection_ref(key)
  if l:
    tf.add_to_collection(to_key, l[index])
    l[:] = []

#https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
def initialize_uninitialized_vars(sess):
  import itertools
  from itertools import compress
  global_vars = tf.global_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                  for var in global_vars])
  not_initialized_vars = list(compress(global_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))

#In [3]: tf.contrib.layers.OPTIMIZER_CLS_NAMES
#Out[3]: 
#{'Adagrad': tensorflow.python.training.adagrad.AdagradOptimizer,
# 'Adam': tensorflow.python.training.adam.AdamOptimizer,
# 'Ftrl': tensorflow.python.training.ftrl.FtrlOptimizer,
# 'Momentum': tensorflow.python.training.momentum.MomentumOptimizer,
# 'RMSProp': tensorflow.python.training.rmsprop.RMSPropOptimizer,
# 'SGD': tensorflow.python.training.gradient_descent.GradientDescentOptimizer}

optimizers = {
  'grad': tf.train.GradientDescentOptimizer,
  'sgd': tf.train.GradientDescentOptimizer,
  'adagrad': tf.train.AdagradOptimizer,  
  # TODO notice tensorflow adagrad no epsilon param..  See if kears optimizers better ?
  #'adagrad': lambda lr: tf.train.AdagradOptimizer(lr, epsilon=1e-06), # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
  #'adam': tf.train.AdamOptimizer,
  #'adam': lambda lr: tf.train.AdamOptimizer(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08), #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  'adam': lambda lr: tf.train.AdamOptimizer(lr, epsilon=1e-08),
  'adam_t2t': lambda lr: tf.train.AdamOptimizer(lr, epsilon=1e-06, beta1=0.85, beta2=0.997),
  #'adadelta': tf.train.AdadeltaOptimizer
  'adadelta': lambda lr: tf.train.AdadeltaOptimizer(lr, epsilon=1e-6),  #follow squad, also keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
  # still not fix https://github.com/tensorflow/tensorflow/pull/15665
  'nadam': tf.contrib.opt.NadamOptimizer, # TODO FIXME nadam not work, InvalidArgumentError (see above for traceback): Incompatible shapes: [2737,300] vs. [91677,300] 
  #'momentum': lambda lr, momentum: tf.train.MomentumOptimizer(lr, momentum=momentum) # in melt.app.train
  #'adamax': tf.contrib.opt.AdaMaxOptimizer, # will got NAN ...
  #'adamax': lambda lr: tf.contrib.opt.AdaMaxOptimizer(lr, epsilon=1e-8),
  'adamax': melt.training.adamax.AdaMaxOptimizer,
  #'adamax': tf.keras.optimizers.Adamax,  # tf can not directly use kears optimzier...
  }

keras_optimizers = {
  'adagrad': tf.keras.optimizers.Adagrad,
  'adam': tf.keras.optimizers.Adam, 
  'adadelta': tf.keras.optimizers.Adadelta,
  'nadam': tf.keras.optimizers.Nadam,
}

def get_session(log_device_placement=False, allow_soft_placement=True, debug=False, device_count=None):
  """
  TODO FIXME get_session will casue  at last
#Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
#TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

global or inside function global sess will cause this but not big problem for convenience just accpet right now
  """
  if not hasattr(get_session, 'sess') or get_session.sess is None:
    if device_count is None:
      config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, 
                            log_device_placement=log_device_placement)
    else:
      config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, 
                            log_device_placement=log_device_placement,
                            device_count=device_count)    
    if FLAGS.use_horovod:
      config.gpu_options.allow_growth = True
      import horovod.keras as hvd
      config.gpu_options.visible_device_list = str(hvd.local_rank())  
    #config.operation_timeout_in_ms=600000
    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
    #config.operation_timeout_in_ms=50000   # terminate on long hangs
    #https://github.com/tensorflow/tensorflow/issues/2292 allow_soft_placement=True
    if not FLAGS.use_tpu:
      get_session.sess = tf.Session(config=config)
    else:
      tpu_cluster_resolver = None
      if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      get_session.sess = tf.Session(tpu_cluster_resolver)
    if debug:
      from tensorflow.python import debug as tf_debug
      get_session.sess = tf_debug.LocalCLIDebugWrapperSession(get_session.sess)
  return get_session.sess

def gen_session(graph=None, log_device_placement=False, allow_soft_placement=True, debug=False):
  config=tf.ConfigProto(allow_soft_placement=allow_soft_placement, 
                        log_device_placement=log_device_placement)
  sess = tf.Session(config=config, graph=graph)
  if debug:
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  return sess

#def get_session(log_device_placement=False, allow_soft_placement=True, debug=False):
#  """
#  TODO FIXME get_session will casue  at last
##Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
##TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

#global or inside function global sess will cause this but not big problem for convenience just accpet right now
#  """
#  if not hasattr(get_session, 'sess') or get_session.sess is None:
#    config=tf.ConfigProto(
#      allow_soft_placement=allow_soft_placement, 
#      log_device_placement=log_device_placement)
#    #config.operation_timeout_in_ms=600000
#    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
#    #config.operation_timeout_in_ms=50000   # terminate on long hangs
#    sess = tf.Session(config=config)
#    if debug:
#      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#  return sess

def get_optimizer(name):
  if not isinstance(name, str):
    return name 
  # TODO how to use keras optimizers especially like nadam ? seems different api
  # if name.lower() in keras_optimizers:
  #   return keras_optimizers[name.lower()]
  # elif name.lower() in optimizers:
  if name.lower() in optimizers:
    return optimizers[name.lower()]
  else:
    return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name]
  # if name in tf.contrib.layers.OPTIMIZER_CLS_NAMES:
  #   return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name]
  # else:
  #   return optimizers[name.lower()]

def gen_train_op(loss, learning_rate, optimizer=tf.train.AdagradOptimizer):
  train_op = optimizer(learning_rate).minimize(loss)  
  return train_op  

def gen_train_op_byname(loss, learning_rate, name='adagrad'):
  optimizer = optimizers.get(name.lower(), tf.train.AdagradOptimizer)
  train_op = optimizer(learning_rate).minimize(loss)  
  return train_op  

#TODO add name, notice if num_gpus=1 is same as num_gpus=0
#but for num_gpus=0 we will not consider multi gpu mode
#so num_gpus=1 will not use much, just for mlti gpu test purpose
#from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py def train()

# def tower(loss_function, num_gpus=1, training=True, name=''):
#   towers = []
#   update_ops = []
#   for i in range(num_gpus):
#     with tf.device('/gpu:%d' % i):
#       #print(tf.get_variable_scope().reuse)
#       with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)): 
#         with tf.name_scope('%s_%d' % ('tower', i)) as name_scope:
#           if 'i' in inspect.getargspec(loss_function).args:
#             loss = loss_function(i)
#           else:
#             loss = loss_function()
#           # Reuse variables for the next tower. itersting.. not work.. for cifar10 ._conv...
#           #print(tf.get_variable_scope().reuse)
#           #tf.get_variable_scope().reuse_variables()
#           #print(tf.get_variable_scope().reuse)
#           # REMIND actually for training other metrics like acc... will only record the last one, I think this is enough!
#           if isinstance(loss, (list, tuple)) and training:
#             loss = loss[0]
#           towers.append(loss)
#           if i == 0 and training:
#             # Only trigger batch_norm moving mean and variance update from
#             # the 1st tower. Ideally, we should grab the updates from all
#             # towers but these stats accumulate extremely fast so we can
#             # ignore the other stats from the other towers without
#             # significant detriment.
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
#                                             name_scope)
#   if training:
#     return towers, update_ops
#   else:
#     towers = [list(x) if isinstance(x, tuple) else x for x in towers]
#     return towers

# TODO will this be ok.. ?
def tower(loss_function, num_gpus=1, training=True, name=''):
  towers = []
  update_ops = []
  for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
      #print(tf.get_variable_scope().reuse)
      with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)): 
        #with tf.name_scope('%s_%d' % ('tower', i)) as name_scope:
        if 'i' in inspect.getargspec(loss_function).args:
          loss = loss_function(i)
        else:
          loss = loss_function()
        # Reuse variables for the next tower. itersting.. not work.. for cifar10 ._conv...
        #print(tf.get_variable_scope().reuse)
        #tf.get_variable_scope().reuse_variables()
        #print(tf.get_variable_scope().reuse)
        # REMIND actually for training other metrics like acc... will only record the last one, I think this is enough!
        if isinstance(loss, (list, tuple)) and training:
          loss = loss[0]
        towers.append(loss)
        if i == 0 and training:
          # Only trigger batch_norm moving mean and variance update from
          # the 1st tower. Ideally, we should grab the updates from all
          # towers but these stats accumulate extremely fast so we can
          # ignore the other stats from the other towers without
          # significant detriment.
          # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
          #                                name_scope)
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if training:
    return towers, update_ops
  else:
    towers = [list(x) if isinstance(x, tuple) else x for x in towers]
    return towers

tower_losses = tower

# from cifar10_estimator example code
# TODO can it be used with out input of batch_size so as can be used for buckets length ? different batch size how to ?
def _split_batch(batch_datas, batch_size, num_shards, training=True):
  #with tf.device('/cpu:0'):
  batch_datas = [tf.unstack(batch_data, num=batch_size, axis=0) for batch_data in batch_datas]

  new_batch_datas = []
  for i in range(len(batch_datas)):
    new_batch_datas.append([[] for i in range(num_shards)])

  batch_size_per_gpu = batch_size // num_shards
  assert batch_size == batch_size_per_gpu * num_shards

  for i in range(batch_size):
    idx = i % num_shards if training else i // batch_size_per_gpu
    for j in range(len(batch_datas)):
      new_batch_datas[j][idx].append(batch_datas[j][i])

  def stack(x):
    try:
      return tf.parallel_stack(x)
    except Exception:
      return tf.stack(x)

  for i in range(len(batch_datas)):
    #new_batch_datas[i] = [tf.parallel_stack(x) for x in new_batch_datas[i] if x]
    new_batch_datas[i] = [stack(x) for x in new_batch_datas[i] if x]

  return tuple(new_batch_datas)

def split_batch(batch_datas, batch_size, num_shards, training=True):
  with tf.device('/cpu:0'):
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return tuple([x] for x in batch_datas)

    if not isinstance(batch_datas[0], dict):
      return _split_batch(batch_datas, batch_size, num_shards, training)
    else:
      # x, y (x is dict, y not)
      assert len(batch_datas) == 2 
      keys = batch_datas[0].keys()
      #batch_datas = [batch_datas[0][key] for key in keys] + [batch_datas[-1]]
      batch_datas = list(batch_datas[0].values()) + [batch_datas[-1]]
      batch_datas = _split_batch(batch_datas, batch_size, num_shards, training)
      # print(batch_datas)
      # TODO... why append ok... x = [{}] * num_shards not ok..
      # x = [{}] * num_shards
      x = []
      for j in range(num_shards):
        m = {}
        for i, key in enumerate(keys):
          #x[j][key] = batch_datas[i][j]
          m[key] = batch_datas[i][j]
        x.append(m)

      y = batch_datas[-1]  
      return x, y

      # for i, key in enumerate(keys):
      #   for j in range(num_shards):
      #     x[j][key] = batch_datas[i][j]
      # y = batch_datas[-1]
      # print('-----------x', x)
      # print('-----------y', y)
      # return x, y
      #return batch_datas


def get_num_gpus():
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print("os.environ['CUDA_VISIBLE_DEVICES']", os.environ['CUDA_VISIBLE_DEVICES'])
    if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
      return 0
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    logging.info('CUDA_VISIBLE_DEVICES is %s'%(os.environ['CUDA_VISIBLE_DEVICES']))
    return num_gpus
  else:
    return None

def is_cudnn_cell(cell):
  return isinstance(cell, (tf.contrib.cudnn_rnn.CudnnGRU, tf.contrib.cudnn_rnn.CudnnLSTM))

# TODO now for hadoop can only run tf 1.2 
try:
  rnn_cells = {
    'basic_lstm': tf.contrib.rnn.BasicLSTMCell,
    'lstm': tf.contrib.rnn.LSTMCell, #LSTMCell is faster then BasicLSTMCell
    'gru': tf.contrib.rnn.GRUCell,
    'lstm_block': tf.contrib.rnn.LSTMBlockCell, #LSTMBlockCell is faster then LSTMCell
    'lstm_block_fused': tf.contrib.rnn.LSTMBlockFusedCell,
    'cudnn_lstm': tf.contrib.cudnn_rnn.CudnnLSTM,
    'cudnn_gru': tf.contrib.cudnn_rnn.CudnnGRU,
    'cudnn_compat_lstm': tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell,
    'cudnn_compat_gru': tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
    }
except Exception:
  rnn_cells = {
    'basic_lstm': tf.contrib.rnn.BasicLSTMCell,
    'lstm': tf.contrib.rnn.LSTMCell, #LSTMCell is faster then BasicLSTMCell
    'gru': tf.contrib.rnn.GRUCell,
    'lstm_block': tf.contrib.rnn.LSTMBlockCell, #LSTMBlockCell is faster then LSTMCell
    'lstm_block_fused': tf.contrib.rnn.LSTMBlockFusedCell,
    }
#from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py lstm_cell()
def create_rnn_cell(num_units, is_training=True, initializer=None, forget_bias=1.0, num_layers=1, 
                    keep_prob=1.0, input_keep_prob=1.0, Cell=None, cell_type='lstm', scope=None):
  with tf.variable_scope(scope or 'create_rnn_cell') as scope:
    if initializer:
       scope.set_initializer(initializer)
    if Cell is None:
      Cell = rnn_cells.get(cell_type.lower(), tf.contrib.rnn.LSTMCell)
      print('cell:', Cell, file=sys.stderr)

    if Cell is tf.contrib.cudnn_rnn.CudnnGRU or Cell is tf.contrib.cudnn_rnn.CudnnLSTM:
      cell = Cell(num_layers=num_layers, num_units=num_units, dropout=(1. - keep_prob))
      return cell 

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def cell_():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          Cell.__init__).args:
        if 'forget_bias' in inspect.getargspec(Cell.__init__).args:
          return Cell(
              num_units, forget_bias=forget_bias,
              reuse=tf.get_variable_scope().reuse)
        else:
          return Cell(
              num_units, 
              reuse=tf.get_variable_scope().reuse)
      else:
        if 'state_is_tuple' in inspect.getargspec(
          Cell.__init__).args:
          if 'forget_bias' in inspect.getargspec(
            Cell.__init__).args:
            return Cell(num_units, forget_bias=forget_bias, state_is_tuple=True)
          else:
            return Cell(num_units, state_is_tuple=True)
        else:
          if 'forget_bias' in inspect.getargspec(
            Cell.__init__).args:
            return Cell(num_units, forget_bias=forget_bias)
          else:
            return Cell(num_units)

    attn_cell = cell_
    if is_training and (keep_prob < 1 or input_keep_prob < 1):
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            cell_(), 
            input_keep_prob=input_keep_prob,
            output_keep_prob=keep_prob)

    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)
    else:
      cell = attn_cell()
    #--now cell share graph by default so below is wrong.. will share cell for each layer
    ##cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True) 
    return cell

def unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]

#-------for train flow
def show_precision_at_k(result, k=1):
  if len(result) == 1:
    accuracy = result
    print('precision@%d:'%k, '%.3f'%accuracy) 
  else:
    loss = result[0]
    accuracy = result[1]
    print('loss:', '%.3f'%loss, 'precision@%d:'%k, '%.3f'%accuracy)

def print_results(results, names=None):
  """
  standard result print
  """
  results = gezi.get_singles(results)
  if names is None:
    print(gezi.pretty_floats(results))
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    if len(names) == len(results):
      print(gezi.get_value_name_list(results, names))
    else:
      print(gezi.pretty_floats(results))

def logging_results(results, names, tag=''):\
  logging.info('\t'.join(
    [tag] + ['%s:[%.4f]'%(name, result) for name, result in zip(names, results)]))
      
def parse_results(results, names=None):
  if type(results[0]) is str:
    temp = results 
    results = names 
    names = temp
  #only single values in results!
  if names is None:
    return gezi.pretty_floats(results)
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    if len(names) == len(results):
      return gezi.get_value_name_list(results, names)
    else:
      return gezi.pretty_floats(results)

def value_name_list_str(results, names=None):
  if names is None:
    return gezi.pretty_floats(results)
  else:
    return gezi.get_value_name_list(results, names)

#-------model load
#def get_model_path(model_dir, model_name=None):
#  """
#  if model_dir ok return latest model in this dir
#  else return orginal model_dir as a model path
#  NOTICE not check if this model path is ok(assume it to be ok) 
#  """
#  model_path = model_dir
#  ckpt = tf.train.get_checkpoint_state(model_dir)
#  if ckpt and ckpt.model_checkpoint_path:
#    #@NOTICE below code will be ok int tf 0.10 but fail int 0.11.rc tensorflow ValueError: Restore called with invalid save path
#    #do not use  ckpt.model_checkpoint_path for we might copy the model to other path so the absolute path(you might train using absoluate path) will not match
#    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
#    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
#  else:
#    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
#  #assert os.path.exists(model_path), model_path
#  #tf.logging.log_if(tf.logging.WARN, '%s not exist'%model_path, not os.path.exists(model_path))
#  #if not os.path.exists(model_path):
#    #model_path = None 
#    #tf.logging.WARN('%s not exist'%model_path)
#    #raise FileNotFoundError(model_path)
#    #raise ValueError(model_path)
#  return model_path 

def latest_checkpoint(model_dir):
  return get_model_path(model_dir)

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return gezi.dirname(model_path), model_path

def get_model_dir(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return gezi.dirname(model_path)

def get_model_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return model_path


#cat checkpoint 
#model_checkpoint_path: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-252000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-253000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-254000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-255000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
def recent_checkpoint(model_dir, latest=False):
  index = -1 if latest else 1
  return open('%s/checkpoint'%(model_dir)).readlines()[index].split()[-1].strip('"')

def checkpoint_exists_in(model_dir):
  if not os.path.exists(model_dir):
    return False
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return True
  elif os.path.isdir(model_dir):
    return False
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return True

def get_model_step(model_path):
  return int(model_path.split('/')[-1].split('-')[-1]) 

def get_model_epoch(model_path):
  try:
    return float(model_path.split('/')[-1].split('-')[-2]) 
  except Exception:
    return None

def get_model_epoch_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  try:
    return float(model_path.split('/')[-1].split('-')[-2]) 
  except Exception:
    return None

def get_model_step_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  return int(model_path.split('/')[-1].split('-')[-1]) 

def save_model(sess, model_dir, step):
  checkpoint_path = os.path.join(model_dir, 'model.ckpt')
  tf.train.Saver().save(sess, checkpoint_path, global_step=step)

def restore(sess, model_dir, var_list=None, model_name=None):
  assert model_dir
  if var_list is None:
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_dir)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))
    var_list = slim.get_variables_to_restore(include=varnames_in_checkpoint)
  saver = tf.train.Saver(var_list)
  model_path = get_model_path(model_dir, model_name)
  #assert model_path and os.path.exists(model_path), model_path
  saver.restore(sess, model_path)
  #@TODO still write to file ? using >
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver

def restore_from_path(sess, model_path, var_list=None):
  if var_list is None:
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_path)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))
    var_list = slim.get_variables_to_restore(include=varnames_in_checkpoint)
  saver = tf.train.Saver(var_list)
  saver.restore(sess, model_path)
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver

def restore_scope_from_path(sess, model_path, scope):
  variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  saver = tf.train.Saver(variables)
  saver.restore(sess, model_path)
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver

def load(model_dir, model_name=None):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore(sess, model_dir, model_name)
  return sess

def load_from_path(model_path):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore_from_path(sess, model_path)
  return sess

def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  # TODO check index replace meta
  files = [file for file in glob.glob('%s/model.ckpt-*'%(model_dir)) if not file.endswith('.index')]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files 

def variables_with_scope(scope):
    #scope is a top scope here, otherwise change startswith part
    return [v for v in tf.all_variables() if v.name.startswith(scope)]

import numpy 
#@TODO better
def npdtype2tfdtype(data_npy):
  if data_npy.dtype == numpy.float32:
    return tf.float32
  if data_npy.dtype == numpy.int32:
    return tf.int32
  if data_npy.dtype == numpy.int64:
    return tf.int64
  if data_npy.dtype == numpy.float64:
    return tf.float32
  return tf.float32

def load_constant(data_npy, sess=None, trainable=False, 
                  dtype=None, shape=None, name=None):
  """
  tf.constant only can be used for small data
  so melt.constant means melt.large_constant and have more general usage
  https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
  """
  name=name or 'constant_data'

  if not hasattr(load_constant, 'constants'):
    load_constant.constants = {}

  if name in load_constant.constants:
    return load_constant.constants[name]

  #or if isinstance(data_npy, str)
  if type(data_npy) is str:
    timer = gezi.Timer('np load %s' % data_npy)
    data_npy = np.load(data_npy)
    timer.print_elapsed()

  if dtype is None:
    dtype = npdtype2tfdtype(data_npy)
  #dtype = tf.float32

  if shape is None:
    shape = data_npy.shape
  
  # BELOW is ok but since not add to collections in tf_train_flow will not save.., if add to collections=[tf.GraphKeys.GLOBAL_VARIABLES] then sess.run(init_op) still need to feed
  # data_init = tf.placeholder(dtype, shape)
  # #data = tf.get_variable(name=name, dtype=dtype, initializer=data_init, trainable=trainable, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
  # data = tf.get_variable(name=name, dtype=dtype, initializer=data_init, trainable=trainable, collections=[])
  # load_constant.constants[name] = data

  # if sess is None:
  #   sess = melt.get_session()
  # timer = gezi.Timer('sess run initializer')
  # sess.run(data.initializer, feed_dict={data_init: data_npy}) 
  # timer.print_elapsed()
  # return data
  
  # TODO below is slow strage, some times not slow.., but should use below and above is just a ungly workaround.. and it has problem not save emb.. so just use below...
  # NOTICE in tf_train_flow sess.run(init_op) will run this again, slow again! TODO better handel
  timer = gezi.Timer('constant_initializer')
  data = tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(data_npy), trainable=trainable)
  load_constant.constants[name] = data
  timer.print_elapsed()
  
  return data

def load_constant_cpu(data_npy, sess=None, trainable=False, 
                      dtype=None, shape=None, name=None):
   with tf.device('/CPU:0'):
    return load_constant(data_npy, 
        sess=sess, 
        trainable=trainable,
        dtype=dtype,
        shape=shape,
        name=name)

def reuse_variables():
  tf.get_variable_scope().reuse_variables()

#---now work.. can set attribute reuse
#def unreuse_variables():
#  tf.get_variable_scope().reuse=None

#------------------------------------tf record save @TODO move to tfrecords
def int_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  if not six.PY2:
    if isinstance(value[0], str):
      value = [x.encode() for x in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

features = lambda d: tf.train.Features(feature=d)

# Helpers for creating SequenceExample objects  copy from \tensorflow\python\kernel_tests\parsing_ops_test.py
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)


def int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[int64_feature(v) for v in values])

def bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])

def float_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[float_feature(v) for v in values])

def get_num_records_single(tf_record_file):
  return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

def get_num_records(files):
  if isinstance(files, str):
    files = gezi.list_files(files) 
  return sum([get_num_records_single(file) for file in files])

def get_num_records_print(files):
  num_records = 0
  if isinstance(files, str):
    files = gezi.list_files(files) 
  num_inputs = len(files)
  index = 0
  for file in files:
    count = get_num_records_single(file)
    print(file, count,  '%.3f'%(index / num_inputs))
    num_records += count
    index += 1
  print('num_records:', num_records)
  return num_records

def load_num_records(input):
  num_records_file = os.path.dirname(input) + '/num_records.txt'
  num_records = int(open(num_records_file).read()) if os.path.isfile(num_records_file) else 0
  return num_records

def get_num_records_from_dir(dir_):
  num_records_file = dir_ + '/num_records.txt'
  num_records = int(open(num_records_file).read()) if os.path.isfile(num_records_file) else 0
  return num_records

# def get_num_records():
#   import dconf  
#   return dconf.NUM_RECORDS 

# def get_num_steps_per_epoch(batch_size):
#   #need dconf.py with NUM_RECORDS setting 0 at first
#   import dconf  
#   num_records = dconf.NUM_RECORDS 
#   return num_records // batch_size 

#-------------histogram util 
def monitor_train_vars(collections=None):
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var, collections=collections)

class MonitorKeys():
  TRAIN = 'train_monitor'

#@FIXME seems not work get_collection always None
from tensorflow.python.framework import ops
def monitor_gradients_from_loss(loss, collections=[MonitorKeys.TRAIN]):
  grads = tf.gradients(loss, tf.trainable_variables())
  for grad in grads:
    if grad is not None:
      tf.histogram_summary(grad.op.name, grad, collections=collections)
    else:
      raise ValueError('None grad')

#TODO check op.name or .name ? diff?
def histogram_summary(name, tensor):
  tf.summary.histogram('{}_{}'.format(name, tensor.op.name), tensor)

def scalar_summary(name, tensor):
  tf.summary.scalar('{}/{}'.format(name, tensor.op.name), tensor)

def monitor_embedding(emb, vocab, vocab_size):
  try:
    histogram_summary('emb_0', tf.gather(emb, 0))
    histogram_summary('emb_1', tf.gather(emb, 1))
    histogram_summary('emb_2', tf.gather(emb, 2))
    histogram_summary('emb_1/4', tf.gather(emb, vocab_size // 4))
    histogram_summary('emb_middle', tf.gather(emb, vocab_size // 2))
    histogram_summary('emb_3/4', tf.gather(emb, vocab_size // 4 * 3))
    histogram_summary('emb_end', tf.gather(emb, vocab_size - 1))
    histogram_summary('emb_end2', tf.gather(emb, vocab_size - 2))
    histogram_summary('emb_start_id', tf.gather(emb, vocab.start_id()))
    histogram_summary('emb_end_id', tf.gather(emb, vocab.end_id()))
    histogram_summary('emb_unk_id', tf.gather(emb, vocab.unk_id()))
  except Exception:
    print('monitor_embedding fail', file=sys.stderr)

def visualize_embedding(emb, vocab_txt='vocab.txt'):
  # You can add multiple embeddings. Here we add only one.
  embedding = melt.flow.projector_config.embeddings.add()
  embedding.tensor_name = emb.name
  # Link this tensor to its metadata file (e.g. labels).
  if not vocab_txt.endswith('.project'):
    if vocab_txt.endswith('.bin'):
      embedding.metadata_path = vocab_txt.replace('.bin', '.project')
    elif vocab_txt.endswith('.txt'):
      embedding.metadata_path = vocab_txt.replace('.txt', '.project')
    else:
      embedding.metadata_path = vocab_txt[:vocab_txt.rindex('.')] + '.project'

def get_summary_ops():
  return ops.get_collection(ops.GraphKeys.SUMMARIES)

def print_summary_ops():
  print('summary ops:')
  sops = ops.get_collection(ops.GraphKeys.SUMMARIES)
  for sop in sops:
    print(sop)

def print_global_varaiables(sope=None):
  for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(item)

def print_varaiables(key, sope=None):
  for item in tf.get_collection(key):
    print(item)

def get_global_int(key, val=0):
  if key not in os.environ:
    return val
  return int(os.environ[key])

def get_global_float(key, val=0.):
  if key not in os.environ:
    return val
  return float(os.environ[key])

def get_global_str(key):
  if key not in os.environ:
    os.environ[key] = ''
  return os.environ[key]

def step():
  return get_global_int('step', 0.)

def epoch():
  return get_global_float('epoch', 0.)

def batch_size():
  return get_global_int('batch_size')

def num_gpus():
  return get_global_int('num_gpus', 1)

def loss():
  loss_ = get_global_str('eval_loss')
  if not loss_:
    loss_ = get_global_str('train_loss')
  if not loss_:
    loss_ = get_global_str('loss')
  return loss_

def train_loss():
  return get_global_str('train_loss')

def eval_loss():
  return get_global_str('eval_loss')

def duration():
  return get_global_float('duration')

def set_global(key, value):
  os.environ[key] = str(value)

#def step():
#  return melt.flow.global_step

#def epoch():
#  return melt.flow.global_epoch

#---------for flow
def default_names(length):
  names = ['metric%d'%(i - 1) for i in range(length)]
  names[0] = 'loss'
  return names 

# TODO better handle, just use op.name , but right now has some problem
# especially optimizer will change op.name ... not readable so now 
# you have to pass the name by yourself manully
def adjust_names(ops, names):
  assert ops
  if names is None:
    #return [x.name.split('/')[-1].split(':')[0] for x in ops]
    #return [x.name for x in ops]
    return default_names(len(ops))
  else:
    if len(names) == len(ops):
      return names
    elif len(names) + 1 == len(ops):
      names.insert(0, 'loss')
      return names
    elif len(names) + 2 == len(ops):
      names.insert(0, 'loss')
      names.insert(1, 'lr')
    else:
      #return [x.name.split('/')[-1].split(':')[0]for x in ops]
      #return [x.name for x in ops]
      return default_names(len(ops))

def add_summarys(summary, values, names, suffix='', prefix=''):
  for value, name in zip(values, names):
    if suffix:
      summary.value.add(tag='%s/%s'%(name, suffix), simple_value=float(value))
    else:
      if prefix:
        summary.value.add(tag='%s/%s'%(prefix, name), simple_value=float(value))
      else:
        summary.value.add(tag=name, simple_value=float(value))

def add_summary(summary, value, name, suffix='', prefix=''):
  if suffix:
    summary.value.add(tag='%s/%s'%(name, suffix), simple_value=float(value))
  else:
    if prefix:
      summary.value.add(tag='%s/%s'%(prefix, name), simple_value=float(value))
    else:
      summary.value.add(tag=name, simple_value=float(value))

#-----------deal with text  TODO move 
# TODO pad for weights start end only zero right now!
import melt

# TODO
def pad_weights(text, weights, start_id=None, end_id=None, end_weight=1.0):
  pass

# TODO simplify without weights
def pad(text, start_id=None, end_id=None, weights=None, end_weight=1.0):
  logging.info('Pad with start_id', start_id, ' end_id', end_id)
  need_start_mark = start_id is not None
  need_end_mark = end_id is not None
  if not need_start_mark and not need_end_mark:
    return text, melt.length(text), weights 
  
  batch_size = tf.shape(text)[0]
  zero_pad = tf.zeros([batch_size, 1], dtype=text.dtype)

  sequence_length = melt.length(text)

  if not need_start_mark:
    text = tf.concat([text, zero_pad], 1)
    if weights is not None:
      weights = tf.concat([weights, tf.ones_like(zero_pad, dtype=tf.float32) * end_weight], 1)
  else:
    if need_start_mark:
      start_pad = zero_pad + start_id
      if need_end_mark:
        text = tf.concat([start_pad, text, zero_pad], 1)
        if weights is not None:
          weights = tf.concat([tf.zeros_like(start_pad, dtype=tf.float32), weights, tf.ones_like(zero_pad, dtype=tf.float32) * end_weight], 1)
      else:
        text = tf.concat([start_pad, text], 1)
        if weights is not None:
          weights = tf.concat([tf.zeros_like(start_pad, dtype=tf.float32), weights], 1)
      sequence_length += 1

  if need_end_mark:
    text = melt.dynamic_append_with_length(
        text, 
        sequence_length, 
        tf.constant(end_id, dtype=text.dtype)) 
    if weights is not None:
      weights = melt.dynamic_append_with_length_float32(
        weights, 
        sequence_length, 
        tf.constant(end_weight, dtype=weights.dtype)) 
    sequence_length += 1

  return text, sequence_length, weights

class GpuHanler(object):
  def __init__(self, num_gpus=None):
    self._cur_gpu = 0

  def next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev


def count_records(files):
  import  multiprocessing
  from multiprocessing import Value

  counter = Value('i', 0)
  def deal_file(file):
    try:
      count = melt.get_num_records_single(file)
    except Exception:
      print('bad file:', file)
    global counter
    with counter.get_lock():
      counter.value += count 

  pool = multiprocessing.Pool()
  pool.map(deal_file, files)
  pool.close()
  pool.join()  

  return counter.value
