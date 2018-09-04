#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-17 10:30:20.286494
#   \Description  
# ==============================================================================

"""
not supporting averaging and multi gpu yet  @TODO
 [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)

 here what we do is 
 create train_op from loss
 and may be using multi gpu deal
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim

__pacage__ = None 

import sys 
import os 

import gezi
import melt 

#or from melt.utils import logging
import melt.utils.logging as logging
#import logging

import tensorflow as tf

if sys.version_info > (3,):
  long = int

flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

# new tf seems to try to use absl which has def of log_dir TODO
try:
  flags.DEFINE_string('log_dir', None, '''if none will set to model_dir
                                        ''')
except Exception:
  pass

#-------input data
flags.DEFINE_integer('batch_size', 32, 'Batch size. default as im2text default')
flags.DEFINE_integer('eval_batch_size', 100, 'Batch size fore eval')

#-------flow
flags.DEFINE_integer('num_epochs', 0, '''Number of epochs to run trainer.
                                         0 means run forever epochs,
                                         -1 mens you want to just run 1 step and stop!, usefull when you want to do eval all once''')
flags.DEFINE_integer('num_steps', 0, '''Number of steps to run trainer. 0 means run forever, 
                                        -1 means you just want to build graph and save without training(changing model value)''')
#-------model
flags.DEFINE_boolean('save_model', True, '')
flags.DEFINE_float('save_interval_epochs', 1, 'if 0 will not save, by default 1 epoch 1 model in modeldir/epoch, you can change to 2, 0.1 etc')
flags.DEFINE_float('save_interval_seconds', 0, 'model/checkpoint save interval by n seconds, if > 0 will use this other wise use save_interval_hours')
flags.DEFINE_float('save_interval_hours', 10000, """model/checkpoint save interval by n hours""")
flags.DEFINE_float('save_interval_steps', 1000, 'model/checkpoint save interval steps')
flags.DEFINE_bool('freeze_graph', True, '''if True like image caption set this to True, will load model faster when inference,
                                            but sometimes it is slow to freeze, might due to tf bug, then you can set it to False,
                                            if you do not need inference also set to False to speed up training,
                                            TODO FIXME for cudnn right now if not set feeze graph can not load, is a bug of melt ?''')
flags.DEFINE_integer('max_models_keep', 2, 'keep recent n models, default 2 for safe')
flags.DEFINE_boolean('restore_from_latest', True, 'more safe to restore from recent but not latest')

#--------show
flags.DEFINE_integer('interval_steps', 100, '')
flags.DEFINE_integer('eval_interval_steps', 1000, """for training suggest 10000, 
                                                     you can check evaluate interval time 
                                                     and evaluate once time the ratio below 0.1""")
flags.DEFINE_integer('metric_eval_interval_steps', 0, 'if > 0 need to be eval_interval_steps * n')
flags.DEFINE_boolean('metric_eval', True, '')

flags.DEFINE_float('train_interval_epochs', 1, '')
flags.DEFINE_float('valid_interval_epochs', 1, '')
flags.DEFINE_float('test_interval_epochs', 1, '')
flags.DEFINE_float('inference_interval_epochs', 1, '')
flags.DEFINE_boolean('write_during_train', True, '')

flags.DEFINE_integer('eval_loops', 1, 'set to max inorder to hack for evaluation..')

#----------optimize
flags.DEFINE_string('optimizer', 'adadelta', 'follow squad of ukhst https://www.quora.com/Why-is-AdaDelta-not-favored-in-Deep-Learning-communities-while-AdaGrad-is-preferred-by-many-over-other-SGD-variants')
flags.DEFINE_float('momentum', 0.9, 'follow cifar10 default')
flags.DEFINE_float('opt_epsilon', 1e-6, 'follow squad of ukhst')
flags.DEFINE_float('learning_rate', 0.5, """follow squad of ukhst
                                            Initial learning rate. for adgrad especially, 
                                            notice keras set for adgrad 0.01 
                                            but seems bad perf hard to converge for some seq2seq/lstm training
                                            see textsum/train/shangpinming/seq2seq-gen-copy-switch.sh""")
flags.DEFINE_float('min_learning_rate', 0.005, 'min learning rate used for dyanmic eval metric decay')

#flags.DEFINE_float('learning_rate_decay_factor', 0.97, 'im2txt 0.5, follow nasnet using 0.97')
flags.DEFINE_float('learning_rate_decay_factor', 0., 'im2txt 0.5, follow nasnet using 0.97')
flags.DEFINE_boolean('dynamic_learning_rate', False, '')
flags.DEFINE_integer('learning_rate_patience', None, 'might be 3 for 3 times eval loss no decay')

flags.DEFINE_float('decay_start_epoch', 0., 'start decay from epoch')
flags.DEFINE_float('num_epochs_per_decay', 2.4, 'follow nasnet')
flags.DEFINE_integer('decay_start_step', 0, 'start decay from steps')
flags.DEFINE_integer('num_steps_per_decay', 0, 'if 0 no effect, if > 0 then will not use this instead of num_epochs_per_decay')
flags.DEFINE_string('learning_rate_values', None, 'like 0.1,0.05,0.005')
flags.DEFINE_string('learning_rate_step_boundaries', None, 'like 10000,20000')
flags.DEFINE_string('learning_rate_epoch_boundaries', None, 'like 10,30 or 10.5,30.6')
#flags.DEFINE_string('lr_ratios', None, '0.2,1,0.2,1,1,1')
flags.DEFINE_boolean('use_finetune_step', False, '')
flags.DEFINE_boolean('use_finetune2_step', False, '')

flags.DEFINE_float('clip_gradients', 5.0, """follow im2text as 5.0 default, 
                                          set to 1.0 in deeipiu/image_caption try sh ./train/flickr-rnn.sh, 
                                          will show perf from 900inst/s to 870ints/s and also slow convergence""")
flags.DEFINE_boolean('optimize_has_scope', True, '')

#----------train
flags.DEFINE_boolean('train_only', False, '')
flags.DEFINE_boolean('train_all', False, 'use for small dataset like competetion or standard dataset where use k folds for train/valid and use all k parts if set train_all==True')
flags.DEFINE_string('work_mode', 'full', 'full/train_valid_show_metric, train, test, train_metric, train_valid, train_valid_metric')
flags.DEFINE_integer('monitor_level', 2, '1 will monitor emb, 2 will monitor gradient')
flags.DEFINE_integer('log_level', 0, '')
flags.DEFINE_boolean('no_log', False, '')
flags.DEFINE_string('mode', 'train', 'or predict')
flags.DEFINE_boolean('freeze_graph_collections', True, '')
flags.DEFINE_integer('random_seed', None, '')

flags.DEFINE_boolean('use_tower_loss', True, '')

#----------multi gpu
##TODO be carefull to use mult gpu mode, since for some case it will casue performance lower then single gpu mode 
##especially for finetune image model or some models that will cause sess.run summary catch exception FIXME
##also google paper lessons from 2015 coco caption contest show and tell model will have low performance also using 
##multi gpu so they use single gpu training
flags.DEFINE_integer('num_gpus', 0, """How many GPUs to use. set 0 to disable multi gpu mode""")
flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
flags.DEFINE_boolean('batch_size_per_gpu', True, '''  per gpu batch size should be dived by gpu num ?
                                                      True means, if num_gpus = 2, batch_size set 128, then each gpu with batch size 128, which means batch_size is actually batch_size_per_gpu
                                                      means 256 insts per step actually, if num_gpus == 0, will try to read env info if you set like 
                                                      CUDA_VISIABLE_DIVICES=0,1 then actually will use 2 GPUS, and 256 insts per step also 
                                                      if not find CUDA_VISIABLE_DIVICES infoep then, just use 1 GPU, single gpu mode, 128 insts per step
                                                      if num_gpus == 1, also 1GPU, 128 insts per step, but will use tower_loss with 1 gpu(mainly for test if tower_loss ok)
                                                      not used much, so if you want use single gpu, just set num_gpus=0 
                                                      if batch_size_per_gpu = False, with 2gpu, then it means each GPU will be of batch size 128 / 2 = 64, total insts 
                                                      are still 128 per step
                                                      batch_size_per_gpu False is better for debug, all program, deal same num instances so better for comparation
                                                      batch_size_per_gpu True is better for speed up, fully use multi gpu, like one gpu can only train batch 32(OOM for
                                                      bigger batch_size) then 2 gpu can deal 2 * 32 instances per step
                                                      For experiment simplicity, set it to True by default, same instances per step if increase gpu num
                                                     ''') 
flags.DEFINE_string('variable_strategy', 'cpu', '')


#----------scope
flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                      set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')
flags.DEFINE_string('restore_include', None, '')
flags.DEFINE_string('restore_exclude', None, '')
flags.DEFINE_string('main_scope', 'main', 'or use other main_scope like run, this is mainly graph scope for varaible reuse')

#----- cross fold
flags.DEFINE_integer('fold', None, '')
flags.DEFINE_integer('num_folds', None, '')

#---------- input reader
if tf.__version__ > '1.5':
  flags.DEFINE_boolean('using_dataset', True, '')
else:
  flags.DEFINE_boolean('using_dataset', False, '')
  
flags.DEFINE_string('buckets', None, 'empty meaning not use, other wise looks like 5,10,15,30')
flags.DEFINE_string('batch_sizes', None, '')

flags.DEFINE_integer('min_after_dequeue', 0, """by deafualt will be 500, 
                                                set to large number for production training 
                                                for better randomness""")
flags.DEFINE_integer('num_prefetch_batches', 0, '')

#---------- input dirs
#@TODO will not use input pattern but use dir since hdfs now can not support glob well
flags.DEFINE_string('train_input', None, 'must provide')
flags.DEFINE_string('train_input2', None, 'must provide')
flags.DEFINE_string('valid_input', None, 'if empty will train only')
flags.DEFINE_string('valid_input2', None, 'some applications might need another valid input')
flags.DEFINE_string('test_input', None, 'maily for inference during train epochs')
flags.DEFINE_string('fixed_valid_input', None, 'if empty wil  not eval fixed images')
flags.DEFINE_string('num_records_file', None, '')
flags.DEFINE_string('base_dir', '../../../mount', '')

flags.DEFINE_boolean('show_eval', True, '')
flags.DEFINE_boolean('eval_shuffle_files', True, '')
flags.DEFINE_boolean('eval_fix_random', True, '')
flags.DEFINE_integer('eval_seed', 1024, '')
flags.DEFINE_integer('seed', None, '1024 input seed')
flags.DEFINE_boolean('fix_sequence', False, '')

flags.DEFINE_string('big_buckets', None, 'empty meaning not use, other wise looks like 5,10,15,30')
flags.DEFINE_string('big_batch_sizes', None, '')
flags.DEFINE_integer('big_batch_size', None, '')

flags.DEFINE_boolean('adjust_global_step', False, '')

flags.DEFINE_boolean('use_eager', False, '')


flags.DEFINE_integer('num_threads', 12, """threads for reading input tfrecords,
                                           setting to 1 may be faster but less randomness
                                        """)


inited = None 

def init(): 
  if FLAGS.use_eager:
    tf.enable_eager_execution()

  if 'FOLD' in os.environ:
    try:
      FLAGS.fold = int(os.environ['FOLD'])
    except Exception:
      sub_dir = os.environ['FOLD']
      assert os.path.isdir(FLAGS.model_dir)
      FLAGS.model_dir = os.path.join(FLAGS.model_dir, sub_dir)
      pass
  
  if not FLAGS.log_dir:
    if 'log_dir' in os.environ and os.environ['log_dir']:
      FLAGS.log_dir = os.environ['log_dir']
  if not FLAGS.log_dir:
    #if not os.path.isdir(FLAGS.model_dir):
    #  FLAGS.log_dir = os.path.dirname(FLAGS.model_dir)
    #else:
    FLAGS.log_dir = FLAGS.model_dir

  if os.path.isfile(FLAGS.model_dir + '.index'):
    FLAGS.log_dir = os.path.dirname(FLAGS.model_dir)

  assert FLAGS.log_dir, 'you need to set log_dir or model_dir'
  print('model_dir', FLAGS.model_dir, 'log_dir', FLAGS.log_dir, file=sys.stderr)
  os.system('mkdir -p %s' % FLAGS.log_dir)
  logging.set_logging_path(FLAGS.log_dir)

  if 'VLOG' in os.environ:
    FLAGS.log_level = int(os.environ['VLOG'])
  logging.info('log_level:', FLAGS.log_level)

  if 'DOUBLE_BATCH' in os.environ:
    FLAGS.batch_size *= 2

  if 'BIG' in os.environ and int(os.environ['BIG']) == True:
    if FLAGS.big_batch_size is not None:
      FLAGS.batch_size = FLAGS.big_batch_size
    if FLAGS.big_buckets is not None:
      FLAGS.buckets = FLAGS.big_buckets
    if FLAGS.big_batch_sizes is not None:
      FLAGS.batch_sizes = FLAGS.big_batch_sizes

  if 'NUM_EPOCHS' in os.environ:
    FLAGS.num_epochs = int(os.environ['NUM_EPOCHS'])
    logging.info('Using num_epochs set from env as %d' % FLAGS.num_epochs)
  
  if 'NUM_STEPS' in os.environ:
    FLAGS.num_steps = int(os.environ['NUM_STEPS'])
    logging.info('Using num_steps set from env as %d' % FLAGS.num_steps)
    if FLAGS.num_steps < 0:
      FLAGS.train_only = True 

  if 'EVAL_STEP' in os.environ:
    FLAGS.metric_eval_interval_steps = int(os.environ['EVAL_STEP'])
    FLAGS.eval_interval_steps = int(os.environ['EVAL_STEP'])

  if 'EVAL_STEPS' in os.environ:
    FLAGS.metric_eval_interval_steps = int(os.environ['EVAL_STEPS'])
    FLAGS.eval_interval_steps = int(os.environ['EVAL_STEPS'])

  if 'LEARNING_RATE_DECAY_FACTOR' in os.environ:
    FLAGS.learning_rate_decay_factor = int(os.environ['LEARNING_RATE_DECAY_FACTOR'])
    logging.info('Using learning_rate_decay_factor set from env as %d' % FLAGS.learning_rate_decay_factor)

  if 'BUCKETS' in os.environ:
    FLAGS.buckets = os.environ['BUCKETS']
    logging.info('Using buckets set from env as ', FLAGS.buckets)

  if 'BATCH_SIZES' in os.environ:
    FLAGS.batch_sizes = os.environ['BATCH_SIZES']
    logging.info('Using batch sizes set from env as ', FLAGS.batch_sizes)

  if 'NUM_LAYERS' in os.environ:
    FLAGS.num_layers = int(os.environ['NUM_LAYERS'])
    logging.info('Using num layers set from env as ', FLAGS.num_layers)

  # TEST means also inference
  if 'TEST_INPUT' in os.environ:
    if int(os.environ['TEST_INPUT']) != 0:    
      if not FLAGS.test_input:
        assert FLAGS.train_input
        assert 'train' in  FLAGS.train_input
        FLAGS.test_input = FLAGS.train_input.replace('train', 'test').split(',')[0]
    else:
      FLAGS.test_input = None
  # else:
  #   if not ('TRAIN_ALL' in os.environ and int(os.environ['TRAIN_ALL']) != 0):
  #     FLAGS.test_input = None

  if FLAGS.test_input == '1':
    assert FLAGS.train_input
    assert 'train' in  FLAGS.train_input
    FLAGS.test_input = FLAGS.train_input.replace('train', 'test').split(',')[0]
  elif FLAGS.test_input == '0':
    FLAGS.test_input = None

  if FLAGS.valid_input == '1':
    assert FLAGS.train_input
    assert 'train' in  FLAGS.train_input
    FLAGS.valid_input = FLAGS.train_input.replace('train', 'valid').split(',')[0]
  elif FLAGS.valid_input == '0':
    FLAGS.valid_input = None

  if 'TRAIN_ALL' in os.environ and int(os.environ['TRAIN_ALL']) != 0:
    FLAGS.train_all = True
    #FLAGS.buckets = None  # no buckets for train all mode TODO might still need buket

    if FLAGS.fold is not None:
      if FLAGS.model_dir:
        FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'all')

    # also evluate on fold 0 if not set fold
    if FLAGS.fold is None:
      FLAGS.fold = 0

    if not FLAGS.test_input:
      assert FLAGS.train_input
      assert 'train' in  FLAGS.train_input
      FLAGS.test_input = FLAGS.train_input.replace('train', 'test').split(',')[0]
  
  if 'TRAIN_ONLY' in os.environ and int(os.environ['TRAIN_ONLY']) != 0:
    FLAGS.train_all = True
    FLAGS.train_only = True

    if FLAGS.fold is not None:
      if FLAGS.model_dir:
        FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'all')

    try:
      FLAGS.fold = None # tf 1.4 not ok to set None
    except Exception:
      pass

  if FLAGS.train_only:
    FLAGS.test_input = None 
    FLAGS.valid_input = None
    FLAGS.metric_eval = False
    
  if 'MIN_AFTER_DEQUEUE' in os.environ:
    FLAGS.min_after_dequeue = int(os.environ['MIN_AFTER_DEQUEUE'])

  if 'BUFFER_SIZE' in os.environ:
    FLAGS.min_after_dequeue = int(os.environ['BUFFER_SIZE'])
  
  if 'RANDOM_EMB' in os.environ and os.environ['RANDOM_EMB'] == '1':
    FLAGS.word_embedding_file = None

  if not FLAGS.num_gpus:
    FLAGS.num_gpus = melt.get_num_gpus()

  if not FLAGS.use_tower_loss:
    FLAGS.num_gpus = 0
  if not FLAGS.batch_size_per_gpu and FLAGS.num_gpus > 1:
    print('batch size is shrink by %d for each gpu to make total insts per step still %d'%(FLAGS.num_gpus, FLAGS.batch_size), file=sys.stderr)
    FLAGS.batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

  num_gpus = FLAGS.num_gpus
  melt.set_global('num_gpus', max(num_gpus, 1))

  melt.set_global('batch_size', FLAGS.batch_size * melt.num_gpus())

  logging.info('batch_size:', melt.batch_size(), 'batch_size_per_gpu:', FLAGS.batch_size, 'num_gpus:', melt.num_gpus())

  # #HACK
  # if FLAGS.cell == 'cudnn_gru' and tf.__version__ < '1.6':
  #   FLAGS.eval_batch_size = FLAGS.batch_size

  try:
    if tf.__version__ < '1.5' and FLAGS.using_dataset:
      FLAGS.using_dataset = False
  except Exception:
    pass

  if FLAGS.learning_rate_patience:
    FLAGS.dynamic_learning_rate = True
    if not FLAGS.learning_rate_decay_factor:
      FLAGS.learning_rate_decay_factor = 0.5

  logging.info('min_after_dequeue:{}'.format(FLAGS.min_after_dequeue))

  global inited
  inited = True

def get_global_scope():
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  return global_scope

def gen_learning_rate(num_steps_per_epoch=None):
  #TODO if app crash reload then we should set smaller learning rate, may adgrad can combine with exponential_decay ?
  #copy from im2txt\im2txt\train.py
  _learning_rate_decay_fn = None
  
  #TODO righ now not to pass float learning_rate as will be variable in optimizer.py and save
  #if restore it will get learning rate form checkpoint, even if you give another learning rate 
  #this might be confusing, so just make it constant, if restart training you can change learning rate 
  #remeber if using decay by defualt beaviour will restore global_step from checkpoint
  #so you can restart and get decayed learning rate direclty, restart is same as training without start
  #you can also set smaller learning rate for example if learning_rate 0.1 before then decay learning rate is 
  #0.1 * decay you can set learning rate to 0.001 manualy after when restart training, then it means you start
  #with 0.001 * decay (0.01 * decayed_learning_rate)
  if not FLAGS.dynamic_learning_rate:
    if FLAGS.learning_rate_decay_factor > 0:
      learning_rate = tf.constant(FLAGS.learning_rate)
      #learning_rate = FLAGS.learning_rate
    else:
      learning_rate = tf.get_variable(
        "learning_rate", [],
        trainable=False,
        initializer=tf.constant_initializer(FLAGS.learning_rate),
        collections=[])
  else:
    logging.info('using dyanmic learning rate')
    learning_rate = tf.get_variable(
      "learning_rate", [],
      trainable=False,
      initializer=tf.constant_initializer(FLAGS.learning_rate))

    if FLAGS.learning_rate_patience:
      assert FLAGS.learning_rate_decay_factor > 0
      logging.info('adjust learning rate by patience {} and decay_factor *{}'.format(FLAGS.learning_rate_patience, FLAGS.learning_rate_decay_factor))
      return learning_rate, None

  logging.info('initial learning_rate:{}'.format(FLAGS.learning_rate))
  
  if not FLAGS.learning_rate > 0:
    assert FLAGS.learning_rate_values, 'if learning rate is 0 then must set learnint rate values'

  if FLAGS.learning_rate_values:
    if not FLAGS.learning_rate:
      learning_rate_values = [float(lr) for lr in FLAGS.learning_rate_values.split(',')]
      assert FLAGS.learning_rate_step_boundaries or FLAGS.learning_rate_epoch_boundaries 
      if FLAGS.learning_rate_step_boundaries:
        assert FLAGS.learning_rate_epoch_boundaries is None, 'use step boundaries or epoch boundaries?'
        boundaries = [long(bound) for bound in FLAGS.learning_rate_step_boundaries.split(',')]
      else:
        assert num_steps_per_epoch is not None, 'need epoch info if using epoch boundaries'
        boundaries = [long(float(epoch_bound) * num_steps_per_epoch) for epoch_bound in FLAGS.learning_rate_epoch_boundaries.split(',')]
      
      assert len(learning_rate_values) == len(boundaries) + 1, \
        'len_values:{} len_bouddaries:{}'.format(len(learning_rate_values), len(boundaries))

      logging.info('learning rate values:{}, epoch_bounds:{} boundaries:{}'.format(
          FLAGS.learning_rate_values, FLAGS.learning_rate_epoch_boundaries, ','.join(map(str, boundaries))))
      def _learning_rate_decay_fn(learning_rate, global_step):
        #return tf.train.piecewise_constant(
        return melt.train.piecewise_constant(
          global_step,
          boundaries, 
          learning_rate_values)
  elif FLAGS.learning_rate_decay_factor > 0:
    #assert FLAGS.learning_rate_values is None, 'use exponential_decay or piecewise_constant?'
    #NOTICE if you do finetune or other things which might change batch_size then you'd better direclty set num_steps_per_decay
    #since global step / decay_steps will not be correct epoch as num_steps per epoch changed
    #so if if you change batch set you have to reset global step as fixed step
    assert FLAGS.num_steps_per_decay or (FLAGS.num_epochs_per_decay and num_steps_per_epoch), 'must set num_steps_per_epoch or num_epochs_per_decay and num_steps_per_epoch'
    decay_steps = FLAGS.num_steps_per_decay or int(num_steps_per_epoch * FLAGS.num_epochs_per_decay)    
    decay_start_step = FLAGS.decay_start_step or int(num_steps_per_epoch * FLAGS.decay_start_epoch)
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    logging.info('learning_rate_decay_factor:{} decay_epochs:{} decay_steps:{} decay_start_epoch:{} decay_start_step:{}'.format(
        FLAGS.learning_rate_decay_factor, FLAGS.num_epochs_per_decay, decay_steps, FLAGS.decay_start_epoch, decay_start_step))

    def _learning_rate_decay_fn(learning_rate, global_step):
      return melt.train.exponential_decay(
          learning_rate,
          global_step,
          decay_steps=decay_steps,
          decay_rate=FLAGS.learning_rate_decay_factor,
          decay_start_step=decay_start_step,
          staircase=True)
  else:
    logging.warning('Will ignore learning rate values since you have learning rate not 0!')

  learning_rate_decay_fn = _learning_rate_decay_fn
  return learning_rate, learning_rate_decay_fn

def train_flow(ops, 
               names=None, 
               gen_feed_dict_fn=None, 
               deal_results_fn=None, 
               eval_ops=None, 
               eval_names=None,
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=melt.print_results,
               optimizer=None, 
               learning_rate=0.1, 
               num_steps_per_epoch=None,
               model_dir=None, 
               log_dir=None,
               metric_eval_fn=None, 
               inference_fn=None,
               debug=False,
               summary_excls=None,
               init_fn=None,
               restore_fn=None,
               restore_include=None,
               restore_exclude=None,
               save_all_scope=False,
               variables_to_restore=None, 
               variables_to_save=None,
               output_collection_names=None, 
               output_node_names=None,
               sess=None):
  """
   #variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8']) not used much 
   variables_to_save might be used but will hack here since we also want to save like seq2seq/OptimizeLoss/
  """
  assert inited, 'Forget to call melt.apps.init() before using melt.apps.train_flow?'

  if sess is None:
    sess = melt.get_session()
  if debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  if FLAGS.random_seed:
    tf.set_random_seed(FLAGS.random_seed)

  model_dir = model_dir or FLAGS.model_dir
  log_dir = log_dir or FLAGS.log_dir

  logging.info('clip_gradients:{}'.format(FLAGS.clip_gradients))
  logging.info('optimizer:{}'.format(FLAGS.optimizer))

  num_gpus = melt.num_gpus()
  
  #batch size right now not define here, but in app code like input_app.py
  
  #NOTICE since melt.__init__.py with from melt.flow import * then you can not 
  #use melt.flow.train.train_flow but you can always use
  #from melt.flow.train.train_flow import train_flow

  if optimizer is None:
    optimizer = FLAGS.optimizer

  # Set up the training ops.
  #notice '' only works in tf >= 0.11, for 0.10 will always add OptimeizeLoss scope
  #the diff is 0.10 use variable_op_scope and 0.11 use variable_scope
  optimize_scope = None if FLAGS.optimize_has_scope else ''
  # NOTICE! initialzer value is step get from model check point if exits otherwise 0
  #will not get step value from checkpoint since you might change batch size it is safe to set step by epoch num and batch size
  #this is controlled by melt.apps.flow where global_step var is removed from restore var list 
  #if set num_steps_per_decay then inital step actually the same as readding from check point global step not modify for batch size change
  #be realy golbal step(not fixed global step)
  # TODO FIXME not flexible... since if you want to use global step in classifier graph.. can not tf.train.get_or_create_global_step()
  initial_step = melt.get_global_step(model_dir, num_steps_per_epoch, fix_step=(not FLAGS.num_steps_per_decay))
  logging.info('global_step init with initial_step from model_dir as %d' % initial_step)
  # TODO right now has global_scope above global_step might need to remove using function creator show_and_tell/global_step (DT_INT64) []
  # global_step = tf.get_variable(tf.GraphKeys.GLOBAL_STEP, shape=[], dtype=tf.int64, 
  #                               initializer=tf.constant_initializer(initial_step))  
  # or can use get_variable(.. collections=['global_step']) but notice her is later then you build graph... 
  # tf.add_to_collection('global_step', global_step)
  global_step = tf.train.get_or_create_global_step()
  sess.run(tf.assign(global_step, tf.constant(initial_step, dtype=tf.int64)))

  if FLAGS.use_finetune_step:
    # NOTICE unlike global step this one will be save to checkpoint and read out without any change 
    finetune_start_step = tf.get_variable('finetune_start_step', shape=[], dtype=tf.int64, 
                                           initializer=tf.constant_initializer(initial_step))
  elif FLAGS.use_finetune2_step:
    # NOTICE if 'finetune_start_step2' then will try to load finetune_start_step2 from checkpoint.. where there only fine_start_step..
    finetune_start_step = tf.get_variable('finetune2_start_step', shape=[], dtype=tf.int64, 
                                           initializer=tf.constant_initializer(initial_step))    
  else:
    finetune_start_step = 0

  logging.info('num_steps_per_epoch:', num_steps_per_epoch)
  learning_rate, learning_rate_decay_fn = gen_learning_rate(num_steps_per_epoch)
  if learning_rate_decay_fn is not None:
    learning_rate = learning_rate_decay_fn(learning_rate, global_step - finetune_start_step)

  #do not let optimizer do decay again!
  learning_rate_decay_fn = None 
  #or judge by FLAGS.num_gpus
  if FLAGS.optimizer == 'momentum':
    optimizer = lambda lr: tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum) 
  else:
    optimizer = melt.util.get_optimizer(optimizer)
  if not isinstance(ops[0], (list,tuple)):  
    # train_op = tf.contrib.layers.optimize_loss(
    #     loss=ops[0],
    #     global_step=global_step,
    #     learning_rate=learning_rate,
    #     optimizer=optimizer,
    #     clip_gradients=FLAGS.clip_gradients,
    #     learning_rate_decay_fn=learning_rate_decay_fn,
    #     name=optimize_scope)  
    train_op = melt.layers.optimize_loss(
      losses=[ops[0]],
      global_step=global_step,
      learning_rate=learning_rate,
      optimizer=optimizer,
      clip_gradients=FLAGS.clip_gradients,
      learning_rate_decay_fn=learning_rate_decay_fn,
      name=optimize_scope)
  else: 
    #---as in cifa10 example, put all but tower loss on cpu, wiki say, that will be faster,
    #but here I find without setting to cpu will be faster..
    #https://github.com/tensorflow/tensorflow/issues/4881
    #I've noticed same thing on cirrascale GPU machines - putting parameters on gpu:0 and using gpu->gpu transfer was a bit faster. I suppose this depends on particular details of hardware -- if you don't have p2p connectivity between your video cards then keeping parameters on CPU:0 gives faster training.
    #err but for my pc no p2p, with PHB connection nvidia-smi topo -m, still hurt by set cpu.. may be should not put cpu here
    # TODO...
    update_ops = ops[0][1]
    ops[0] = ops[0][0]
    if FLAGS.variable_strategy == 'cpu' and FLAGS.num_gpus and FLAGS.num_gpus > 1:
      with tf.device('/cpu:0'):
        train_op = melt.layers.optimize_loss(
            losses=ops[0],
            num_gpus=num_gpus,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_gradients=FLAGS.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn,
            update_ops=update_ops,
            name=optimize_scope)
    else:
      train_op = melt.layers.optimize_loss(
          losses=ops[0],
          num_gpus=num_gpus,
          global_step=global_step,
          learning_rate=learning_rate,
          optimizer=optimizer,
          clip_gradients=FLAGS.clip_gradients,
          learning_rate_decay_fn=learning_rate_decay_fn,
          update_ops=update_ops,
          name=optimize_scope)
      
    #set the last tower loss as loss in ops
    ops[0] = ops[0][-1]
    #ops[0] = ops[0][0]
    ## cifar10 below down 3600 -> 3100 examples/s
    # consolidation_device = '/gpu:0' if FLAGS.variable_strategy == 'gpu' else '/cpu:0'
    # with tf.device(consolidation_device):
    #   ops[0] = tf.reduce_mean(ops[0])
 
  ops.insert(0, train_op)
  ops.insert(1, learning_rate)

  tf.add_to_collection('learning_rate', learning_rate)

  learning_rate_weight = tf.get_variable('learning_rate_weight', initializer= tf.ones(shape=(), dtype=tf.float32))
  tf.add_to_collection('learning_rate_weight', learning_rate_weight)

  try:
    sess.run(tf.variables_initializer([learning_rate]))
  except Exception:
    pass

  #-----------post deal
  save_interval_seconds = FLAGS.save_interval_seconds if FLAGS.save_interval_seconds > 0 \
     else FLAGS.save_interval_hours * 3600 

  interval_steps=FLAGS.interval_steps
  eval_interval_steps=FLAGS.eval_interval_steps
  metric_eval_interval_steps=FLAGS.metric_eval_interval_steps
  save_model=FLAGS.save_model 
  save_interval_steps = FLAGS.save_interval_steps 
  num_steps = FLAGS.num_steps
  num_epochs = FLAGS.num_epochs

  if not save_interval_steps:
    save_interval_steps = 1000000000000

  if not FLAGS.metric_eval:
    metric_eval_interval_steps = 0

  if FLAGS.work_mode == 'train' or FLAGS.train_only:
    eval_ops = None 
    metric_eval_fn = None
    logging.info('running train only mode')
  elif FLAGS.work_mode == 'train_metric':
    eval_ops = None 
    assert metric_eval_fn is not None, 'set metric_eval to 1'
    logging.info('running train+metric mode')
  elif FLAGS.work_mode == 'train_valid':
    metric_eval_fn = None
    logging.info('running train+valid mode')
  elif FLAGS.work_mode.startswith('test'):
    ops = None
    logging.info('running test only mode')
    interval_steps = 0
    eval_interval_steps = 1
    metric_eval_interval_steps /= FLAGS.eval_interval_steps
    save_model = False
  elif FLAGS.work_mode.startswith('metric') or FLAGS.work_mode.startswith('eval') or gezi.env_has('METRIC'):
    #TODO name is a bit cofusing for work_mode, eval or metric means using metric evaluation
    #test above means using eval_loss(valid_loss) as composed to train_loss for evaluation
    ops = None 
    eval_ops = None
    logging.info('running metric eval only mode')
    interval_steps = 0 
    eval_interval_steps = 1
    metric_eval_interval_steps /= FLAGS.eval_interval_steps    
    save_model = False
    assert metric_eval_fn is not None 
  
  if FLAGS.work_mode.endswith('once'):
    num_epochs = -1 #hack to make only do one step!

  #TODO hack seq2seq/OptimizeLoss/seq2seq/main/decode/rnn/basic_lstm_cell/kernel/Adagrad (DT_FLOAT) [1280,4096] need to save
  if variables_to_save is not None:
    optimize_vars = set(slim.get_variables(get_global_scope() + '/OptimizeLoss'))
    assert optimize_vars, 'optimizer must has scope %s'%(get_global_scope() + '/OptimizeLoss')
    variables_to_save = list(set(variables_to_save) | optimize_vars)
    #print('final varables_to_save', variables_to_save)

  if output_collection_names is None and FLAGS.freeze_graph_collections:
    all_keys = sess.graph.get_all_collection_keys()
    exclude_keys = set(['variables', 'queue_runners', 'summaries', 'train_op', 'update_ops', 'model_variables', 'cond_context', 'while_context'])
    output_collection_names = [x for x in all_keys if x not in exclude_keys and not 'train' in x and not x.endswith('_end_points')]
  logging.info('all collection keys: {}'.format(all_keys[:100]))
  logging.info('collection names to freeze: {}'.format(output_collection_names))

  print('ops', ops, file=sys.stderr)
  print('eval_ops', eval_ops, file=sys.stderr)
  print('names', names, file=sys.stderr)
  print('eval_names', eval_names, file=sys.stderr)

  if FLAGS.learning_rate_patience:
    assert metric_eval_fn is not None, 'need to use metrci eval fn to monitor and decay learning rate'

  restore_include = restore_include or FLAGS.restore_include.split(',') if FLAGS.restore_include else None
  restore_exclude = restore_exclude or FLAGS.restore_exclude.split(',') if FLAGS.restore_exclude else None
  return melt.flow.train_flow(
             ops, 
             names=names,
             gen_feed_dict_fn=gen_feed_dict_fn,
             deal_results_fn=deal_results_fn,
             eval_ops=eval_ops,
             eval_names=eval_names,
             gen_eval_feed_dict_fn=gen_eval_feed_dict_fn,
             deal_eval_results_fn=deal_eval_results_fn,
             interval_steps=interval_steps,
             eval_interval_steps=eval_interval_steps,
             eval_loops=FLAGS.eval_loops,
             num_epochs=num_epochs,
             num_steps=num_steps,
             save_interval_seconds=save_interval_seconds,
             save_interval_steps=save_interval_steps,
             save_model=save_model,
             save_interval_epochs=FLAGS.save_interval_epochs,
             freeze_graph=FLAGS.freeze_graph,
             #optimizer=optimizer, 
             optimizer=None, #must set None since here we have done choosing optimizer
             learning_rate=learning_rate,
             learning_rate_patience=FLAGS.learning_rate_patience,
             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
             num_steps_per_epoch=num_steps_per_epoch,
             max_models_keep=FLAGS.max_models_keep,
             model_dir=model_dir,
             log_dir=log_dir,
             restore_from_latest=FLAGS.restore_from_latest,
             metric_eval_fn=metric_eval_fn,
             metric_eval_interval_steps=metric_eval_interval_steps,
             valid_interval_epochs=FLAGS.valid_interval_epochs,
             inference_fn=inference_fn,
             inference_interval_epochs=FLAGS.inference_interval_epochs,
             no_log=FLAGS.no_log,
             summary_excls=summary_excls,
             init_fn=init_fn,
             restore_fn=restore_fn,
             restore_include=restore_include,
             restore_exclude=restore_exclude,
             save_all_scope=save_all_scope,
             variables_to_restore=variables_to_restore,
             variables_to_save=variables_to_save,
             output_collection_names=output_collection_names,
             output_node_names=output_node_names,
             write_during_train=FLAGS.write_during_train,
             sess=sess)
