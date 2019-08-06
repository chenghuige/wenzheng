#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-09-03 15:40:04.947138
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import torch 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
  pass

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

tfe = tf.contrib.eager

import sys 
import os
from tqdm import tqdm 
import numpy as np
import inspect
import traceback
import copy
import itertools

import gezi
import melt
logging = melt.logging

try:
  #import horovod.tensorflow as hvd
  #import horovod.torch as hvd
  #hvd.init()
  import mpi4py
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
except Exception:
  pass


def torch_(x):
  if FLAGS.torch_only:
    return x
  for dim in x.shape:
    if dim == 0:
      return x

  x = x.numpy()
  # TODO..
  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
  #if type(x) != np.str_:
    x = torch.from_numpy(x)
    #if torch.cuda.is_available():
      #x = x.cuda()
    x = x.to(device)

  return x

def to_torch(x, y=None):
  if FLAGS.torch_only:
    for key in x:
      if type(x[key][0]) != np.str_:
      #if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
        x[key] = x[key].to(device)
    return x, y.to(device)
  if y is not None:
    y = torch_(y)

  if not isinstance(x, dict):
    x = torch_(x)
  else:
    for key in x:
      x[key] = torch_(x[key])
  if y is None:
    return x
  else:
    return x, y

# TODO not support multiple gpu right now!

def evaluate(model, dataset, eval_fn, model_path=None, 
             names=None, write_fn=None, write_streaming=False,
             num_steps=None, num_examples=None,
             suffix='.valid', sep=','):
  if FLAGS.use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd

  if hasattr(model, 'eval'):
    model.eval()
  if not write_fn:
    write_streaming = True
  predicts_list = []
  labels_list = []
  ids_list = []
  ofile = model_path + suffix if model_path else None
  if write_streaming:
    out = open(ofile, 'w', encoding='utf-8') if ofile else None
    if out:
      if names is not None:
        print(*names, sep=sep, file=out)
  else:
    out = None

  for x, y in tqdm(dataset, total=num_steps, ascii=True):
    if FLAGS.torch:
      x, y = to_torch(x, y)

    predicts = model(x)
    if FLAGS.torch:
      predicts = predicts.detach().cpu()
      y = y.detach().cpu()

    predicts_list.append(predicts)
    labels_list.append(y)
    if not FLAGS.torch:
      ids = gezi.decode(x['id'].numpy())
    else:
      ids = gezi.decode(x['id'])
    ids_list.append(ids)

    if out:
      for id, label, predict in zip(ids, y.numpy(), predicts.numpy()):
        if write_fn is None:
          if not gezi.iterable(label):
            label = [label]
          if not gezi.iterable(predict):
            predict = [predict]
          print(id, *label, *predict, sep=sep, file=out)
        else:
          write_fn(id, label, predict, out)

  if out:
    out.close()

    # if FLAGS.torch:
    #   predicts_list = [x.detach().numpy() for x in predicts_list]
    #   labels_lis = [x.detach().numpy() for x in labels_list]

  if FLAGS.use_horovod and FLAGS.horovod_eval:
    #import horovod.torch as hvd
    #print('----------------------before hvd reduce')
    # TODO check eager mode ok...
    tensor = tf.constant(0) if not FLAGS.torch else torch.zeros(0)
    hvd.allreduce(tensor)
    ## here for horovod mutliple gpu dataset is not repeat mode 
    ids_list = comm.allgather(np.concatenate(ids_list))
    predicts_list = comm.allgather(np.concatenate(predicts_list))
    labels_list = comm.allgather(np.concatenate(labels_list))
    comm.barrier()

    ids2 = np.concatenate(ids_list)
    predicts2 = np.concatenate(predicts_list)
    labels2 = np.concatenate(labels_list)
    #----below is for batch parse which if not repeat mode then final batch will still same size not smaller
    # and not use repeat mode so last batch fill with id '' empty we can remove here
    ids = []
    predicts = []
    labels = []
    for i in range(len(ids2)):
      if not ids2[i] == '':
        ids.append(ids2[i])
        predicts.append(predicts2[i])
        labels.append(labels2[i])
    ids = np.array(ids)
    predicts = np.array(predicts)
    labels = np.array(labels)
  else:
    try:
      # concat list so like [[512,], [512,]...] -> [512 * num_batchs]
      # ore [[512, 3], [512,3] ..] -> [512 * num_batchs, 3]
      ids = np.concatenate(ids_list)[:num_examples]
    except Exception:
      ids = ['0'] * num_examples
    predicts = np.concatenate(predicts_list)[:num_examples]
    labels = np.concatenate(labels_list)[:num_examples]
  
  if not write_streaming and ofile and (not FLAGS.use_horovod or hvd.rank() == 0):
    write_fn(ids, labels, predicts, ofile)
    
  if len(inspect.getargspec(eval_fn).args) == 4:
    vals, names = eval_fn(labels, predicts, ids=ids, model_path=model_path)
  elif len(inspect.getargspec(eval_fn).args) == 3:
    if 'ids' in inspect.getargspec(eval_fn).args:
      vals, names = eval_fn(labels, predicts, ids)
    else:
      vals, names = eval_fn(labels, predicts, model_path)
  else:
    vals, names = eval_fn(labels, predicts)
  
  if model_path and (not FLAGS.use_horovod or hvd.rank() == 0):
    with open(model_path + '.valid.metrics', 'w') as out:
      for val, name in zip(vals, names):
        print(name, val, sep='\t', file=out)

  return vals, names

def inference(model, dataset, model_path, 
              names=None, debug_names=None, 
              write_fn=None, write_streaming=False,
              num_steps=None, num_examples=None,
              suffix='.infer', sep=','):
  if FLAGS.use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd
  if has_attr(model, 'eval'):
    model.eval()
  if not write_fn:
    write_streaming = True
  ofile = model_path + suffix
  ofile2 = ofile + '.debug'
  if write_streaming:
    if write_fn and len(inspect.getargspec(write_fn).args) == 4:
      out_debug = open(ofile2, 'w', encoding='utf-8')
    else:
      out_debug = None
    out = open(ofile, 'w', encoding='utf-8') 
  else:
    out = None
    out_debug = None
  
  if write_streaming:
    if names is not None:
      print(*names, sep=sep, file=out)
    if debug_names and out_debug:
      print(*debug_names, sep=sep, file=out_debug)

  predicts_list = []
  ids_list = []
  for (x, _) in tqdm(dataset, total=num_steps, ascii=True):
    if FLAGS.torch:
      x = to_torch(x)
    predicts = model(x)
    if FLAGS.torch:
      predicts = predicts.detach().cpu()
    # here id is str in py3 will be bytes
    if not FLAGS.torch:
      ids = gezi.decode(x['id'].numpy())
    else:
      ids = gezi.decode(x['id'])

    if not write_streaming:
      predicts_list.append(predicts)
      ids_list.append(ids)
    else:
      for id, predict in zip(ids, predicts.numpy()):
        if write_fn is None:
          if not gezi.iterable(predict):
            predict = [predict]
          print(id, *predict, sep=sep, file=out)
        else:
          if out_debug:
            write_fn(id, predict, out, out_debug)
          else:
            write_fn(id, predict, out)
  
  if out:
    out.close()
  if out_debug:
    out_debug.close()

  if not write_streaming:
    if FLAGS.use_horovod and FLAGS.horovod_eval:
      #import horovod.torch as hvd
      #print('----------------------before hvd reduce')
      tensor = tf.constant(0) if not FLAGS.torch else torch.zeros(0)
      hvd.allreduce(tensor)
      ## here for horovod mutliple gpu dataset is not repeat mode 
      ids_list = comm.allgather(np.concatenate(ids_list))
      predicts_list = comm.allgather(np.concatenate(predicts_list))
      comm.barrier()

      ids2 = np.concatenate(ids_list)
      predicts2 = np.concatenate(predicts_list)
      #----below is for batch parse which if not repeat mode then final batch will still same size not smaller
      # and not use repeat mode so last batch fill with id '' empty we can remove here
      ids = []
      predicts = []
      for i in range(len(ids2)):
        if not ids2[i] == '':
          ids.append(ids2[i])
          predicts.append(predicts2[i])
      ids = np.array(ids)
      predicts = np.array(predicts)
    else:
      try:
        # concat list so like [[512,], [512,]...] -> [512 * num_batchs]
        # ore [[512, 3], [512,3] ..] -> [512 * num_batchs, 3]
        ids = np.concatenate(ids_list)[:num_examples]
      except Exception:
        ids = ['0'] * num_examples
      predicts = np.concatenate(predicts_list)[:num_examples]

    if (not FLAGS.use_horovod or hvd.rank() == 0):
      if len(inspect.getargspec(write_fn).args) == 4:
        write_fn(ids, predicts, ofile, ofile2)
      else:
        write_fn(ids, predicts, ofile)


def load_torch_model(model, path):
  checkpoint = torch.load(path)
  state = checkpoint['state_dict']   
  
  new_state = {}
  model_ = model.module if hasattr(model, 'module') else model

  for key, val in state.items():
    if key in model_.state_dict():
      new_state[key] = val

  logging.info('num updated keys from checkpoint', len(new_state), 'epoch:', checkpoint['epoch'], 'step:', checkpoint['step'])

  # this is for model state has more params then loaded so just partial update mode state with key,vals from loaded     
  new_params = model_.state_dict()
  new_params.update(new_state)
  model_.load_state_dict(new_params)

  model.eval()

  return checkpoint

def horovod_torch_deal_optimizer(optimizer, model):
  import horovod.torch as hvd
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  optimizer = hvd.DistributedOptimizer(optimizer,
                                       named_parameters=model.named_parameters())
  return optimizer

def get_torch_optimizer(optimizer, model, num_steps_per_epoch=None, param_groups=None):
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    import horovod.torch as hvd
  if optimizer is None:
    import lele
    is_dynamic_opt = True
    if FLAGS.optimizer == 'noam':
      optimizer = torch.optim.Adamax(model.parameters(), lr=0)
      if use_horovod:
        optimizer = horovod_torch_deal_optimizer(optimizer, model)
      optimizer = lele.training.optimizers.NoamOpt(128, 2, 4000, optimizer)
    elif FLAGS.optimizer == 'bert':
      num_train_steps = int(num_steps_per_epoch * (FLAGS.num_decay_epochs or FLAGS.num_epochs))
      if FLAGS.warmup_steps and use_horovod:
        FLAGS.warmup_steps = max(int(FLAGS.warmup_steps / hvd.size()), 1)
      num_warmup_steps = FLAGS.warmup_steps or int(num_steps_per_epoch * FLAGS.warmup_epochs) or int(num_train_steps * FLAGS.warmup_proportion) 
      logging.info('num_train_steps', num_train_steps, 'num_warmup_steps', num_warmup_steps, 'warmup_proportion', FLAGS.warmup_proportion)
      optimizer = torch.optim.Adamax(model.parameters(), lr=0)
      if use_horovod:
        optimizer = horovod_torch_deal_optimizer(optimizer, model)
      optimizer = lele.training.optimizers.BertOpt(
                          FLAGS.learning_rate, 
                          FLAGS.min_learning_rate,
                          num_train_steps,
                          num_warmup_steps,
                          optimizer
                          )
  else:
    is_dynamic_opt = False
    optimizer = torch.optim.Adamax(param_groups if param_groups else model.parameters(), lr=FLAGS.learning_rate)
    if use_horovod:
      optimizer = hvd.DistributedOptimizer(optimizer,
                                           named_parameters=model.named_parameters())
  return optimizer, is_dynamic_opt

 
def train(model, 
          loss_fn,
          Dataset=None,  
          dataset=None,
          valid_dataset=None,
          valid_dataset2=None,
          test_dataset=None,
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          write_valid=True,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer',
          write_streaming=False,
          optimizer=None,
          param_groups=None,
          init_fn=None,
          sep=','):
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd

  if Dataset is None:
    assert dataset
  logging.info('Dataset', Dataset, 'dataset', dataset, 'valid_dataset', valid_dataset, 'test_dataset', test_dataset, loss_fn)

  if FLAGS.torch:
    logging.info(model) # keras will show model after first training step
    torch.manual_seed(FLAGS.seed or 0)
    if torch.cuda.device_count():
      torch.cuda.manual_seed(FLAGS.seed or 0)
    if torch.cuda.device_count() > 1 and not use_horovod:
      model = torch.nn.DataParallel(model)
    model.to(device)
    
  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  #batch_size = FLAGS.batch_size
  batch_size = melt.batch_size()

  num_gpus = melt.num_gpus()

  #batch_size = max(batch_size, 1)
  #batch_size_ = batch_size if not FLAGS.batch_sizes else int(FLAGS.batch_sizes.split(',')[-1])
  batch_size_ = FLAGS.eval_batch_size or batch_size

  if dataset is None:
    if FLAGS.fold is not None:
      inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold) and not x.endswith('%d.tfrecord' % FLAGS.fold)]
      # if FLAGS.valid_input:
      #   inputs += [x for x in gezi.list_files(FLAGS.valid_input) if not x.endswith('%d.record' % FLAGS.fold)]
    logging.info('inputs', len(inputs), inputs[:100])
  num_folds = FLAGS.num_folds or len(inputs) + 1

  if dataset is None:
    dataset = Dataset('train')
    assert len(inputs) > 0
    # for eager train, here still repat True right now, but if set None wil by defualt return False for eager similar as pytorch
    train_dataset = dataset.make_batch(batch_size, inputs, repeat=True, simple_parse=FLAGS.simple_parse)
    num_examples = dataset.num_examples_per_epoch('train') 
  else:
    assert FLAGS.torch_only, 'only torch only currently support input dataset not Dataset class type, because we do not have len function there'
    train_dataset = dataset
    num_examples = len(train_dataset.dataset)

  num_all_examples = num_examples

  if valid_dataset is None:
    valid_inputs = None
    if FLAGS.valid_input:
      valid_inputs = gezi.list_files(FLAGS.valid_input)
    else:
      if FLAGS.fold is not None:
        #valid_inputs = [x for x in all_inputs if x not in inputs]
        if not FLAGS.test_aug:
          valid_inputs = [x for x in all_inputs if not 'aug' in x and x not in inputs]
        else:
          valid_inputs = [x for x in all_inputs if 'aug' in x and x not in inputs]

    logging.info('valid_inputs', valid_inputs)

  num_valid_examples = None
  if valid_dataset is not None:
    # mainly for torch now
    num_valid_examples = len(valid_dataset.dataset)
    num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_) if num_valid_examples else None   
    valid_dataset2_iter = itertools.cycle(valid_dataset2)
  else:
    if valid_inputs:
      valid_dataset = dataset.make_batch(batch_size_, valid_inputs, subset='valid',  hvd_shard=FLAGS.horovod_eval )
      valid_dataset2 = dataset.make_batch(batch_size, valid_inputs, subset='valid', repeat=True, initializable=False, hvd_shard=False)
      valid_dataset2_iter = iter(valid_dataset2)
    else:
      valid_dataset = None
      valid_dataset2 = None

  if num_examples:
    if FLAGS.fold is not None:
      num_examples = int(num_examples * (num_folds - 1) / num_folds)
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None
  logging.info('num_train_examples:', num_examples)
  if use_horovod and num_examples:
    num_steps_per_epoch = -(-num_examples // (batch_size * hvd.size()))

  if num_valid_examples is None:
    if FLAGS.valid_input:
      num_valid_examples = dataset.num_examples_per_epoch('valid')
      num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_) if num_valid_examples else None   
    else:
      if FLAGS.fold is not None:
        if num_examples:
          num_valid_examples = int(num_all_examples * (1 / num_folds))
          num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_)
        else:
          num_valid_steps_per_epoch = None
  if use_horovod and FLAGS.horovod_eval and num_valid_examples:
      num_valid_steps_per_epoch = -(-num_valid_examples // (batch_size_ * hvd.size()))
  logging.info('num_valid_examples:', num_valid_examples)

  if test_dataset is None:
    if FLAGS.test_input:
      test_inputs = gezi.list_files(FLAGS.test_input)
      #test_inputs = [x for x in test_inputs if not 'aug' in x]
      logging.info('test_inputs', test_inputs)
    else:
      test_inputs = None
  
  num_test_examples = None
  if test_dataset is not None:
    num_test_examples = len(test_dataset.dataset)
  else:
    if test_inputs:
      test_dataset = dataset.make_batch(batch_size_, test_inputs, subset='test') 
      num_test_examples = dataset.num_examples_per_epoch('test')
    else:
      test_dataset = None
  num_test_steps_per_epoch = -(-num_test_examples // batch_size_) if num_test_examples else None
  if use_horovod and FLAGS.horovod_eval and num_test_examples:
      num_test_steps_per_epoch = -(-num_test_examples // (batch_size_ * hvd.size()))
  logging.info('num_test_examples:', num_test_examples)
  
  summary = tf.contrib.summary
  # writer = summary.create_file_writer(FLAGS.log_dir + '/epoch')
  # writer_train = summary.create_file_writer(FLAGS.log_dir + '/train')
  # writer_valid = summary.create_file_writer(FLAGS.log_dir + '/valid')
  writer = summary.create_file_writer(FLAGS.log_dir)
  writer_train = summary.create_file_writer(FLAGS.log_dir)
  writer_valid = summary.create_file_writer(FLAGS.log_dir)
  
  if not FLAGS.torch:
    global_step = tf.train.get_or_create_global_step()
  else:
    global_step = melt.GlobalStep(0)
  
  ## RuntimeError: tf.summary.FileWriter is not compatible with eager execution. Use tf.contrib.summary instead.
  #logger = gezi.SummaryWriter(FLAGS.log_dir)

  if not FLAGS.torch:
    learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  else:
    learning_rate = melt.LearningRate(FLAGS.learning_rate)
  
  #tf.add_to_collection('learning_rate', learning_rate)
  melt.add_global('learning_rate', learning_rate)

  #learning_rate_weight = tf.get_collection('learning_rate_weight')[-1]
  learning_rate_weight = melt.get_global('learning_rate_weight')
  try:
    #learning_rate_weights = tf.get_collection('learning_rate_weights')[-1]
    learning_rate_weights = melt.get_global('learning_rate_weights')
  except Exception:
    learning_rate_weights = None

  # ckpt dir save models one per epoch
  ckpt_dir = os.path.join(FLAGS.model_dir, 'ckpt')
  os.system('mkdir -p %s' % ckpt_dir)
  # HACK ckpt dir is actually save mini epoch like when you set save_interval_epochs=0.1, this is usefull when you training large dataset
  ckpt_dir2 = os.path.join(FLAGS.model_dir, 'ckpt2')
  os.system('mkdir -p %s' % ckpt_dir2)

  #TODO FIXME now I just changed tf code so to not by default save only latest 5
  # refer to https://github.com/tensorflow/tensorflow/issues/22036
    # manager = tf.contrib.checkpoint.CheckpointManager(
  #     checkpoint, directory=ckpt_dir, max_to_keep=5)
  # latest_checkpoint = manager.latest_checkpoint

  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  if latest_checkpoint:
    logging.info('Latest checkpoint:', latest_checkpoint)
  else:
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir2)
    logging.info('Latest checkpoint:', latest_checkpoint)
  
  if os.path.exists(FLAGS.model_dir + '.index'):
    latest_checkpoint = FLAGS.model_dir  

  if 'test' in FLAGS.work_mode or 'valid' in FLAGS.work_mode:
    #assert not os.path.isdir(FLAGS.model_dir), FLAGS.model_dir
    latest_checkpoint = FLAGS.model_dir
    #assert os.path.exists(latest_checkpoint) and os.path.isfile(latest_checkpoint)

  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
  checkpoint_prefix2 = os.path.join(ckpt_dir2, 'ckpt')

  if not FLAGS.torch:
    try:
      optimizer = optimizer or melt.get_optimizer(FLAGS.optimizer)(learning_rate)
    except Exception:
      logging.warning(f'Fail to using {FLAGS.optimizer} use adam instead')
      optimizer = melt.get_optimizer('adam')(learning_rate)
    
    # TODO...
    if  learning_rate_weights is None:
      checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            model=model,
            optimizer=optimizer,
            global_step=global_step)
    else:
      checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            learning_rate_weights=learning_rate_weights,
            model=model,
            optimizer=optimizer,
            global_step=global_step) 

    checkpoint.restore(latest_checkpoint)
    checkpoint2 = copy.deepcopy(checkpoint)

    start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint and 'ckpt' in latest_checkpoint else 0
    start_step = 0 # TODO
  else:
    # TODO torch with learning rate adjust
      # https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py
  # TODO full support for pytorch now not work
    optimizer, is_dynamic_opt = get_torch_optimizer(optimizer, model, num_steps_per_epoch, param_groups)

    start_epoch = 0  
    latest_path = latest_checkpoint + '.pyt' if latest_checkpoint else os.path.join(FLAGS.model_dir, 'latest.pyt')
    if not os.path.exists(latest_path):
      latest_path = os.path.join(FLAGS.model_dir, 'latest.pyt')
    if os.path.exists(latest_path):
      logging.info('loading torch model from', latest_path)
      checkpoint = load_torch_model(model, latest_path)
      if not FLAGS.torch_finetune:
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        global_step.assign(step + 1)
      if FLAGS.torch_load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if not FLAGS.torch:
      # TODO by this way restart can not change learning rate..
      if learning_rate_weights is None:
        checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, 
            learning_rate_weight=learning_rate_weight,
            global_step=global_step)
      else:
        checkpoint = tf.train.Checkpoint(
              learning_rate=learning_rate, 
              learning_rate_weight=learning_rate_weight,
              learning_rate_weights=learning_rate_weights,
              global_step=global_step)

    try:
      checkpoint.restore(latest_checkpoint)
      checkpoint2 = copy.deepcopy(checkpoint)
    except Exception:
      pass

  if FLAGS.torch and is_dynamic_opt:
    optimizer._step = global_step.numpy()
    
  #model.load_weights(os.path.join(ckpt_dir, 'ckpt-1'))
  #model.save('./weight3.hd5')
  logging.info('optimizer:', optimizer)

  if FLAGS.torch_lr:
    learning_rate.assign(optimizer.rate(1))
  if FLAGS.torch:
    learning_rate.assign(optimizer.param_groups[0]['lr'])
    logging.info('learning rate got from pytorch latest.py as', learning_rate.numpy())

  learning_rate.assign(learning_rate * FLAGS.learning_rate_start_factor)
  if learning_rate_weights is not None:
    learning_rate_weights.assign(learning_rate_weights * FLAGS.learning_rate_start_factor)

  # TODO currently not support 0.1 epoch.. like this
  num_epochs = FLAGS.num_epochs if FLAGS.num_epochs != 0 else 1024

  will_valid = valid_dataset and not FLAGS.work_mode == 'test' and not 'SHOW' in os.environ and not 'QUICK' in os.environ
  if global_step.numpy() == 0 :
    will_valid = False

  if gezi.get_env('EVFIRST') == '1':
    will_valid = True
  
  if gezi.get_env('EVFIRST') == '0':
    will_valid = False

  if will_valid:
    logging.info('----------valid')
    if hasattr(model, 'eval'):
      model.eval()
    names = None 
    if evaluate_fn is not None:
      vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
    elif eval_fn:
      model_path = None if not write_valid else latest_checkpoint
      names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

      logging.info('model_path:', model_path, 'model_dir:', FLAGS.model_dir)
      vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                             names, valid_write_fn, write_streaming,
                             num_valid_steps_per_epoch, num_valid_examples,
                             suffix=valid_suffix, sep=sep)
    if names:
      logging.info2('epoch:%.2f/%d step:%d' % (global_step.numpy() / num_steps_per_epoch, num_epochs, global_step.numpy()), 
                    ['%s:%.4f' % (name, val) for name, val in zip(names, vals)])
  
    if FLAGS.work_mode == 'valid' or gezi.get_env('METRIC') == '1':
      exit(0)

  if 'test' in FLAGS.work_mode or gezi.get_env('TEST') == '1' or gezi.get_env('INFER') == '1':
    logging.info('--------test/inference')
    if test_dataset:
      if hasattr(model, eval):
        model.eval()
      if inference_fn is None:
        # model_path = FLAGS.model_dir + '.pyt' if not latest_checkpoint else latest_checkpoint
        # logging.info('model_path', model_path)
        assert latest_checkpoint
        inference(model, test_dataset, latest_checkpoint, 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, num_test_examples, suffix=infer_suffix)
      else:
        inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)
    exit(0)
  
  if 'SHOW' in os.environ:
    num_epochs = start_epoch + 1

  class PytObj(object):
    def __init__(self, x):
      self.x = x
    def numpy(self):
      return self.x

  class PytMean(object):
    def __init__(self):
      self._val = 0. 
      self.count = 0

      self.is_call = True

    def clear(self):
      self._val = 0
      self.count = 0

    def __call__(self, val):
      if not self.is_call:
        self.clear()
        self.is_call = True
      self._val += val.item()
      self.count += 1

    def result(self):
      if self.is_call:
        self.is_call = False
      if not self.count:
        val = 0
      else:
        val = self._val / self.count
      # TODO just for compact with tf ..
      return PytObj(val)
      
  Mean =  tfe.metrics.Mean if not FLAGS.torch else PytMean
  
  num_insts = 0

  if FLAGS.learning_rate_decay_factor > 0:
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

  #-------------------------start training
  if hasattr(model, 'train'):
    model.train()

  if use_horovod:
    if FLAGS.torch:
      hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # if using tf eager reading need to shink size 
    if not FLAGS.torch_only:
      # if use torch dataset, can get len directly for distributed train
      num_steps_per_epoch = -(-num_examples // (batch_size * hvd.size())) if num_examples else None  
      if FLAGS.horovod_eval:
        # multiple gpu eval
        num_valid_steps_per_epoch = -(-num_valid_examples // (batch_size_ * hvd.size())) if num_valid_examples else None
        num_test_steps_per_epoch = -(-num_test_examples // (batch_size_ * hvd.size())) if num_test_examples else None

  timer = gezi.Timer()
  loss_avg = Mean()

  # for eager right now is repeat mode, so not need for epoch, but we can stop at inter loop last check global step and exit
  if not num_epochs:
    num_epochs = 1024 # not stop if not manu set num epochs

  #----------------------------------------main loop here
  for epoch in range(start_epoch, start_epoch + num_epochs):
    # FLAGS.torch only will not use eager, FLAGS.torch still use eager tf reading
    if FLAGS.torch_only:
      if train_dataset.sampler and hasattr(train_dataset.sampler, 'set_epoch'):
        # if not set each epoch shuffle same seed..
        train_dataset.sampler.set_epoch(epoch)

    for i, (x, y) in enumerate(train_dataset):
      if FLAGS.torch:
        x, y = to_torch(x, y)
        if is_dynamic_opt:
          learning_rate.assign(optimizer.rate())

      def loss_fn_(x, y):
        if not FLAGS.torch and 'training' in inspect.getargspec(model.call).args:
          y_ = model(x, training=True)
        else:
          y_ = model(x)
        if not FLAGS.torch:
          return loss_fn(y, y_)
        else:
          return loss_fn(y_, y)
      
      if not FLAGS.torch:
        loss, grads = melt.eager.grad(model, x, y, loss_fn)
        grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
        #optimizer.apply_gradients(zip(grads, model.variables))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist_eager.py
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        # TODO check eager mode
        if use_horovod and epoch == start_epoch and i == 0:
          hvd.broadcast_variables(model.variables, root_rank=0)
          hvd.broadcast_variables(optimizier.variables(), root_rank=0)
      else:
        optimizer.zero_grad()
        loss = loss_fn_(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        FLAGS.clip_gradients)
        optimizer.step()

      global_step.assign_add(1)
      loss_avg(loss)
    
      ## https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
      # if FLAGS.torch:
      #   del loss

      batch_size_ = list(x.values())[0].shape[FLAGS.batch_size_dim] if type(x) == type({}) else x.shape[FLAGS.batch_size_dim]
      num_insts += int(batch_size_)
      if global_step.numpy() % FLAGS.interval_steps == 0:
        #checkpoint.save(checkpoint_prefix)
        elapsed = timer.elapsed()
        steps_per_second = FLAGS.interval_steps / elapsed
        instances_per_second = num_insts / elapsed
        num_insts = 0

        if num_steps_per_epoch is None:
          epoch_time_info = ''
        else:
          hours_per_epoch = num_steps_per_epoch / FLAGS.interval_steps * elapsed / 3600
          epoch_time_info = '1epoch:[{:.2f}h]'.format(hours_per_epoch)

        if valid_dataset2:
          # try:
          #   x, y = next(iter(valid_dataset2))
          # except Exception:
          #   # TODO FIXME how.. iterate stop restart.., here hack for my iterator see projects/lm/dataset 
          #   x, y = next(iter(valid_dataset2))
          ## valid dataset2 is repeated
          ## NOTICE will always the first batch ... as below
          #x, y = next(iter(valid_dataset2))
          x, y = next(valid_dataset2_iter)
          #print(x['id'][0])
          if FLAGS.torch:
            x, y = to_torch(x, y)
          if hasattr(model, 'eval'):  
            model.eval()
          valid_loss = loss_fn_(x, y)
          valid_loss = valid_loss.numpy() if not FLAGS.torch else valid_loss.item()
          if hasattr(model, 'train'):
            model.train()

          if not use_horovod or hvd.rank() == 0:
                        # 'train_loss:[%.4f]' % loss_avg.result().numpy(),
                        # 'valid_loss:[%.4f]' % valid_loss_avg.result().numpy()
            logging.info2('epoch:%.2f/%d' % ((global_step.numpy() / num_steps_per_epoch), num_epochs), 
                        'step:%d' % global_step.numpy(), 
                        'elapsed:[%.2f]' % elapsed,
                        'batch_size:[%d]' % batch_size_,
                        'gpus:[%d]' % num_gpus, 
                        'batches/s:[%.2f]' % steps_per_second,
                        'insts/s:[%d]' % instances_per_second,
                        '%s' % epoch_time_info,
                        'lr:[%.6f]' % learning_rate.numpy(),
                        'train_loss:[%.4f]' % loss_avg.result().numpy(),
                        'valid_loss:[%.4f]' % valid_loss
                        )
            if global_step.numpy() % FLAGS.valid_interval_steps == 0:
              with writer_valid.as_default(), summary.always_record_summaries():
                summary.scalar('loss/valid', valid_loss)
                writer_valid.flush()
        else:
          if not use_horovod or hvd.rank() == 0:
            #'train_loss:[%.4f]' % loss_avg.result().numpy()
            logging.info2('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                        'step:%d' % global_step.numpy(), 
                        'elapsed:[%.2f]' % elapsed,
                        'batch_size:[%d]' % batch_size_,
                        'gpus:[%d]' % num_gpus, 
                        'batches/s:[%.2f]' % steps_per_second,
                        'insts/s:[%d]' % instances_per_second,
                        '%s' % epoch_time_info,
                        'lr:[%.6f]' % learning_rate.numpy(),
                        'train_loss:[%.4f]' % loss_avg.result().numpy()
                        )      

        if not use_horovod or hvd.rank() == 0:
          if global_step.numpy() % FLAGS.valid_interval_steps == 0:
            with writer_train.as_default(), summary.always_record_summaries():
              summary.scalar('loss/train_avg', loss_avg.result().numpy())
              summary.scalar('learning_rate', learning_rate.numpy())
              summary.scalar('other/batch_size', batch_size_)
              summary.scalar('other/epoch', melt.epoch())
              summary.scalar('perf/steps_per_second', steps_per_second)
              summary.scalar('perf/instances_per_second', instances_per_second)
              writer_train.flush()

      if valid_dataset and FLAGS.metric_eval_interval_steps and global_step.numpy() and global_step.numpy() % FLAGS.metric_eval_interval_steps == 0:
        if hasattr(model, eval):
          model.eval()
        vals, names = None, None
        if evaluate_fn is not None:
          vals, names = evaluate_fn(model, valid_dataset, None, num_valid_steps_per_epoch)
        elif eval_fn:
          names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None
          vals, names = evaluate(model, valid_dataset, eval_fn, None, 
                                  names, valid_write_fn, write_streaming,
                                  num_valid_steps_per_epoch, num_valid_examples, sep=sep)
        if not use_horovod or hvd.rank() == 0:
          if vals and names:
            with writer_valid.as_default(), summary.always_record_summaries():
              for name, val in zip(names, vals):
                summary.scalar(f'step_eval/{name}', val)
              writer_valid.flush()
      
        if FLAGS.torch:
          if not FLAGS.torch_lr:
            # control learning rate by tensorflow learning rate
            for param_group in optimizer.param_groups:
              # important learning rate decay
              param_group['lr'] = learning_rate.numpy()
        if hasattr(model, 'train'):  
          model.train()
        if not use_horovod or hvd.rank() == 0:
          if names and vals:
            logging.info2('epoch:%.2f/%d' % ((global_step.numpy() / num_steps_per_epoch), num_epochs),  
                          'valid_step:%d' % global_step.numpy(),
                          'valid_metrics',
                          ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])
      
      if not use_horovod or hvd.rank() == 0:
      # TODO save ok ?
        if global_step.numpy() % FLAGS.save_interval_steps == 0:
          if FLAGS.torch:
            state = {
                    'epoch': int(global_step.numpy() / num_steps_per_epoch),
                    'step': global_step.numpy(),
                    'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                  }
            torch.save(state, os.path.join(FLAGS.model_dir, 'latest.pyt'))     

        # TODO fixme why if both checpoint2 and chekpoint used... not ok..
        if FLAGS.save_interval_epochs and global_step.numpy() % int(num_steps_per_epoch * FLAGS.save_interval_epochs) == 0:
          checkpoint2.save(checkpoint_prefix2) 
          if FLAGS.torch:
            state = {
                    'epoch': int(global_step.numpy() / num_steps_per_epoch),
                    'step': global_step.numpy(),
                    'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                  }
            torch.save(state, tf.train.latest_checkpoint(ckpt_dir2) + '.pyt')

      if FLAGS.learning_rate_decay_factor > 0:
        if global_step.numpy() >= decay_start_step and global_step.numpy() % decay_steps == 0:
          lr = max(learning_rate.numpy() * FLAGS.learning_rate_decay_factor, FLAGS.min_learning_rate)
          if lr < learning_rate.numpy():
            learning_rate.assign(lr)
            if FLAGS.torch:
              for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate.numpy()

      if i == 0 and epoch == start_epoch:
        try:
          if not FLAGS.torch:
            logging.info(model.summary())
            # #tf.keras.utils.plot_model(model, to_file='/home/gezi/model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
            # import keras
            # keras.utils.plot_model(model, to_file='/home/gezi/model.png', show_shapes=False, show_layer_names=True, rankdir='LR', expand_nested=True, dpi=96)
        except Exception:
          traceback.print_exc()
          logging.info('Fail to do model.summary() may be you have layer define in init but not used in call')
        if 'SHOW' in os.environ:
          exit(0)
      
      if valid_dataset and  global_step.numpy() % int(num_steps_per_epoch * FLAGS.valid_interval_epochs) == 0:
        if hasattr(model, 'eval'):
          model.eval()

        vals, names = None, None
        if evaluate_fn is not None:
          vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
        elif eval_fn:
          model_path = None if not write_valid else tf.train.latest_checkpoint(ckpt_dir)
          print('---------metric evaluate step', global_step.numpy(), 'model_path:', model_path)
          names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

          vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                                names, valid_write_fn, write_streaming,
                                num_valid_steps_per_epoch, num_valid_examples, suffix=valid_suffix, sep=sep)

        if not use_horovod or hvd.rank() == 0:
          if vals and names:
            logging.info2('epoch:%.2f/%d' % (global_step.numpy() / num_steps_per_epoch, num_epochs), 
                          'step:%d' % global_step.numpy(),
                          'valid_metrics',
                          ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])

        if not use_horovod or hvd.rank() == 0:
          with writer.as_default(), summary.always_record_summaries():
            temp = global_step.value()
            global_step.assign(int(global_step.numpy() / int(num_steps_per_epoch * FLAGS.valid_interval_epochs)))
            if valid_dataset:
              if hasattr(model, 'eval'):
                model.eval()
              if vals and names:
                for name, val in zip(names, vals):
                  summary.scalar(f'eval/{name}', val)
            writer.flush()
            global_step.assign(temp)

      if test_dataset and global_step.numpy() % int(num_steps_per_epoch * FLAGS.inference_interval_epochs) == 0:
        if hasattr(model, 'eval'):
          model.eval()
        if inference_fn is None:
          inference(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), 
                    infer_names, infer_debug_names, infer_write_fn, write_streaming,
                    num_test_steps_per_epoch, num_test_examples, suffix=infer_suffix, sep=sep)
        else:
          inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)

      if num_epochs and (global_step.numpy() % num_steps_per_epoch) == 0 and int(global_step.numpy() / num_steps_per_epoch) >= num_epochs:
        logging.info(f'Finshed training of {num_epochs} epochs')
        exit(0)

