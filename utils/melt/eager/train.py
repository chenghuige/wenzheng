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

import gezi
import melt
logging = melt.logging


def torch_(x):
  for dim in x.shape:
    if dim == 0:
      return x

  x = x.numpy()
  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
      x = x.cuda()

  return x


def to_torch(x, y=None):
  if y is not None:
    y = torch_(y)

  for key in x:
    x[key] = torch_(x[key])
  if y is None:
    return x
  else:
    return x, y

# TODO not support multiple gpu right now!

def evaluate(model, dataset, eval_fn, model_path=None, 
             names=None, write_fn=None, write_streaming=False,
             num_steps_per_epoch=None, 
             suffix='.valid', sep=','):
    if FLAGS.torch:
      model.eval()
    if not write_fn:
      write_streaming = True
    predicts_list = []
    labels_list = []
    ids_list = []
    ofile = model_path + suffix if model_path else None
    if write_streaming:
      out = open(ofile, 'w') if ofile else None
      if out:
        if names is not None:
          print(*names, sep=sep, file=out)
    else:
      out = None

    for x, y in tqdm(dataset, total=num_steps_per_epoch, ascii=True):
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

    # if FLAGS.torch:
    #   predicts_list = [x.detach().numpy() for x in predicts_list]
    #   labels_lis = [x.detach().numpy() for x in labels_list]

    predicts = np.concatenate(predicts_list)
    labels = np.concatenate(labels_list)
    ids = np.concatenate(ids_list)
    
    if out:
      out.close()
    
    if not write_streaming and ofile:
      write_fn(ids, labels, predicts, ofile)
      
    if len(inspect.getargspec(eval_fn).args) == 4:
      return eval_fn(labels, predicts, ids=ids, model_path=model_path)
    elif len(inspect.getargspec(eval_fn).args) == 3:
      if 'ids' in inspect.getargspec(eval_fn).args:
        return eval_fn(labels, predicts, ids)
    else:
      return eval_fn(labels, predicts)

def inference(model, dataset, model_path, 
              names=None, debug_names=None, 
              write_fn=None, write_streaming=False,
              num_steps_per_epoch=None, 
              suffix='.infer', sep=','):
  if FLAGS.torch:
    model.eval()
  if not write_fn:
    write_streaming = True
  ofile = model_path + suffix
  ofile2 = ofile + '.debug'
  if write_streaming:
    if write_fn and len(inspect.getargspec(write_fn).args) == 4:
      out_debug = open(ofile2, 'w')
    else:
      out_debug = None
    out = open(ofile, 'w') 
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
  for (x, _) in tqdm(dataset, total=num_steps_per_epoch, ascii=True):
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
    # if FLAGS.torch:
    #   predicts_list = [x.detach().numpy() for x in predicts_list]
    predicts = np.concatenate(predicts_list)
    ids = np.concatenate(ids_list)

    if len(inspect.getargspec(write_fn).args) == 4:
      write_fn(ids, predicts, ofile, ofile2)
    else:
      write_fn(ids, predicts, ofile)
 
def train(Dataset, 
          model, 
          loss_fn, 
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
          sep=','):
  if FLAGS.torch:
    #torch.cuda.set_device(0)  # set the device back to 0
    if torch.cuda.is_available():
      model.cuda()
  
  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  batch_size = FLAGS.batch_size

  num_gpus = melt.num_gpus()
  if num_gpus > 1:
    assert False, 'Eager mode train currently not support for num gpus > 1'

  #batch_size_ = batch_size if not FLAGS.batch_sizes else int(FLAGS.batch_sizes.split(',')[-1])
  batch_size_ = batch_size

  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]
    # if FLAGS.valid_input:
    #   inputs += [x for x in gezi.list_files(FLAGS.valid_input) if not x.endswith('%d.record' % FLAGS.fold)]
  logging.info('inputs', len(inputs), inputs[:100])
  num_folds = FLAGS.num_folds or len(inputs) + 1


  train_dataset_ = Dataset('train')
  train_dataset = train_dataset_.make_batch(batch_size, inputs)
  num_examples = train_dataset_.num_examples_per_epoch('train') 
  num_all_examples = num_examples

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

  if valid_inputs:
    valid_dataset_ = Dataset('valid')
    valid_dataset = valid_dataset_.make_batch(batch_size_, valid_inputs)
    valid_dataset2 = valid_dataset_.make_batch(batch_size_, valid_inputs, repeat=True)
  else:
    valid_datsset = None
    valid_dataset2 = None

  if num_examples:
    if FLAGS.fold is not None:
      num_examples = int(num_examples * (num_folds - 1) / num_folds)
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None
  logging.info('num_train_examples:', num_examples)

  num_valid_examples = None
  if FLAGS.valid_input:
    num_valid_examples = valid_dataset_.num_examples_per_epoch('valid')
    num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_) if num_valid_examples else None   
  else:
    if FLAGS.fold is not None:
      if num_examples:
        num_valid_examples = int(num_all_examples * (1 / num_folds))
        num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_)
      else:
        num_valid_steps_per_epoch = None
  logging.info('num_valid_examples:', num_valid_examples)

  if FLAGS.test_input:
    test_inputs = gezi.list_files(FLAGS.test_input)
    #test_inputs = [x for x in test_inputs if not 'aug' in x]
    logging.info('test_inputs', test_inputs)
  else:
    test_inputs = None
  
  num_test_examples = None
  if test_inputs:
    test_dataset_ = Dataset('test')
    test_dataset = test_dataset_.make_batch(batch_size_, test_inputs) 
    num_test_examples = test_dataset_.num_examples_per_epoch('test')
    num_test_steps_per_epoch = -(-num_test_examples // batch_size_) if num_test_examples else None
  else:
    test_dataset = None
  logging.info('num_test_examples:', num_test_examples)
  
  summary = tf.contrib.summary
  # writer = summary.create_file_writer(FLAGS.log_dir + '/epoch')
  # writer_train = summary.create_file_writer(FLAGS.log_dir + '/train')
  # writer_valid = summary.create_file_writer(FLAGS.log_dir + '/valid')
  writer = summary.create_file_writer(FLAGS.log_dir)
  writer_train = summary.create_file_writer(FLAGS.log_dir)
  writer_valid = summary.create_file_writer(FLAGS.log_dir)
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  
  tf.add_to_collection('learning_rate', learning_rate)

  learning_rate_weight = tf.get_collection('learning_rate_weight')[-1]
  try:
    learning_rate_weights = tf.get_collection('learning_rate_weights')[-1]
  except Exception:
    learning_rate_weights = None

  ckpt_dir = FLAGS.model_dir + '/ckpt'

  #TODO FIXME now I just changed tf code so to not by default save only latest 5
  # refer to https://github.com/tensorflow/tensorflow/issues/22036
    # manager = tf.contrib.checkpoint.CheckpointManager(
  #     checkpoint, directory=ckpt_dir, max_to_keep=5)
  # latest_checkpoint = manager.latest_checkpoint

  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  logging.info('Latest checkpoint:', latest_checkpoint)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

  if not FLAGS.torch:
    optimizer = melt.get_optimizer(FLAGS.optimizer)(learning_rate)
    
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
      
    if os.path.exists(FLAGS.model_dir + '.index'):
      latest_checkpoint = FLAGS.model_dir   

    checkpoint.restore(latest_checkpoint)

    start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint else 0
  else:
    # TODO torch with learning rate adjust
    optimizer = torch.optim.Adamax(model.parameters(), lr=FLAGS.learning_rate)

    if latest_checkpoint:
      checkpoint = torch.load(latest_checkpoint + '.pyt')
      start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      model.eval()
    else:
      start_epoch = 0

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
    checkpoint.restore(latest_checkpoint)

  #model.load_weights(os.path.join(ckpt_dir, 'ckpt-1'))
  #model.save('./weight3.hd5')

  learning_rate.assign(learning_rate * FLAGS.learning_rate_start_factor)

  # TODO currently not support 0.1 epoch.. like this
  num_epochs = FLAGS.num_epochs if FLAGS.num_epochs != 0 else 1024

  will_valid = valid_dataset and not FLAGS.mode == 'test' and not 'SHOW' in os.environ
  if start_epoch == 0 and not 'EVFIRST' in os.environ and will_valid:
    will_valid = False

  if start_epoch > 0 and not 'QUICK' in os.environ and will_valid:
    will_valid = True 
  
  if will_valid:
    logging.info('----------valid')
    if FLAGS.torch:
      model.eval()
    if evaluate_fn is not None:
      vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
    elif eval_fn:
      model_path = None if not write_valid else latest_checkpoint
      names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

      logging.info('model_path:', model_path, 'model_dir:', FLAGS.model_dir)
      vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                             names, valid_write_fn, write_streaming,
                             num_valid_steps_per_epoch,
                             suffix=valid_suffix, sep=sep)
    logging.info2('epoch:%d/%d' % (start_epoch, num_epochs), 
                  ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])
  
  if FLAGS.mode == 'valid':
    exit(0)

  if 'test' in FLAGS.mode:
    logging.info('--------test/inference')
    if test_dataset:
      if FLAGS.torch:
        model.eval()
      if inference_fn is None:
        inference(model, test_dataset, latest_checkpoint, 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, suffix=infer_suffix)
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
  timer = gezi.Timer()
  num_insts = 0

  for epoch in range(start_epoch, num_epochs):
    melt.set_global('epoch', '%.4f' % (epoch))

    if FLAGS.torch:
      model.train()

    epoch_loss_avg = Mean()
    epoch_valid_loss_avg = Mean()

    #for i, (x, y) in tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True):
    for i, (x, y) in enumerate(train_dataset):
      # print(x, y)
      # continue

      if FLAGS.torch:
        x, y = to_torch(x, y)

      if not FLAGS.torch:
        loss, grads = melt.eager.grad(model, x, y, loss_fn)
        grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
        optimizer.apply_gradients(zip(grads, model.variables))
      else:
        optimizer.zero_grad()
        loss = loss_fn(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       FLAGS.clip_gradients)
        optimizer.step()
      
      epoch_loss_avg(loss)  # add current batch loss

      if FLAGS.torch:
        del loss

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
          try:
            x, y = next(iter(valid_dataset2))
          except Exception:
            # TODO FIXME how.. iterate stop restart.., here hack for my iterator see projects/lm/dataset 
            x, y = next(iter(valid_dataset2))

          if FLAGS.torch:
            x, y = to_torch(x, y)
            model.eval()
          valid_loss = loss_fn(model, x, y)
          epoch_valid_loss_avg(valid_loss)
          if FLAGS.torch:
            model.train()

          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'elapsed:[%.3f]' % elapsed,
                      'batch_size:[%d]' % batch_size_,
                      'batches/s:[%.2f]' % steps_per_second,
                      'insts/s:[%d]' % instances_per_second,
                      '%s' % epoch_time_info,
                      'lr:[%.7f]' % learning_rate.numpy(),
                      'train_loss:[%.4f]' % epoch_loss_avg.result().numpy(),
                      'valid_loss:[%.4f]' % epoch_valid_loss_avg.result().numpy())
          if global_step.numpy() % FLAGS.eval_interval_steps == 0:
            with writer_valid.as_default(), summary.always_record_summaries():
              #summary.scalar('step/loss', epoch_valid_loss_avg.result().numpy())
              summary.scalar('loss/eval', epoch_valid_loss_avg.result().numpy())
              writer_valid.flush()
        else:
          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'elapsed:[%.3f]' % elapsed,
                      'batch_size:[%d]' % batch_size_,
                      'batches/s:[%.2f]' % steps_per_second,
                      'insts/s:[%d]' % instances_per_second,
                      '%s' % epoch_time_info,
                      'lr:[%.7f]' % learning_rate.numpy(),
                      'train_loss:[%.4f]' % epoch_loss_avg.result().numpy())      

        if global_step.numpy() % FLAGS.eval_interval_steps == 0:
          with writer_train.as_default(), summary.always_record_summaries():
            #summary.scalar('step/loss', epoch_loss_avg.result().numpy())
            summary.scalar('loss/train_avg', epoch_loss_avg.result().numpy())
            summary.scalar('learning_rate', learning_rate.numpy())
            summary.scalar('batch_size', batch_size_)
            summary.scalar('epoch', melt.epoch())
            summary.scalar('steps_per_second', steps_per_second)
            summary.scalar('instances_per_second', instances_per_second)
            writer_train.flush()

          if FLAGS.log_dir != FLAGS.model_dir:
            assert FLAGS.log_dir
            command = 'rsync -l -r -t %s/* %s' % (FLAGS.log_dir, FLAGS.model_dir) 
            print(command, file=sys.stderr)
            os.system(command)
      
      if valid_dataset and FLAGS.metric_eval_interval_steps and global_step.numpy() and global_step.numpy() % FLAGS.metric_eval_interval_steps == 0:
        if FLAGS.torch:
          model.eval()
        vals, names = None, None
        if evaluate_fn is not None:
          vals, names = evaluate_fn(model, valid_dataset, None, num_valid_steps_per_epoch)
        elif eval_fn:
          names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None
          vals, names = evaluate(model, valid_dataset, eval_fn, None, 
                                 names, valid_write_fn, write_streaming,
                                 num_valid_steps_per_epoch, sep=sep)
        if vals and names:
          with writer_valid.as_default(), summary.always_record_summaries():
            for name, val in zip(names, vals):
              summary.scalar(f'step/valid/{name}', val)
            writer_valid.flush()
      
        if FLAGS.torch:
          for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate.numpy()
          model.train()

        if names and vals:
          logging.info2('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs),  
                        'valid_step:%d' % global_step.numpy(),
                        'valid_metrics',
                        ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])
        
      
      # if i == 5:
      #   print(i, '---------------------save')
      #   print(len(model.trainable_variables))
      ## TODO FIXME seems save weighs value not ok... not the same as checkpoint save
      #   model.save_weights(os.path.join(ckpt_dir, 'weights'))
      #   checkpoint.save(checkpoint_prefix)
      #   exit(0)


      global_step.assign_add(1)

      if epoch == start_epoch and i == 0:
        try:
          if not FLAGS.torch:
            logging.info(model.summary())
        except Exception:
          traceback.print_exc()
          logging.info('Fail to do model.summary() may be you have layer define in init but not used in call')
        if 'SHOW' in os.environ:
          exit(0)

    logging.info('epoch:%d/%d' % (epoch + 1, num_epochs), 
                'step:%d' % global_step.numpy(), 
                'batch_size:[%d]' % batch_size,
                'lr:[%.7f]' % learning_rate.numpy(),
                'train_loss:[%.4f]' % epoch_loss_avg.result().numpy(),
                'valid_loss::[%.4f]' % epoch_valid_loss_avg.result().numpy())


    timer = gezi.Timer(f'save model to {checkpoint_prefix}-{checkpoint.save_counter.numpy()}', False)
    checkpoint.save(checkpoint_prefix)
    if FLAGS.torch:
      state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
              }
      if torch.cuda.is_available():
       model.cpu()
      torch.save(state, tf.train.latest_checkpoint(ckpt_dir) + '.pyt')
      if torch.cuda.is_available():
       model.cuda()
    timer.print_elapsed()
    
    if valid_dataset and (epoch + 1) % FLAGS.valid_interval_epochs == 0:
      if FLAGS.torch:
        model.eval()

      vals, names = None, None
      if evaluate_fn is not None:
        vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
      elif eval_fn:
        model_path = None if not write_valid else tf.train.latest_checkpoint(ckpt_dir)
        names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

        vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                               names, valid_write_fn, write_streaming,
                               num_valid_steps_per_epoch, suffix=valid_suffix, sep=sep)

      if vals and names:
        logging.info2('epoch:%d/%d' % (epoch + 1, num_epochs), 
                      'step:%d' % global_step.numpy(),
                      'epoch_valid_metrics',
                      ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])

    with writer.as_default(), summary.always_record_summaries():
      temp = global_step.value()
      global_step.assign(epoch + 1)
      summary.scalar('epoch/train/loss', epoch_loss_avg.result().numpy())
      if valid_dataset:
        if FLAGS.torch:
          model.eval()
        if vals and names:
          for name, val in zip(names, vals):
            summary.scalar(f'epoch/valid/{name}', val)
      writer.flush()
      global_step.assign(temp)

    if test_dataset and (epoch + 1) % FLAGS.inference_interval_epochs == 0:
      if FLAGS.torch:
        model.eval()
      if inference_fn is None:
        inference(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, suffix=infer_suffix, sep=sep)
      else:
         inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)

  if FLAGS.log_dir != FLAGS.model_dir:
    assert FLAGS.log_dir
    command = 'rsync -l -r -t %s/* %s' % (FLAGS.log_dir, FLAGS.model_dir) 
    print(command, file=sys.stderr)
    os.system(command)
