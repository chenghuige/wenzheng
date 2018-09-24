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

# TODO not support multiple gpu right now!

def evaluate(model, dataset, eval_fn, model_path=None, 
             names=None, write_fn=None, write_streaming=False,
             num_steps_per_epoch=None, 
             suffix='.valid', sep=','):
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
      predicts = model(x)
      predicts_list.append(predicts)
      labels_list.append(y)
      ids = gezi.decode(x['id'].numpy())
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
    predicts = model(x).numpy()
    # here id is str in py3 will be bytes
    ids = gezi.decode(x['id'].numpy())

    if not write_streaming:
      predicts_list.append(predicts)
      ids_list.append(ids)
    else:
      for id, predict in zip(ids, predicts):
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

  logging.info('inputs', inputs)

  train_dataset_ = Dataset('train')
  train_dataset = train_dataset_.make_batch(batch_size, inputs)
  num_examples = train_dataset_.num_examples_per_epoch('train') 
  num_all_examples = num_examples

  if FLAGS.fold is not None:
    valid_inputs = [x for x in all_inputs if x not in inputs]
  else:
    valid_inputs = gezi.list_files(FLAGS.valid_input)
  
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
      num_examples = int(num_examples * (len(inputs) / (len(inputs) + 1)))
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None

  if FLAGS.fold is not None:
    if num_examples:
      num_valid_examples = int(num_all_examples * (1 / (len(inputs) + 1)))
      num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_)
    else:
      num_valid_steps_per_epoch = None
  else:
    num_valid_examples = valid_dataset_.num_examples_per_epoch('valid')
    num_valid_steps_per_epoch = -(-num_valid_examples // batch_size_) if num_valid_examples else None

  test_inputs = gezi.list_files(FLAGS.test_input)
  logging.info('test_inputs', test_inputs)
  
  if test_inputs:
    test_dataset_ = Dataset('test')
    test_dataset = test_dataset_.make_batch(batch_size_, test_inputs) 
    num_test_examples = test_dataset_.num_examples_per_epoch('test')
    num_test_steps_per_epoch = -(-num_test_examples // batch_size_) if num_test_examples else None
  else:
    test_dataset = None

  learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  tf.add_to_collection('learning_rate', learning_rate)

  learning_rate_weight = tf.get_collection('learning_rate_weight')[-1]
  try:
    learning_rate_weights = tf.get_collection('learning_rate_weights')[-1]
  except Exception:
    learning_rate_weights = None

  optimizer = melt.get_optimizer(FLAGS.optimizer)(learning_rate)
  
  summary = tf.contrib.summary
  # writer = summary.create_file_writer(FLAGS.model_dir + '/epoch')
  # writer_train = summary.create_file_writer(FLAGS.model_dir + '/train')
  # writer_valid = summary.create_file_writer(FLAGS.model_dir + '/valid')
  writer = summary.create_file_writer(FLAGS.model_dir)
  writer_train = summary.create_file_writer(FLAGS.model_dir)
  writer_valid = summary.create_file_writer(FLAGS.model_dir)
  global_step = tf.train.get_or_create_global_step()

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
        
  ckpt_dir = FLAGS.model_dir + '/ckpt'

  #TODO FIXME now I just changed tf code so to not by default save only latest 5
  # refer to https://github.com/tensorflow/tensorflow/issues/22036
    # manager = tf.contrib.checkpoint.CheckpointManager(
  #     checkpoint, directory=ckpt_dir, max_to_keep=5)
  # latest_checkpoint = manager.latest_checkpoint

  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  if os.path.exists(FLAGS.model_dir + '.index'):
    latest_checkpoint = FLAGS.model_dir

  checkpoint.restore(latest_checkpoint)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

  logging.info('Latest checkpoint:', latest_checkpoint)

  #model.load_weights(os.path.join(ckpt_dir, 'ckpt-1'))
  #model.save('./weight3.hd5')

  start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint else 0
  # TODO currently not support 0.1 epoch.. like this
  num_epochs = FLAGS.num_epochs
  
  if valid_dataset and not FLAGS.mode == 'test' and not 'QUICK' in os.environ and not 'SHOW' in os.environ:
    logging.info('valid')
    if evaluate_fn is not None:
      vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
    elif eval_fn:
      model_path = None if not write_valid else latest_checkpoint
      names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

      print('model_path:', model_path)
      vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                             names, valid_write_fn, write_streaming,
                             num_valid_steps_per_epoch,
                             suffix=valid_suffix, sep=sep)
  
    logging.info2('epoch:%d/%d' % (start_epoch, num_epochs), 
                  ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])
  
  if FLAGS.mode == 'valid':
    exit(0)

  if 'test' in FLAGS.mode:
    logging.info('test/inference')
    if test_dataset:
      if inference_fn is None:
        inference(model, test_dataset, latest_checkpoint, 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, suffix=infer_suffix)
    else:
        inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)
    exit(0)
  
  if 'SHOW' in os.environ:
    num_epochs = start_epoch + 1

  for epoch in range(start_epoch, num_epochs):
    melt.set_global('epoch', '%.4f' % (epoch))
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_valid_loss_avg = tfe.metrics.Mean()

    for i, (x, y) in tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True):
      #if global_step.numpy() == 0:
      #loss = loss_fn(model, x, y, training=True)
      loss, grads = melt.eager.grad(model, x, y, loss_fn)
      optimizer.apply_gradients(zip(grads, model.variables))
      epoch_loss_avg(loss)  # add current batch loss

      batch_size_ = list(x.values())[0].shape[0] if type(x) == type({}) else x[0].shape[0]

      if global_step.numpy() % FLAGS.interval_steps == 0:
        #checkpoint.save(checkpoint_prefix)
        if valid_dataset2:
          x, y = next(iter(valid_dataset2))
          valid_loss = loss_fn(model, x, y)
          epoch_valid_loss_avg(valid_loss)

          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'batch_size:%d' % batch_size_,
                      'learning_rate:%.5f' % learning_rate.numpy(),
                      'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                      'valid_loss:%.4f' % epoch_valid_loss_avg.result().numpy())
          with writer_valid.as_default(), summary.always_record_summaries():
            #summary.scalar('step/loss', epoch_valid_loss_avg.result().numpy())
            summary.scalar('loss/eval', epoch_valid_loss_avg.result().numpy())
            writer_valid.flush()
        else:
          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'batch_size:%d' % batch_size_,
                      'learning_rate:%.3f' % learning_rate.numpy(),
                      'train_loss:%.4f' % epoch_loss_avg.result().numpy())       

        with writer_train.as_default(), summary.always_record_summaries():
          #summary.scalar('step/loss', epoch_loss_avg.result().numpy())
          summary.scalar('loss/train_avg', epoch_loss_avg.result().numpy())
          writer_train.flush()
      
      if valid_dataset and FLAGS.metric_eval_interval_steps and global_step.numpy() and global_step.numpy() % FLAGS.metric_eval_interval_steps == 0:
        if evaluate_fn is not None:
          vals, names = evaluate_fn(model, valid_dataset, None, num_valid_steps_per_epoch)
        elif eval_fn:
          names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None
          vals, names = evaluate(model, valid_dataset, eval_fn, None, 
                                names, valid_write_fn, write_streaming,
                                num_valid_steps_per_epoch, sep=sep)
          with writer_valid.as_default(), summary.always_record_summaries():
            for name, val in zip(names, vals):
              summary.scalar(f'step/valid/{name}', val)
            writer_valid.flush()
      
        logging.info2('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs),  
                      'step:%d' % global_step.numpy(),
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
          logging.info(model.summary())
        except Exception:
          traceback.print_exc()
          logging.info('Fail to do model.summary() may be you have layer define in init but not used in call')
        if 'SHOW' in os.environ:
          exit(0)

    logging.info('epoch:%d/%d' % (epoch + 1, num_epochs), 
                 'step:%d' % global_step.numpy(), 
                 'batch_size:%d' % batch_size,
                 'learning_rate:%.3f' % learning_rate.numpy(),
                 'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                 'valid_loss::%.4f' % epoch_valid_loss_avg.result().numpy())

    timer = gezi.Timer(f'save model to {checkpoint_prefix}', False)
    checkpoint.save(checkpoint_prefix)
    timer.print_elapsed()
    

    if valid_dataset and (epoch + 1) % FLAGS.valid_interval_epochs == 0:
      if evaluate_fn is not None:
        vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
      elif eval_fn:
        model_path = None if not write_valid else tf.train.latest_checkpoint(ckpt_dir)
        names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

        vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                               names, valid_write_fn, write_streaming,
                               num_valid_steps_per_epoch, sep=sep)
    
      logging.info2('epoch:%d/%d' % (epoch + 1, num_epochs), 
                    ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])

    with writer.as_default(), summary.always_record_summaries():
      temp = global_step.value()
      global_step.assign(epoch + 1)
      summary.scalar('epoch/train/loss', epoch_loss_avg.result())
      for name, val in zip(names, vals):
        summary.scalar(f'epoch/valid/{name}', val)
      writer.flush()
      global_step.assign(temp)

    if test_dataset and (epoch + 1) % FLAGS.inference_interval_epochs == 0:
      if inference_fn is None:
        inference(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, suffix=infer_suffix, sep=sep)
      else:
         inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)
