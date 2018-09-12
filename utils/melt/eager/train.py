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

import gezi
import melt
logging = melt.logging

# TODO not support multiple gpu right now!

def evaluate(model, dataset, eval_fn, model_path=None, 
             names=None, write_fn=None,
             num_steps_per_epoch=None, suffix='.valid'):
    predicts_list = []
    labels_list = []
    ids_list = []
    ofile = model_path + suffix if model_path else None
    out = open(ofile, 'w', encoding='UTF-8') if ofile else None
    if out:
      if names is not None:
        print(*names, sep=',', file=out)
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
            print(id, *label, *predict, sep=',', file=out)
          else:
            write_fn(id, label, predict, out)

    predicts = np.concatenate(predicts_list)
    labels = np.concatenate(labels_list)
    ids = np.concatenate(ids_list)
    if out:
      out.close()
      
    if len(inspect.getargspec(eval_fn).args) == 4:
      return eval_fn(labels, predicts, ids=ids, model_path=model_path)
    elif len(inspect.getargspec(eval_fn).args) == 3:
      if 'ids' in inspect.getargspec(eval_fn).args:
        return eval_fn(labels, predicts, ids)
    else:
      return eval_fn(labels, predicts)

def inference(model, dataset, model_path, 
              names=None, write_fn=None,
              num_steps_per_epoch=None, suffix='.infer'):
  ofile = model_path + suffix
  with open(ofile, 'w', encoding='UTF-8') as out:
    if names is not None:
      print(*names, sep=',', file=out)
    for (x, _) in tqdm(dataset, total=num_steps_per_epoch, ascii=True):
      predicts = model(x).numpy()
      # here id is str in py3 will be bytes
      ids = gezi.decode(x['id'].numpy())
      for id, predict in zip(ids, predicts):
        if write_fn is None:
          if not gezi.iterable(predict):
            predict = [predict]
          print(id, *predict, sep=',', file=out)
        else:
          write_fn(id, predict, out)

def train(Dataset, 
          model, 
          loss_fn, 
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          write_valid=False,
          valid_names=None,
          infer_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer'):
  input_ =  FLAGS.train_input 
  inputs = gezi.list_files(input_)
  inputs.sort()

  all_inputs = inputs

  batch_size = FLAGS.batch_size

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
    valid_dataset = valid_dataset_.make_batch(batch_size, valid_inputs)
    valid_dataset2 = valid_dataset_.make_batch(batch_size, valid_inputs, repeat=True)
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
      num_valid_steps_per_epoch = -(-num_valid_examples // batch_size)
    else:
      num_valid_steps_per_epoch = None
  else:
    num_valid_examples = valid_dataset_.num_examples_per_epoch('valid')
    num_valid_steps_per_epoch = -(-num_valid_examples // batch_size) if num_valid_examples else None

  test_inputs = gezi.list_files(FLAGS.test_input)
  logging.info('test_inputs', test_inputs)
  
  if test_inputs:
    test_dataset_ = Dataset('test')
    test_dataset = test_dataset_.make_batch(batch_size, test_inputs) 
    num_test_examples = test_dataset_.num_examples_per_epoch('test')
    num_test_steps_per_epoch = -(-num_test_examples // batch_size) if num_test_examples else None
  else:
    test_dataset = None

  learning_rate = tfe.Variable(FLAGS.learning_rate, name="learning_rate")
  optimizer = melt.get_optimizer(FLAGS.optimizer)(learning_rate)
  
  summary = tf.contrib.summary
  writer = summary.create_file_writer(FLAGS.model_dir + '/epoch')
  writer_train = summary.create_file_writer(FLAGS.model_dir + '/train')
  writer_valid = summary.create_file_writer(FLAGS.model_dir + '/valid')
  global_step = tf.train.get_or_create_global_step()

  checkpoint = tf.train.Checkpoint(
        learning_rate=learning_rate, 
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
  checkpoint.restore(latest_checkpoint)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

  # logging.info('Latest checkpoint:', latest_checkpoint)

  #model.load_weights(os.path.join(ckpt_dir, 'ckpt-1'))
  #model.save('./weight3.hd5')

  start_epoch = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint else 0
  # TODO currently not support 0.1 epoch.. like this
  num_epochs = FLAGS.num_epochs
  
  if valid_dataset:
    if evaluate_fn is not None:
      vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
    elif eval_fn:
      model_path = None if not write_valid else tf.train.latest_checkpoint(ckpt_dir)
      names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:]

      vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                             names, valid_write_fn, num_valid_steps_per_epoch,
                             suffix=valid_suffix)
  
    logging.info2('epoch:%d/%d' % (start_epoch, num_epochs), 
                  ['%s:%.5f' % (name, val) for name, val in zip(names, vals)])

  for epoch in range(start_epoch, num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_valid_loss_avg = tfe.metrics.Mean()

    for i, (x, y) in tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True):
      #if global_step.numpy() == 0:
      #loss = loss_fn(model, x, y, training=True)
      loss, grads = melt.eager.grad(model, x, y, loss_fn)
      optimizer.apply_gradients(zip(grads, model.variables))
      epoch_loss_avg(loss)  # add current batch loss

      if global_step.numpy() % FLAGS.interval_steps == 0:
        #checkpoint.save(checkpoint_prefix)
        if valid_dataset2:
          x, y = next(iter(valid_dataset2))
          valid_loss = loss_fn(model, x, y)
          epoch_valid_loss_avg(valid_loss)

          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'batch_size:%d' % x[0].shape[0],
                      'learning_rate:%.3f' % learning_rate.numpy(),
                      'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                      'valid_loss::%.4f' % epoch_valid_loss_avg.result().numpy())
          with writer_valid.as_default(), summary.always_record_summaries():
            summary.scalar('step/loss', epoch_valid_loss_avg.result().numpy())
            writer_valid.flush()
        else:
          logging.info('epoch:%.2f/%d' % ((epoch + i / num_steps_per_epoch), num_epochs), 
                      'step:%d' % global_step.numpy(), 
                      'batch_size:%d' % batch_size,
                      'learning_rate:%.3f' % learning_rate.numpy(),
                      'train_loss:%.4f' % epoch_loss_avg.result().numpy())                
        with writer_train.as_default(), summary.always_record_summaries():
          summary.scalar('step/loss', epoch_loss_avg.result().numpy())
          writer_train.flush()
      
      # if i == 5:
      #   print(i, '---------------------save')
      #   print(len(model.trainable_variables))
      ## TODO FIXME seems save weighs value not ok... not the same as checkpoint save
      #   model.save_weights(os.path.join(ckpt_dir, 'weights'))
      #   checkpoint.save(checkpoint_prefix)
      #   exit(0)


      global_step.assign_add(1)
      if epoch == start_epoch and i == 0:
        logging.info(model.summary())

    logging.info('epoch:%d/%d' % (epoch + 1, num_epochs), 
                 'step:%d' % global_step.numpy(), 
                 'batch_size:%d' % batch_size,
                 'learning_rate:%.3f' % learning_rate.numpy(),
                 'train_loss:%.4f' % epoch_loss_avg.result().numpy(),
                 'valid_loss::%.4f' % epoch_valid_loss_avg.result().numpy())

    checkpoint.save(checkpoint_prefix)

    if valid_dataset and (epoch + 1) % FLAGS.valid_interval_epochs == 0:
      if evaluate_fn is not None:
        vals, names = evaluate_fn(model, valid_dataset, tf.train.latest_checkpoint(ckpt_dir), num_valid_steps_per_epoch)
      elif eval_fn:
        model_path = None if not write_valid else tf.train.latest_checkpoint(ckpt_dir)
        names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:]

        vals, names = evaluate(model, valid_dataset, eval_fn, model_path, 
                               names, valid_write_fn, num_valid_steps_per_epoch)
    
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
                  infer_names, infer_write_fn, num_test_steps_per_epoch,
                  suffix=infer_suffix)
      else:
         inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)
