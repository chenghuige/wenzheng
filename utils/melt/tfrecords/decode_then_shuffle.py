#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2016-08-15 20:10:44.183328
#   \Description   Read from TFRecords
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gezi
import melt

# deprecated just using dataset_decode.py

# TODO https://www.tensorflow.org/programmers_guide/datasets  using DATASET API instead 

def _read_decode(filename_queue, decode_fn, thread_id=0):
  reader = tf.TFRecordReader()
  _, decoded_example = reader.read(filename_queue)
  #TODO better handle? http://stackoverflow.com/questions/218616/getting-method-parameter-names-in-python
  # inspect.getargspec(aMethod)
  try:
    values = decode_fn(decoded_example)
  except Exception:
    values = decode_fn(decoded_example, thread_id)
  #---for safe, or decode can make sure this for single value turn to list []
  if not isinstance(values, (list, tuple)):
    values = [values]
  return values

def inputs(files, decode_fn, batch_size=64,
           num_epochs = None, num_threads=12, 
           shuffle_files=True, batch_join=True, shuffle_batch=True, 
           min_after_dequeue=None, seed=None, enqueue_many=False,
           fix_random=False, no_random=False, fix_sequence=False,
           allow_smaller_final_batch=False, 
           num_prefetch_batches=None, 
           dynamic_pad=False,
           bucket_boundaries=None,
           length_index=None,
           length_fn=None,
           keep_fn=None,
           name='input'):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read decode decoded_example
  2. shuffle decoded values
  3. return batch decoded values
  Args:
  decode: user defined decode #TODO should be decode_fn
  #---decode example
  # features = tf.parse_single_example(
  #     decoded_example,
  #     features={
  #         'feature': tf.FixedLenFeature([], tf.string),
  #         'name': tf.FixedLenFeature([], tf.string),
  #         'comment_str': tf.FixedLenFeature([], tf.string),
  #         'comment': tf.FixedLenFeature([], tf.string),
  #         'num_words': tf.FixedLenFeature([], tf.int64),
  #     })
  # feature = tf.decode_raw(features['feature'], tf.float32)
  # feature.set_shape([IMAGE_FEATURE_LEN])
  # comment = tf.decode_raw(features['comment'], tf.int64)
  # comment.set_shape([COMMENT_MAX_WORDS])
  # name = features['name']
  # comment_str = features['comment_str']
  # num_words = features['num_words']
  # return name, feature, comment_str, comment, num_words
  Returns:
  list of tensors
  """
  #with tf.device('/cpu:0'):
  if isinstance(files, str):
    files = gezi.list_files(files)

  assert len(files) > 0
    
  if not min_after_dequeue:
    min_after_dequeue = melt.tfrecords.read.MIN_AFTER_QUEUE
  if not num_epochs: 
    num_epochs = None
  
  if fix_random:
    if seed is None:
      seed = 1024
    shuffle_files = True  
    batch_join = False  #check can be True ?

    #to get fix_random 
    #shuffle_batch = True  and num_threads = 1 ok
    #shuffle_batch = False and num_threads >= 1 ok
    #from models/iamge-text-sim/read_records shuffle_batch = True will be quick, even single thread
    #and strange num_threas = 1 will be quicker then 12
    
    shuffle_batch = True
    num_threads = 1

    #shuffle_batch = False

  if fix_sequence:
    no_random = True 
    allow_smaller_final_batch = True
   
  if no_random:
    shuffle_files = False
    batch_join = False
    shuffle_batch = False 
    num_threads = 1

  if dynamic_pad:
    #use tf.batch
    shuffle_batch = False


  #shuffle=True
  #batch_join = True #setting to False can get fixed result
  #seed = 1024

  with tf.name_scope(name):
    filename_queue = tf.train.string_input_producer(
      files, 
      num_epochs=num_epochs,
      shuffle=shuffle_files,
      seed=seed)
    
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    #@TODO cifa10 always use num_prefetch_batches = 3, 3 * batch_size, check which is better
    if not num_prefetch_batches:
      num_prefetch_batches = num_threads + 3

    capacity = min_after_dequeue + num_prefetch_batches * batch_size

    if batch_join:
      batch_list = [_read_decode(filename_queue, decode_fn, thread_id) for thread_id in xrange(num_threads)]
      #print batch_list
      if shuffle_batch:
        batch = tf.train.shuffle_batch_join(
            batch_list, 
            batch_size=batch_size, 
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=seed,
            enqueue_many=enqueue_many,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name='shuffle_batch_join_queue')
      else:
        batch = tf.train.batch_join(
          batch_list, 
          batch_size=batch_size, 
          capacity=capacity,
          enqueue_many=enqueue_many,
          allow_smaller_final_batch=allow_smaller_final_batch,
          dynamic_pad=dynamic_pad,
          name='batch_join_queue')
    else:
      decoded_example = list(_read_decode(filename_queue, decode_fn))
      num_threads = 1 if fix_random else num_threads
      if bucket_boundaries:
        if not isinstance(bucket_boundaries, (list, tuple)):
          bucket_boundaries = [int(x) for x in bucket_boundaries.split(',') if x]
        if length_index is not None:
          input_length=decoded_example[length_index]
        else:
          assert length_fn is not None, 'you must set length_index or pass length_fn'
          input_length = length_fn(decoded_example)
        keep_input = input_length >= 1 if keep_fn is None else keep_fn(decoded_example)
        _, batch = tf.contrib.training.bucket_by_sequence_length(
              input_length=tf.to_int32(input_length),
              bucket_boundaries=bucket_boundaries,
              tensors=decoded_example,
              batch_size=batch_size,
              keep_input=keep_input,
              num_threads=num_threads,
              dynamic_pad=True,
              capacity=capacity,
              allow_smaller_final_batch=allow_smaller_final_batch,
              name="bucket_queue")
      else:
        if shuffle_batch:	    
          batch = tf.train.shuffle_batch(	
             decoded_example,
             batch_size=batch_size, 
             num_threads=num_threads,
             capacity=capacity,
             min_after_dequeue=min_after_dequeue,
             seed=seed,
             enqueue_many=enqueue_many,
             allow_smaller_final_batch=allow_smaller_final_batch,
             name='shuffle_batch_queue')
        else:
          #http://honggang.io/2016/08/19/tensorflow-data-reading/
          batch = tf.train.batch(
             decoded_example, 
             batch_size=batch_size, 
             num_threads=num_threads,
             capacity=capacity,
             enqueue_many=enqueue_many,
             allow_smaller_final_batch=allow_smaller_final_batch,
             dynamic_pad=dynamic_pad,
             name='batch_queue')

    return batch
