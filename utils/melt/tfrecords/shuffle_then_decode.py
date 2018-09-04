#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   read_sparse.py
#        \author   chenghuige  
#          \date   2016-08-15 20:13:06.751843
#   \Description  @TODO https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/contrib/slim/python/slim/data/
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gezi
import melt

# deprecated just using dataset_decode.py

def _read(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return [serialized_example]

def inputs(files, decode_fn, batch_size=64,
           num_epochs = None, num_threads=12, 
           shuffle_files=True, batch_join=True, shuffle_batch=True, 
           min_after_dequeue=None, seed=None, enqueue_many=False,
           fix_random=False, no_random=False, fix_sequence=False,
           allow_smaller_final_batch=False, 
           num_prefetch_batches=None, 
           dynamic_pad=False,
           buckets=None,
           length_index=None,
           length_fn=None,
           name='input'):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read serialized_example
  2. shuffle serialized_examples
  3. decdoe batch_serialized_examples
  notice read_sparse.inputs and also be used for dense inputs,but if you 
  only need to decode part from serialized_example, then read.inputs will 
  be better, less to put to suffle
  #--------decode example, can refer to libsvm-decode.py
  # def decode(batch_serialized_examples):
  #   features = tf.parse_example(
  #       batch_serialized_examples,
  #       features={
  #           'label' : tf.FixedLenFeature([], tf.int64),
  #           'index' : tf.VarLenFeature(tf.int64),
  #           'value' : tf.VarLenFeature(tf.float32),
  #       })

  #   label = features['label']
  #   index = features['index']
  #   value = features['value']

  #   return label, index, value 

  #string_input_reducer will shuffle files
  #shuffle will read file by file and shuffle withn file(in shuffle queue) 
  #shuffle_batch_join will read multiple files and shuffle in shuffle queue(from many files)

  To get fixed sequence 
  shuffle=False  so by this way the sequence is as your data input unchange
  or
  shuffle=True
  seed=1024 #set
  batch_join=False  by this way you have fixed random, so get same result
  NOTICE, shuffle=True,seed=1024,batch_join=True will not get same result
  shuffle=False,seed=1024,batch_join=True also, so batch_join seems seed only control inqueue random, can not get fixed result

  for no random -> fixed result set shuffle=False wihch will force batch_join=False then use batch
  for fixed random ->  shuffle=True, seed set or  fix_random=True
  read-records.py show above ok, but train-evaluate.py show not, only shuffle=False can get fixed result.. @FIXME strange
  for train-evaluate.py it looks you can set shuffle in string_input_producer True, but then must use batch,
  batch_join and shuffle_batch join all not fixed even with seed set, may be due to trainset two inputs read ?
  for read-records.py batch_join will be fixed, shuffle_batch_join not 

  defualt parmas will give max random...

  Args:
  decode: user defined decode 
  min_after_dequeue: set to >2w for production train, suggesed will be 0.4 * num_instances, but also NOTICE do not exceed mem
  #--default parmas will make most randomness
  shuffle_files: wehter shuffle file 
  shuffle_batch: batch or shuffle_batch
  batch_join: wether to use multiple reader or use one reader mutlitple thread
  fix_random: if True make at most random which can fix random result
  allow_smaller_final_batch: set True usefull if you want verify on small dataset
  #but seems only works here for single epoch case
  """
  #with tf.device('/cpu:0'):
  if isinstance(files, str):
    files = gezi.list_files(files)
  
  assert len(files) > 0

  if not min_after_dequeue : 
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
    num_threads = 1

  if no_random:
    shuffle_files = False
    batch_join = False
    shuffle_batch = False 

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
    #@TODO diff between tf.batch_join and tf.batch, batch_join below means shuffle_batch_join.. TODO
    if batch_join:
      batch_list = [_read(filename_queue) for _ in xrange(num_threads)]
      #print batch_list
      if shuffle_batch:
        batch_serialized_examples = tf.train.shuffle_batch_join(
            batch_list, 
            batch_size=batch_size, 
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=seed,
            enqueue_many=enqueue_many,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name='shuffle_batch_join_queue')
      else:
        batch_serialized_examples = tf.train.batch_join(
          batch_list, 
          batch_size=batch_size, 
          capacity=capacity,
          enqueue_many=enqueue_many,
          allow_smaller_final_batch=allow_smaller_final_batch,
          dynamic_pad=dynamic_pad,
          name='batch_join_queue')
    else:
      serialized_example = list(_read(filename_queue))
      #@FIXME... for bug now can not be more random if want fix random see D:\mine\tensorflow-exp\models\image-text-sim\train-evaluate-fixrandom.py
      if shuffle_batch:	      
        batch_serialized_examples = tf.train.shuffle_batch(	
            serialized_example, 
            batch_size=batch_size, 
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=seed,
            enqueue_many=enqueue_many,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name='shuffle_batch_queue')		    
      else:	    
        batch_serialized_examples = tf.train.batch(
            serialized_example, 
            batch_size=batch_size, 
            #@TODO to make really fxied result use num_threads=1, may be shuffle_batch will be fix random?
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=enqueue_many,
            allow_smaller_final_batch=allow_smaller_final_batch,
            dynamic_pad=dynamic_pad,
            name='batch_queue')

    return decode_fn(batch_serialized_examples) if decode_fn is not None else batch_serialized_examples


