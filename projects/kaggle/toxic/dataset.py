# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import gezi
import melt
logging = melt.logging

from collections import namedtuple

from algos.config import NUM_CLASSES
import prepare.config

NUM_COMMENT_FEATURES = 7

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)

  def parser(self, example):
    comment_key = 'comment'

    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'comment_str':  tf.FixedLenFeature([], tf.string),
      'comment_tokens_str':  tf.FixedLenFeature([], tf.string),
        comment_key: tf.VarLenFeature(tf.int64),
      'comment_chars':  tf.VarLenFeature(tf.int64),
      'comment_ngrams': tf.VarLenFeature(tf.int64),
      #'comment_fngrams': tf.VarLenFeature(tf.int64),
      'simple_chars':  tf.VarLenFeature(tf.int64),
      #'simple_ngrams': tf.VarLenFeature(tf.int64),
      'tokens_info': tf.VarLenFeature(tf.float32),
      #'comment_info': tf.VarLenFeature(tf.float32),
      #'comment_info':  tf.FixedLenFeature([NUM_COMMENT_FEATURES], tf.float32),
      'pos': tf.VarLenFeature(tf.int64),
      'tag': tf.VarLenFeature(tf.int64),
      'ner': tf.VarLenFeature(tf.int64),
      'classes': tf.FixedLenFeature([NUM_CLASSES], tf.float32),
      #'weight': tf.FixedLenFeature([1], tf.float32),
      }

    # # support weight from v17, but notice token change from v16
    # if not ('TOXIC_VERSION' in os.environ and int(os.environ['TOXIC_VERSION']) <= 16):
    #   features_dict['weight'] = tf.FixedLenFeature([1], tf.float32)

    # if FLAGS.use_word:
    #   features_dict['comment'] = tf.VarLenFeature(tf.int64)
    # if FLAGS.use_char:
    #   features_dict['comment_chars'] = tf.VarLenFeature(tf.int64)  
    # if FLAGS.use_simple_char:
    #   features_dict['simple_chars'] = tf.VarLenFeature(tf.int64)
    # if FLAGS.use_token_info:
    #   features_dict['tokens_info'] = tf.VarLenFeature(tf.float32),
    # if FLAGS.use_pos:
    #   features_dict['pos'] = tf.VarLenFeature(tf.int64)
    # if FLAGS.use_tag:
    #   features_dict['tag'] = tf.VarLenFeature(tf.int64)
    # if FLAGS.use_ner:
    #   features_dict['ner'] = tf.VarLenFeature(tf.int64)

    features = tf.parse_single_example(example, features=features_dict)

    id = features['id']

    comment = None
    comment_chars = None
    simple_chars = None
    tokens_info = None
    pos = None
    tag = None
    ner = None

    try:
      weight = features['weight'][0]
    except Exception:
      weight = tf.constant([1.])

    #----var len features
    #if FLAGS.use_word:
    comment = features[comment_key]
    comment = melt.sparse_tensor_to_dense(comment)    
    features[comment_key] = comment

    #if FLAGS.use_char:
    comment_chars = features['comment_chars']
    comment_chars = melt.sparse_tensor_to_dense(comment_chars)
    features['comment_chars'] = comment_chars

    #if FLAGS.use_token_info:
    tokens_info = features['tokens_info']
    tokens_info = melt.sparse_tensor_to_dense(tokens_info)
    features['tokens_info'] = tokens_info

    #comment_info = features['comment_info']
    #comment_info = melt.sparse_tensor_to_dense(comment_info)

    classes = features['classes']
    comment_str = features['comment_str']
    comment_tokens_str = features['comment_tokens_str']

    #----------- simple chars (per whole comment),  'what a pity' -> 'w|h|a|t| |a| |p|i|t|y'
    # TODO simple char can change to use ngram model seq or sum ngram 
    #if FLAGS.use_simple_char:
    simple_chars = features['simple_chars']
    simple_chars = melt.sparse_tensor_to_dense(simple_chars)
    features['simple_chars'] = simple_chars

    #simple_ngrams = features['simple_ngrams']
    #simple_ngrams = melt.sparse_tensor_to_dense(simple_ngrams)

    #if FLAGS.use_pos:
    pos = features['pos']
    pos = melt.sparse_tensor_to_dense(pos)
    features['pos'] = pos()

    tag = features['tag']
    tag = melt.sparse_tensor_to_dense(tag)
    features['tag'] = tag

    ner = features['ner']
    ner = melt.sparse_tensor_to_dense(ner)
    features['ner'] = ner

    comment_ngrams = features['comment_ngrams']
    comment_ngrams = melt.sparse_tensor_to_dense(comment_ngrams)
    features['comment_ngrams'] = comment_ngrams

    # comment_fngrams = features['comment_fngrams']
    # comment_fngrams = melt.sparse_tensor_to_dense(comment_fngrams)

    char_vocab = gezi.Vocabulary(FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt'))

    #--- will this be slow then after padding slice ?  
    #--- notice here will be shape(,) 1 d, since is parse_single_example then will batch in dtaset.padded_batch
    #--- not used much, actually, just limit max length when building tfrecords (for toxic can not limit)
    #--- then when train use bucket method like buckets=[400] will be fine
    #--- limit length , might be better do int when gen tf record
    limit = FLAGS.comment_limit if self.subset is 'train' else FLAGS.test_comment_limit
    if limit:
      comment = comment[:limit]
      comment_chars = comment_chars[:limit * FLAGS.char_limit]
      tokens_info = tokens_info[:limit * len(attribute_names)]
      if FLAGS.use_pos:
        pos = pos[:limit]
        tag = tag[:limit]
        ner = ner[:limit]

    if FLAGS.use_pos:
      pos_vocab = gezi.Vocabulary(FLAGS.vocab.replace('vocab.txt', 'pos_vocab.txt'))
      tag_vocab = gezi.Vocabulary(FLAGS.vocab.replace('vocab.txt', 'tag_vocab.txt'))
      ner_vocab = gezi.Vocabulary(FLAGS.vocab.replace('vocab.txt', 'ner_vocab.txt'))
      def append_start_end_mark(tag, start, end):
        tag_list = [tag]
        # if FLAGS.encode_start_mark:
        #   tag_list.insert(0, tf.constant([start], dtype=tf.int64))
        # if FLAGS.encode_end_mark:
        #   tag_list.append(tf.constant([end], dtype=tf.int64))
        if len(tag_list) > 1:
          tag = tf.concat(tag_list, 0)
        return tag  
      
      pos = append_start_end_mark(pos, pos_vocab.start_id(), pos_vocab.end_id())
      tag = append_start_end_mark(tag, tag_vocab.start_id(), tag_vocab.end_id())  
      ner = append_start_end_mark(ner, ner_vocab.start_id(), ner_vocab.end_id())

    #-----------comment deal start end mark
    comment_list = [comment]
    # if FLAGS.encode_start_mark:
    #   logging.info('add encode start mark')
    #   comment_list.insert(0, tf.constant([vocabulary.start_id()], dtype=tf.int64))
    # if FLAGS.encode_end_mark:
    #   logging.info('add encode end mark')
    #   comment_list.append(tf.constant([vocabulary.end_id()], dtype=tf.int64))
    
    if len(comment_list) > 1:
      comment = tf.concat(comment_list, 0)

    char_comment_limit = FLAGS.comment_limit if FLAGS.save_char else 1

    #----------deal tokens info  # TODO tokens embedding ? maybe
    if FLAGS.use_token_info:
      tokens_info_list = [tokens_info]
      # if FLAGS.encode_start_mark:
      #   tokens_info_list.insert(0, tf.constant(attribute_default_values, dtype=tf.float32))
      # if FLAGS.encode_end_mark:
      #   tokens_info_list.append(tf.constant(attribute_default_values, dtype=tf.float32))
      
      if len(tokens_info_list) > 1:
        tokens_info = tf.concat(tokens_info_list, 0)  

    #---------comment chars
    if FLAGS.use_char:
      comment_chars_list = [comment_chars]
      # if FLAGS.encode_start_mark:
      #   #comment_chars_list.insert(0, tf.ones([FLAGS.char_limit], dtype=tf.int64))
      #   # TODO below indices[15794,0] = 593 is not in [0, 593), because in merge_char_emb no start and end mark save 
      #   # Will change to use below next time merge-char-emb add start and end mark
      #   comment_chars_list.insert(0, tf.scatter_nd(tf.constant([[0]]), tf.constant([char_vocab.start_id()], dtype=tf.int64), tf.constant([FLAGS.char_limit])))
      # if FLAGS.encode_end_mark:
      #   #comment_chars_list.append(tf.ones([FLAGS.char_limit], dtype=tf.int64))
      #   comment_chars_list.append(tf.scatter_nd(tf.constant([[0]]), tf.constant([char_vocab.end_id()], dtype=tf.int64), tf.constant([FLAGS.char_limit])))

      if len(comment_chars_list) > 1:
        comment_chars = tf.concat(comment_chars_list, 0)  

    #---------comment ngrams
    if FLAGS.use_ngrams:
      ngram_vocab = gezi.Vocabulary(FLAGS.vocab.replace('vocab.txt', 'ngram_vocab.txt'))
      comment_ngrams_list = [comment_ngrams]
      # if FLAGS.encode_start_mark:
      #   comment_ngrams_list.insert(0, tf.scatter_nd(tf.constant([[0]]), tf.constant([ngram_vocab.start_id()], dtype=tf.int64), tf.constant([FLAGS.char_limit])))
      # if FLAGS.encode_end_mark:
      #   comment_ngrams_list.append(tf.scatter_nd(tf.constant([[0]]), tf.constant([ngram_vocab.end_id()], dtype=tf.int64), tf.constant([FLAGS.char_limit])))
      
      if len(comment_ngrams_list) > 1:
        comment_ngrams = tf.concat(comment_ngrams_list, 0)

    simple_chars_list = [simple_chars]
    # if FLAGS.encode_start_mark:
    #   simple_chars_list.insert(0, tf.constant([char_vocab.start_id()], dtype=tf.int64))
    # if FLAGS.encode_end_mark:
    #   simple_chars_list.append(tf.constant([char_vocab.end_id()], dtype=tf.int64))
    if len(simple_chars_list) > 1:
      simple_chars = tf.concat(simple_chars_list, 0)    


    features[comment_key] = comment
    features['comment_chars'] = comment_chars
    features['simple_chars'] = simple_chars
    features['comment_ngrams'] = comment_ngrams

    x = features
    y = classes
    return x, y

