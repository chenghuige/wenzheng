from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import json 
import random

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('comment_limit', 0, '2500 can cover 95% data, 25000 can cover all, 1000')
flags.DEFINE_integer('test_comment_limit', 0, '2500 can cover 95% data, 25000 can cover all, 1000')

flags.DEFINE_integer("char_limit", 16, "Limit length for character, 16")
flags.DEFINE_integer("ngram_limit", 80, "Limit length for ngrams")
flags.DEFINE_integer("ngram_buckets", 2000000, "ngram hash buckets")
flags.DEFINE_bool("save_char", True, "")
flags.DEFINE_integer("simple_char_limit", 1000, "")

flags.DEFINE_bool('ftngram', False, '')


