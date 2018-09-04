#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   reset-model-top-scope.py
#        \author   chenghuige  
#          \date   2016-09-29 21:20:54.473502
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string("input_checkpoint", "",
                           """TensorFlow variables file to load.""")
tf.app.flags.DEFINE_string("model_dir", "",
                           """load latest model from model dir, 
                           either model dir or input_checkpoint must be non empty""")

tf.app.flags.DEFINE_string("output_checkpoint", "",
                           """output checkpint with top scope changed""")

tf.app.flags.DEFINE_string("restore_op_name", "save/restore_all",
                           """The name of the master restore operator.""")

tf.app.flags.DEFINE_string("scope", "",
                           """input top scope""")

tf.app.flags.DEFINE_string("out_scope", "",
                           """out top scope""")

tf.app.flags.DEFINE_integer("index", 0,
                            """out index of checkpoints""")


import os, time
from google.protobuf import text_format

import melt 

def reset_model_top_scope():
  assert FLAGS.input_checkpoint or FLAGS.model_dir

  sess = tf.InteractiveSession()
  input_checkpoint = FLAGS.input_checkpoint if FLAGS.input_checkpoint \
    else melt.latest_checkpoint(FLAGS.model_dir)
  _, input_checkpoint_name = os.path.split(input_checkpoint)
  print('input_checkpoint:', input_checkpoint)

  meta_filename = ".".join([input_checkpoint, "meta"])
  while not os.path.exists(meta_filename):
    print('%s:not ready yet, try again'%meta_filename)
    time.sleep(3)
  saver = tf.train.import_meta_graph(meta_filename)
  saver.restore(sess, input_checkpoint)

  print('tf.all_variables:', [v.name for v in tf.all_variables()])
  scope = FLAGS.scope
  out_scope = FLAGS.out_scope if FLAGS.out_scope else '%s_%d'%(scope, FLAGS.index)
  output_checkpoint = FLAGS.output_checkpoint if FLAGS.output_checkpoint \
      else '/tmp/%s'%(input_checkpoint_name)
  print('scope:', scope)
  print('out_scope:', out_scope)

  src_vars = melt.variables_with_scope(scope)
  print([v.name for v in src_vars])

  out_vars = {v.name[:v.name.rfind(':')].replace(scope, out_scope, 1): v for v in src_vars}
  print('out_vars:', [key for key in out_vars])
  tf.train.Saver(var_list=out_vars).save(sess, output_checkpoint)

  print('save done to %s'%output_checkpoint)

  sess.close()


def main(unused_args):
  reset_model_top_scope()

if __name__ == "__main__":
  tf.app.run()
