#!/usr/bin/env python
# ==============================================================================
#          \file   show-var-of-model.py
#        \author   chenghuige  
#          \date   2017-09-06 07:52:34.258312
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import melt

from tensorflow.python import pywrap_tensorflow
model_dir = sys.argv[1]
var_name = sys.argv[2]
checkpoint_path = melt.get_model_path(model_dir)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
  if var_name in key:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
