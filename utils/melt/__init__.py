from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import matplotlib
  matplotlib.use('Agg')
except Exception:
  pass


import tensorflow as tf 
import traceback

import sys
print('tensorflow_version:', tf.__version__, file=sys.stderr) 

import torch
print('torch_version:', torch.__version__, file=sys.stderr) 

from melt.training import training as train 
import melt.training 

import melt.utils
from melt.utils import logging
from melt.utils import EmbeddingSim

from melt.util import *
from melt.ops import *
from melt.variable import * 
from melt.tfrecords import * 

from melt.inference import *

import melt.layers

import melt.slim2

import melt.flow
from melt.flow import projector_config

import melt.metrics 
from melt.metrics import *

try:
  import melt.apps
except Exception:
	print(traceback.format_exc(), file=sys.stderr)
	pass

import melt.rnn 
import melt.cnn 
import melt.encoder 

import melt.seq2seq 
import melt.image  
from melt.image import *

import melt.losses  

import melt.eager 

import melt.torch 
