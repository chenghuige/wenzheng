#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   __init__.py
#        \author   chenghuige  
#          \date   2016-08-15 16:32:00.341661
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
from gezi.timer import *
from gezi.nowarning import * 
from gezi.gezi_util import * 
from gezi.avg_score import *
from gezi.zhtools import *
from gezi.util import * 
from gezi.rank_metrics import *
from gezi.topn import *
from gezi.vocabulary import Vocabulary
from gezi.word_counter import WordCounter
from gezi.ngram import *
from gezi.hash import *

#if using baidu segmentor set encoding='gbk'
encoding='utf8' 
#encoding='gbk'

try:
  import matplotlib
  matplotlib.use('Agg')
except Exception:
  pass 

import traceback

from gezi.segment import *
#try:
#  from gezi.libgezi_util import *
#  import gezi.libgezi_util as libgezi_util
#  from gezi.segment import *
#  import gezi.bigdata_util
#except Exception:
#  print(traceback.format_exc(), file=sys.stderr)
#  print('import libgezi, segment bigdata_util fail')
#
#try:
#  from gezi.pydict import *
#except Exception:
#  #print(traceback.format_exc(), file=sys.stderr)
#  #print('import pydict fail')
#  pass

try:
  from gezi.libgezi_util import *
except Exception:
  print(traceback.format_exc(), file=sys.stderr)

try:
  import gezi.metrics
except Exception:
  print(traceback.format_exc(), file=sys.stderr) 

try:
  import gezi.melt
  from gezi.melt import *
except Exception:
  print(traceback.format_exc(), file=sys.stderr)  
