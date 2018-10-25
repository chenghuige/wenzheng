#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-simplify.py
#        \author   chenghuige  
#          \date   2018-10-19 12:58:07.505225
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from tqdm import tqdm
import pandas as pd
import gezi
import traceback 
import json
import six  
#you may need to ln ~/soft/bseg/ data,conf,lib to current path and run in pyenv(python2)
assert six.PY2, 'must using py2 env to do simplify'
  
for line in open(sys.argv[1]):
  m = json.loads(line.rstrip('\n')) 
  m['passage'] = gezi.to_simplify(m['passage'])
  m['query'] = gezi.to_simplify(m['query'])
  m['alternatives'] = gezi.to_simplify(m['alternatives'])
  print(json.dumps(m, ensure_ascii=False).encode('utf8'))
