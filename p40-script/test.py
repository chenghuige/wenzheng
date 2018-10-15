#!/usr/bin/env python
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2017-11-04 23:23:38.351100
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

command = 'cat run-head.txt > run.sh'
os.system(command)

with open('./run.sh', 'a') as out: 
  print('abcdefg----------', file=out)
  
  
