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

import pandas as pd
import gezi
  
key = 'content'

if len(sys.argv) > 3:
  key = sys.argv[3]

df = pd.read_csv(sys.argv[1])

contents = df[key].values 

for i in range(len(contents)):
  scontent = gezi.normalize(contents[i])
  if scontent != contents[i]:
    print('------------------', i)
    print(contents[i])
    print(scontent)
    contents[i] = scontent

df[key] = contents

df.to_csv(sys.argv[2], index=False, encoding='utf_8_sig')
