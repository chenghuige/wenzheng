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

import six 
assert six.PY2, 'must using py2 env to do simplify'
  
key = 'content'

if len(sys.argv) > 3:
  key = sys.argv[3]

df = pd.read_csv(sys.argv[1], lineterminator='\n')

contents = df[key].values 

num_modified = 0
num_errs = 0
for i in tqdm(range(len(contents)), ascii=True):
  try:
    scontent = gezi.to_simplify(contents[i])
  except Exception:
    num_errs += 1
    print(traceback.format_exc())
    continue
  if scontent != contents[i]:
    # print('------------------', i)
    # print(contents[i])
    # print(scontent)
    contents[i] = scontent
    num_modified += 1

df[key] = contents

print('modify ratio', num_modified / len(df))
print('num_errs', num_errs)

df.to_csv(sys.argv[2], index=False, encoding='utf_8_sig')
