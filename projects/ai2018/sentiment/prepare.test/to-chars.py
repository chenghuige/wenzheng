#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-chars.py
#        \author   chenghuige  
#          \date   2018-10-28 08:37:28.846557
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import six 
  
assert six.PY3 

import pandas as pd

from projects.ai2018.sentiment.prepare import filter

from tqdm import tqdm
import traceback

ifile = sys.argv[1]
ofile = sys.argv[2]

ids_set = set()
fm = 'w'
if os.path.exists(ofile):
  fm = 'a'
  for line in open(ofile):
    ids_set.add(line.split('\t')[0])

print('%s already done %d' % (ofile, len(ids_set)))

num_errs = 0
with open(ofile, fm) as out:
  df = pd.read_csv(ifile, lineterminator='\n')
  contents = df['content'].values 
  ids = df['id'].values
  for i in tqdm(range(len(df)), ascii=True):
    if str(ids[i]) in ids_set:
      continue
    #if i != 2333:
    #  continue
    #print(gezi.cut(filter.filter(contents[i]), type_))
    try:
      l = []
      for ch in filter.filter(contents[i]):
        l.append(ch)
      print(' '.join(l), file=out)
    except Exception:
      if num_errs == 0:
        print(traceback.format_exc())
      num_errs += 1
      continue
    #exit(0)

print('num_errs:', num_errs, 'ratio:', num_errs / len(df))
