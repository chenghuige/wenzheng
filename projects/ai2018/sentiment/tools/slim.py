#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   remove-content.py
#        \author   chenghuige  
#          \date   2018-11-13 20:52:59.621179
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd 

df = pd.read_csv(sys.argv[1])
df = df.sort_values('id')

contents = ['abc'] * len(df)
df['content'] = contents 

df.to_csv(sys.argv[1].replace('.csv', '.slim.csv'), index=False, encoding="utf_8_sig")
