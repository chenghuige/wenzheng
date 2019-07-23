#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2019-07-23 09:24:50.235252
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import numpy as np

def hack_lb(test_preds):
  id_codes = np.load('../input/aptos2019/aptos2019-test/small_id_codes.npy', allow_pickle = True)
  small_ids_df = pd.DataFrame(id_codes, columns=["id_code"])

  test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
  sample_df = small_ids_df
  sample_df["diagnosis"] = test_preds
  sub = pd.merge(test_df, sample_df, on='id_code', how='left').fillna(0)
  sub["diagnosis"] = sub["diagnosis"].astype(int)
  return sub
      

submit = pd.read_csv('../input/aptos2019-result/submission.csv') 
predicted = submit['diagnosis']

submit = hack_lb(predicted)
submit.to_csv('submission.csv', index=False)
submit.head()

