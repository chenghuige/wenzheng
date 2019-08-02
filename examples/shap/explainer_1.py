#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   explainer_1.py
#        \author   chenghuige  
#          \date   2019-08-02 22:12:47.710013
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import seaborn as sns
import keras
import shap

#let's load the diamonds dataset
df=sns.load_dataset(name='diamonds')
print(df.head())
print(df.describe())  
