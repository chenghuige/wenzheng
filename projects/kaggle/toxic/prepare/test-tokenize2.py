#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test-tokenize.py
#        \author   chenghuige  
#          \date   2018-02-14 23:49:07.815632
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tokenizer 
import gezi

def tokenize(text):
  print(text)
  #results = tokenizer.tokenize(text)
  results = tokenizer.full_tokenize(text)
  print(results)
  print(gezi.segment.tokenize_filter_empty(text))
  
tokenize('{{unblock|Please')

tokenize('actually his given name Cao(操) didnot mean fuck, the exact one is 肏.(Kèyì)')

tokenize('''19 August 2011 (UTC)\nLook who is talking. Legilas is the user who's got Autopatrolled rights although he has  16 articles on the list.''')
tokenize('Hi\n\nwhy did you give User:Barneca a kindness star.')
tokenize('proud to be indonesial')
