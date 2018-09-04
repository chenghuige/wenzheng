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
  results, infos = tokenizer.tokenize(text)
  print(results)
  print(list(zip(results, infos)))
  print(len(results))

  results = tokenizer.full_tokenize(text)
  print(list(zip(results.tokens, results.ners)))

  # doc = gezi.doc(text)
  # #print(dir(doc))
  # for token in doc:
  #   #print(dir(token))
  #   assert token.pos_
  #   #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
  #   #       token.shape_, token.is_alpha, token.is_stop)
  #   print(token.text, token.pos, token.pos_, token.tag, token.tag_)
  #   print(token.prefix_, token.suffix_)
  #   #print(token.vector)
  #   print(token.rank)
  # print('name entities')
  # for ent in doc.ents:
  #   #print(dir(ent))
  #   print(ent.text, ent.start_char, ent.end_char, ent.label_)

tokenize('SECURITYFUCK dimension dimenison really fuck you')
tokenize('Uh I hate to break it to you but Jusdafax is not an administrator')
tokenize('I love beijing university. Mike is a Chinese boy. You are a ditry Frenchman')
tokenize('Apple is looking at buying U.K. startup for $1 billion')
tokenize('Both articles should be merged.  (Operibus anteire)')
tokenize('You have no right to tell people to die in a fire. Just because that IP address is being used for vandilism does not mean you can talk to people like that. You are in fact a bigger loser than the person who you block.",1,0,0,0,1,0,"Dear Mokele , <N> You have no right to tell people to die in a fire . Just because that IP address is being used for vandilism does not mean you can talk to people like that . You are in fact a bigger loser than the person who you block .')


