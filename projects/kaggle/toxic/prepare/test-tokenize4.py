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

  #results = tokenizer.full_tokenize(text)
  #print(results.tokens)
  #print(len(results.tokens))
  #print(list(zip(results.tokens, results.poses)))
  #print(list(zip(results.ori_tokens, results.poses)))
  #print(list(zip(results.tokens, results.tags)))
  #print(list(zip(results.tokens, results.ners)))
  ## print(list(zip(results.tokens, results.attributes)))

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


tokenize('''ok im talking to you as a wikipedian so please forget from this conflict and focused on the facts ok. the herut was never israeli minister therefor she can not be the major party ok. and about her policy i see you know hebrew so read this המפלגה אימצה קו שנחשב אז ימני קיצוני, ובין היתר תבעה את החזרה מההסכמה לתוכנית החלוקה. סיסמתה תאמה את קו אצ""""ל, """"שתי גדות לירדן, זו שלנו זו גם כן"""" http://he.wikipedia.org/wiki/%D7%9E%D7%A0%D7%97%D7%9D_%D7%91%D7%92%D7%99%D7%9F for god sake this is wikipedia is mean to bring people facts we should not enter this conflict here ok there is forums for that. i won't even argue with you about this conflict if you want you can be a member here https://www.facebook.com/JewishVoiceforPeace?fref=ts and write your opinion wiki is not the right place
''')
tokenize('chink fag ? 216.145.245.93')
tokenize('''peoples cock'n'balls.''')

tokenize('Thanks for the barnstar, I greatly appreciate it. BW21.lackWatch21')
tokenize('Credibility_node_ue')
tokenize('oh thats right.......it doesnt')
tokenize('Could you please stop your disruptive edits on Muamer Tanković.')
tokenize('Images in the societal attitudes article should be about societal attutudes.')
tokenize('95.43.48.134   12:59, 30 January 2008 {UTC}')
tokenize('fu**ck')
tokenize('Ni(gge)rMan')
tokenize('Hi Edivorce, thank you for your supportive comment.')
tokenize('discussion pages using four tildes (~~~~)')

tokenize('actually his given name Cao(操) didnot mean fuck, the exact one is 肏.(Kèyì)')

tokenize('FREEDOM!FREEDOM!FREEDOM!FREEDOM!FREEDOM!')
