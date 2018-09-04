#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige  
#          \date   2018-02-25 17:17:55.001974
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('is_twitter', False, '')

import sys, os
import re

import gezi  
  
def normalize(text):
  text = gezi.filter_quota(text)
  # hack for spacy tokenzier
  #text = text.replace('\n', ' \x01 ') # for spacy to tokenize left |\x01| alonw
  text = text.replace('\n', ' NEWLINE ')
  #text = text.replace('''don't''', ' \x02 ')
  #text = text.replace('(', ' \x02 ')

  if FLAGS.is_twitter:
    text = glove_twitter_preprocess(text)

  return text

tokens_map = {
  #'\x01': '<N>',
  'NEWLINE': '<N>',
  '<IP>': '<IP>',
  '<HTTP>': '<HTTP>',
  #'\x02': '''don't''',
  #'\x02': '('
  'TWURL': '<URL>',
  'TWUSER': '<USER>',
  'TWHEART': '<HEART>',
  'TWNUMBER': '<NUMBER>',
  'TWSMILE': '<SMILE>',
  'TWLOLFACE': '<LOLFACE>',
  'TWSADFACE': '<SADFACE>',
  'TWNEUTRALFACE': '<NEUTRALFACE>',
  'TWREPEAT': '<REPEAT>',
  'TWELONG': '<ELONG>',
}

# def glove_twitter_preprocess(text):
#     """
#     adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

#     """
#     # Different regex parts for smiley faces
#     eyes = "[8:=;]"
#     nose = "['`\-]?"
#     text = re.sub("https?:* ", "<URL>", text)
#     text = re.sub("www.* ", "<URL>", text)
#     text = re.sub("\[\[User(.*)\|", '<USER>', text)
#     text = re.sub("<3", '<HEART>', text)
#     text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
#     text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
#     text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
#     text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
#     text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
#     text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
#     text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
#     text = re.sub("/", " / ", text)
#     text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
#     text = re.sub("([!]){2,}", "! <REPEAT>", text)
#     text = re.sub("([?]){2,}", "? <REPEAT>", text)
#     text = re.sub("([.]){2,}", ". <REPEAT>", text)
#     pattern = re.compile(r"(.)\1{2,}")
#     text = pattern.sub(r"\1" + " <ELONG>", text)

#     return text

def glove_twitter_preprocess(text):
    """
    adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    """
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?:* ", " TWURL ", text)
    text = re.sub("www.* ", " TWURL ", text)
    text = re.sub("\[\[User(.*)\|", ' TWUSER ', text)
    text = re.sub("<3", ' TWHEART ', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " TWNUMBER ", text)
    text = re.sub(eyes + nose + "[Dd)]", ' TWSMILE ', text)
    text = re.sub("[(d]" + nose + eyes, ' SMILE ', text)
    text = re.sub(eyes + nose + "p", ' TWLOLFACE ', text)
    text = re.sub(eyes + nose + "\(", ' TWSADFACE ', text)
    text = re.sub("\)" + nose + eyes, ' TWSADFACE ', text)
    text = re.sub(eyes + nose + "[/|l*]", ' TWNEUTRALFACE ', text)
    text = re.sub("/", " / ", text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " TWNUMBER ", text)
    text = re.sub("([!]){2,}", "!  TWREPEAT ", text)
    text = re.sub("([?]){2,}", "?  TWREPEAT ", text)
    text = re.sub("([.]){2,}", ".  TWREPEAT ", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " TWELONG ", text)

    return text
