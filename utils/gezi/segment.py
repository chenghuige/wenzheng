#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   segment.py
#        \author   chenghuige  
#          \date   2016-08-25 19:09:51.756927
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi.nowarning
import gezi
import sys
#sys.path.append("./predeal")
#import traceback
try:
  import gezi.keyword_fetch as kf
except Exception:
  #print(traceback.format_exc())
  pass
  
from gezi.libgezi_util import get_single_cns
import six
import re

#TODO need improve
def segment_gbk_char(text, cn_only=False):
  assert six.PY2
  l = []
  pre_is_cn = False
  unicode_text = text.decode('gbk', 'ignore')
  for word in unicode_text:
    if u'\u4e00' <= word <= u'\u9fff':
      pre_is_cn = True
      if l:
        l.append(' ')
    else:
      l.append(' ')
      if pre_is_cn:
        pre_is_cn = False
    if not cn_only or pre_is_cn:
      l.append(word)
  text = ''.join(l)
  text = text.encode('gbk', 'ignore')
  l = text.split()
  return [x.strip() for x in l if x.strip()]  


#def segment_utf8_charv1(text, cn_only=False):
#  """
#  Code by limin
#  """
#  content= text.replace('===>>>签名：','')
#  if six.PY2:
#    content = re.sub(ur'[\s]+', '', content.decode('utf8', 'ignore'))
#    sentance = content.encode('utf8', 'ignore')
#    seglist = sentance
#    output = ' '.join(seglist.decode('utf8','ignore')).encode('utf8','ignore')
#  else:
#    output = ' '.join(list(content))
#  return output 

def segment_utf8_char(text, cn_only=False):
  if not six.PY2 and not cn_only:
    return [x.strip() for x in text if x.strip()]
  l = []
  pre_is_cn = False
  if six.PY2:
    unicode_text = text.decode('utf-8', 'ignore')
  else:
    unicode_text = text
  for word in unicode_text:
    #print('-----------word', word, pre_is_cn)
    if u'\u4e00' <= word <= u'\u9fff':
      pre_is_cn = True
      if l:
        l.append(' ')
    else:
      l.append(' ')
      if pre_is_cn:
        pre_is_cn = False
    if not cn_only or pre_is_cn:
      l.append(word)
  text = ''.join(l)
  if six.PY2:
    text = text.encode('utf-8')
  l = text.split()
  return [x.strip() for x in l if x.strip()]    

def segment_utf8_pinyin(text, cn_only=False):
    """
    by yuhan
    """
    pinyin_char = kf.deform_dis_get(text)
    #print (pinyin_char)
    l = pinyin_char.split()
    return [x.strip() for x in l if x.strip()]

def segment_utf8_pinyin2(text, cn_only=False):
    """
    by yuhan
    """
    pinyin_char = kf.deform_dis_get_v2(text)
    #print (pinyin_char)
    l = pinyin_char.split()
    return [x.strip() for x in l if x.strip()]


def segment_en(text):
  l = text.strip().split()
  return [x.strip() for x in l if x.strip()]

def filter_quota(text):
  return text.replace("''", '" ').replace("``", '" ')

# def try_strip(text, ch='-'):
#   otext = text.strip(ch)
#   if otext:
#     return otext 
#   return text
    
nlp = None
def tokenize(text):
  #text = filter_quota(text)
  #try:
  import spacy
  global nlp 
  if nlp is None:
    nlp = spacy.blank("en")
  if six.PY2:
    text = text.decode('utf-8')
  doc = nlp(text)
  if six.PY2:
    return [token.text.encode('utf-8') for token in doc]
  else:
    return [token.text for token in doc]
  #except Exception:
  #  return segment_en(text)

full_nlp = None

def init_spacy_full():
  import spacy
  global full_nlp
  if full_nlp is None:
    timer = gezi.Timer('load spacy model')
    full_nlp = spacy.load('/usr/local/lib/python3.5/dist-packages/spacy/data/en_core_web_md-2.0.0/')
    timer.print_elapsed()   

def doc(text):
  import spacy
  global full_nlp 
  if full_nlp is None:
    # TODO FIXME
    #full_nlp = spacy.load("en")
    timer = gezi.Timer('load spacy model')
    full_nlp = spacy.load('/usr/local/lib/python3.5/dist-packages/spacy/data/en_core_web_md-2.0.0/')
    timer.print_elapsed()

  if six.PY2:
    text = text.decode('utf-8')
  doc = full_nlp(text)
  return doc

def tokenize_filter_empty(text):
  #text = filter_quota(text)
  #try:
  import spacy
  global nlp 
  if nlp is None:
    nlp = spacy.blank("en")
  if six.PY2:
    text = text.decode('utf-8')
  doc = nlp(text)
  if six.PY2:
    return list(filter(lambda x: x.strip(), [token.text.encode('utf-8') for token in doc]))
  else:
    return list(filter(lambda x: x.strip(), [token.text for token in doc]))

try:
  import jieba 
except Exception:
  pass 

segment_char = segment_utf8_char
segment_pinyin = segment_utf8_pinyin

#TODO hack how to better deal? now for c++ part must be str..
#TODO py3
class JiebaSegmentor(object):
  def __init__(self):
    pass

  def segment_basic_single(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=True)]
    results += [word for word in jieba.cut(text)]
    return results  

  def segment_basic_single_all(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=False)]
    results += [word for word in jieba.cut(text)]
    return results  

  def segment_full_single(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=True)]
    results += [word for word in jieba.cut_for_search(text)]
    return results  

  def Segment(self, text, method='basic'):
    """
    default means all level combine
    """
    method = method.replace('phrase', 'basic')

    words = None
    if method == 'default' or method == 'basic' or method == 'exact':
      words = [x for x in jieba.cut(text, cut_all=False)]
    elif method == 'basic_digit':
      words = [x for x in jieba.cut(text, cut_all=False)]
      def sep_digits(word):
        l = []
        s = ''
        for c in word:
          if c.isdigit():
            if s:
              l.append(s)
            l.append(c)
            s = ''
          else:
            s += c
        if s:
          l.append(s)
        return l
      l = []
      for w in words:
        l += sep_digits(w)
      words = l
    elif method == 'basic_single' or method == 'exact_single':
      words = self.segment_basic_single(text)
    elif method == 'basic_single_all':
      words = self.segment_basic_single_all(text)
    elif method == 'search':
      words = jieba.cut_for_search(text)
    elif method == 'cut_all':
      words = jieba.cut(text, cut_all=True)
    elif method == 'all' or method == 'full':
      words = self.segment_full_single(text)
    elif method == 'en':
      words = segment_en(text)
    elif method == 'char':
      words = segment_char(text)
    elif method == 'pinyin':
      words = segment_pinyin(text)
    elif method == 'pinyin2':
      words = segment_utf8_pinyin2(text)
    elif method == 'basic2':
      words = kf.deform_dis_get_words(text)
    elif method =='char_pinyin':
      from pypinyin import lazy_pinyin as pinyin
      #return pinyin(text)
      return [x.strip() for x in pinyin(text)]
    elif method =='char_pinyin2':
      from pypinyin import lazy_pinyin as pinyin
      return [''.join(pinyin(x)) for x in text]
    elif method == 'char_then_pinyin':
      # In [2]: pinyin('补肾微信xtx0329')
      # Out[2]: ['bu', 'shen', 'wei', 'xin', 'xtx0329'] 
      # so as 补,肾,微,信,x,t,x,0,3,2,9,<S>,bu,shen,wei,xin,<UNK>
      from pypinyin import lazy_pinyin as pinyin
      return  segment_char(text) + ['<S>'] + [x.strip() for x in pinyin(text)]
    elif method == 'char_then_pinyin2':
      from pypinyin import lazy_pinyin as pinyin
      return  segment_char(text) + ['<S>'] + [''.join(pinyin(x)).strip() for x in text if x.strip()]
    elif method == 'word_char':
      return [x for x in jieba.cut(text, cut_all=False)] + ['<S>'] + segment_char(text)
    elif method == 'word_char_pinyin':
      from pypinyin import lazy_pinyin as pinyin
      return [x for x in jieba.cut(text, cut_all=False)] + ['<S>'] + segment_char(text) + ['<S>'] + [x.strip() for x in pinyin(text)]
    elif method == 'word_char_pinyin2':
      from pypinyin import lazy_pinyin as pinyin
      return [x for x in jieba.cut(text, cut_all=False)] + ['<S>'] + segment_char(text) + ['<S>'] + [''.join(pinyin(x)).strip() for x in text if x.strip()]
    elif method == 'tab':
      words = text.strip().split('\t')
    elif method == 'white_space':
      words = text.strip().split()
    else:
      raise ValueError('%s not supported'%method)

    #words = [w for w in words]
    
    if six.PY2:
      for i in range(len(words)):
        if isinstance(words[i], unicode):
          words[i] = words[i].encode('utf-8')
          
    return words

Segmentor = JiebaSegmentor


if gezi.encoding == 'gbk' or gezi.env_has('BSEG'):
  import libgezi
  import libsegment
  seg = libsegment.Segmentor

  segment_char = segment_gbk_char 

  class BSegmentor(object):
    def __init__(self, data='./data/wordseg', conf='./conf/scw.conf'):
      assert six.PY2, 'baidu segmentor now only support py2'
      seg.Init(data_dir=data, conf_path=conf)

    #@TODO add fixed pre dict like 1500w keyword ? 
    def segment_nodupe_noseq(self, text):
      results = set()
      for word in seg.Segment(text):
        results.add(word)
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        results.add(word)
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        results.add(word)
      for word in get_single_cns(text):
        results.add(word)
      return list(results)

    def Segment_nodupe_noseq(self, text):
      return self.segment_noseq(text)

    def segment_nodupe(self, text):
      results = [word for word in seg.Segment(text)]
      results += [wrod for word in seg.Segment(text, libsegment.SEG_NEWWORD)]
      results += [wrod for word in seg.Segment(text, libsegment.SEG_BASIC)]
      results += [word for word in get_single_cns(text)]
      return gezi.dedupe_list(results)

    def Segment_nodupe(self, text):
      return self.segment(text)

    #@TODO is this best method ?
    def segment(self, text):
      results = [word for word in get_single_cns(text)]
      results_set = set(results) 
      
      for word in seg.Segment(text):
        if word not in results_set:
          results.append(word)
          results_set.add(word)
      
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        if word not in results_set:
          results.append(word)
          results_set.add(word)
      
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        if word.isdigit():
          word = '<NUM>'
        if word not in results_set:
          results.append(word)
          results_set.add(word)

      return results

    def segment_seq_all(self, text):
      results = [word for word in get_single_cns(text)]

      results.append('<SEP0>')
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        results.append(word)
       
      results.append('<SEP1>')
      for word in seg.Segment(text):
        results.append(word)
      
      results.append('<SEP2>')
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        results.append(word)
      
      return results
    
    def segment_phrase(self, text):
      return seg.Segment(text)

    def segment_basic(self, text):
      return seg.Segment(text, libsegment.SEG_BASIC)

    def segment_phrase_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text)]
      return results

    def segment_phrase_single_all(self, text):
      results = [word for word in segment_char(text, cn_only=False)]
      results += [word for word in seg.Segment(text)]
      return results  

    def segment_basic_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text, libsegment.SEG_BASIC)]
      return results

    def segment_basic_single_all(self, text):
      results = [word for word in segment_char(text, cn_only=False)]
      results += [word for word in seg.Segment(text, libsegment.SEG_BASIC)]
      return results  

    def segment_phrase_single_all(self, text):
      results = [word for word in segment_char(text, cn_only=False)]
      results += [word for word in seg.Segment(text)]
      return results  

    def segment_merge_newword_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text, libsegment.SEG_MERGE_NEWWORD)]
      return results
    
    def Segment(self, text, method='default'):
      """
      default means all level combine
      """
      if gezi.env_has('JIEBA_SEG'):
        ori_text = text
        try:
          text = text.decode('utf8').encode('gbk')
        except Exception:
          #print('------------jieba cut')
          return JiebaSegmentor().Segment(ori_text, method=method)
      else:
        text = text.decode('utf8').encode('gbk', 'ignore')

      if method == 'default' or method == 'all' or method == 'full':
        l = self.segment(text)
      elif method == 'phrase_single':
        l = self.segment_phrase_single(text)
      elif method == 'phrase_single_all':
        l = self.segment_phrase_single_all(text)
      elif method == 'phrase':
        l = seg.Segment(text)
      elif method == 'basic':
        l = seg.Segment(text, libsegment.SEG_BASIC)
      elif method == 'basic_digit':
        words = seg.Segment(text, libsegment.SEG_BASIC)
        def sep_digits(word):
          l = []
          s = ''
          for c in word:
            if c.isdigit():
              if s:
                l.append(s)
              l.append(c)
              s = ''
            else:
              s += c
          if s:
            l.append(s)
          return l
        l = []
        for w in words:
          l += sep_digits(w)
        words = l
      elif method == 'basic_single':
        l = self.segment_basic_single(text)
      elif method == 'basic_single_all':
        l = self.segment_basic_single_all(text)
      elif method == 'phrase_single_all':
        l = self.segment_phrase_single_all(text)
      elif method == 'merge_newword':
        l = seg.Segment(text, libsegment.SEG_MERGE_NEWWORD)
      elif method == 'merge_newword_single':
        l = self.segment_merge_newword_single(text)
      elif method == 'seq_all':
        l = self.segment_seq_all(text)
      elif method == 'en':
        l = segment_en(text)
      elif method == 'tokenize':
        l = tokenize(text)
      elif method == 'char':
        l = segment_char(text)
      elif method == 'tab':
        l = text.strip().split('\t')
      elif method == 'white_space':
        l = text.strip().split()
      else:
        raise ValueError('%s not supported'%method)

      return [x.decode('gbk').encode('utf8') for x in l]


  Segmentor = BSegmentor


import threading
#looks like by this way... threads create delete too much cost @TODO
#and how to prevent this log?
#DEBUG: 08-27 13:18:50:   * 0 [seg_init_by_model]: set use ne=0
#TRACE: 08-27 13:18:50:   * 0 [clear]: tag init stat error
#DEBUG: 08-27 13:18:50:   * 0 [init_by_model]: max_nbest=1, max_word_num=100, max_word_len=100, max_y_size=6
#2>/dev/null 2>&1
#so multi thread only good for create 12 threads each do many works at parrallel, here do so little work.. slow!
def segments(texts, segmentor):
  results = [None] * len(texts)
  def segment_(i, text):
    seg.Init()
    results[i] = segmentor.segment(text)
  threads = []
  for args in enumerate(texts):
    t = threading.Thread(target=segment_, args=args) 
    threads.append(t) 
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  return results

#seems multiprocessing same as above ?
#segmentor resource only init once! whyï¼? @TODO
import multiprocessing
from multiprocessing import Manager 
def segments_multiprocess(texts, segmentor):
  manager = Manager()
  dict_ = manager.dict()
  results = [None] * len(texts)

  def segment_(i, text):
    seg.Init()
    dict_[i] = segmentor.segment(text)
  record = []
  for args in enumerate(texts):
    process =  multiprocessing.Process(target=segment_, args=args) 
    process.start()
    record.append(process) 
  for process in record:
    process.join()
  for i in xrange(len(record)):
    results[i] = dict_[i]
  return results
