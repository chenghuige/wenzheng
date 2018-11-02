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
import os 

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
  import jieba.posseg
except Exception:
  pass 

segment_char = segment_utf8_char
segment_pinyin = segment_utf8_pinyin

stanford_nlp = None

# NOTICE emoji will cause segment error for stanford nlp
# https://stackoverflow.com/questions/43146528/how-to-extract-all-the-emojis-from-text
def init_stanford_nlp(path='/home/gezi/soft/stanford-corenlp', lang='zh'):
#def init_stanford_nlp(path='/stanford-corenlp', lang='zh'):
  global stanford_nlp
  if not stanford_nlp:
    from stanfordcorenlp import StanfordCoreNLP 
    if not os.path.exists(path):
      path = './stanford-corenlp'
      if not os.path.exists(path):
        print('init stanford nlp from local server 8818 port', file=sys.stderr)
        stanford_nlp = StanfordCoreNLP('http://localhost', port=8818, lang=lang, memory='8g')
    if not stanford_nlp:    
      import logging
      print('init stanford nlp from %s' % path, file=sys.stderr)
      stanford_nlp = StanfordCoreNLP(path, lang=lang, memory='4g')

def remove_duplicate(text):
  if six.PY2:
    text = text.decode('utf8')
  l = []

  start = 0 
  end = 0
  duplicates = []
  while start < len(text):
    while  end < len(text) and text[start] == text[end]:
      duplicates.append(text[start])
      end += 1
    l.append(''.join(duplicates[:5]))
    duplicates = []
    start = end
  
  text = ''.join(l)

  if six.PY2:
    text = text.encode('utf8')
  return text

def cut(text, type='word'):
  # move it to filter.py
  #text = remove_duplicate(text) 
  if type == 'word':
    return word_cut(text)
  elif type == 'pos':
    return pos_cut(text)
  elif type == 'ner':
    return ner_cut(text)
  else:
    raise ValueError('bad type: %s' % type)

# HACK for emoji stanford nlp

def is_emoji_en(word):
  for w in word:
    if not (w == '_' or (w <= 'z' and w >= 'a')):
      return False
  return True

def hack_emoji(l):
  import emoji
  res = []
  i = 0
  while i < len(l):
    if i < len(l) - 2 and (l[i] == ':' and l[i+2] == ':' and  is_emoji_en(l[i + 1])):
      res.append(emoji.emojize(':%s:' % l[i + 1]))
      i += 2
    else:
      res.append(l[i])
      i += 1

  return res

def hack_emoji2(l):
  import emoji
  res = []
  i = 0
  while i < len(l):
    if i < len(l) - 2:
      x, _ = l[i]
      x2, _ = l[i + 1]
      x3, _ = l[i + 2]
      if x == ':' and x3 == ':' and  is_emoji_en(x2):
        res.append((emoji.emojize(':%s:' % x2), 'EMOJI'))
        i += 3
      else:
        res.append(l[i])
        i += 1
    else:
      res.append(l[i])
      i += 1

  return res

expressions_len2 = set([':)', ':(',])
expressions_len3 = set(['^_^', '^O^', '≥▽≤', '+_+',\
                        '＾◇＾', '～o～', '-ω-', '-_-', '^ω^',\
                        '^o^', '╯ε╰', '∩_∩', '^O^', '=_=', 'T_T'])
expressions_len4 =  set(['-_-#'])

expressions = [expressions_len2, expressions_len3, expressions_len4]

def merge_expression(l):
  res = []

  i = 0
  while i < len(l):
    if i + 3 < len(l) and ''.join(l[i:i+4]) in expressions_len4:
      res.append(''.join(l[i:i+4]))
      i = i + 4
    elif i + 2 < len(l) and ''.join(l[i:i + 3]) in expressions_len3:
      res.append(''.join(l[i:i + 3]))
      i += 3
    elif i + 1 < len(l) and ''.join(l[i:i + 2]) in expressions_len2:
      res.append(''.join(l[i:i + 2]))
      i += 2
    else:
      res.append(l[i])
      i += 1
  return res 

def merge_expression2(l):
  res = []
  i = 0
  while i < len(l):
    if i + 3 < len(l) and ''.join([x for x,y in l[i:i+4]]) in expressions_len4:
      res.append((''.join([x for x,y in l[i:i+4]]), 'EXPRESSION'))
      i += 4
    elif i + 2 < len(l) and ''.join([x for x,y in l[i:i + 3]]) in expressions_len3:
      res.append((''.join([x for x,y in l[i:i + 3]]), 'EXPRESSION'))
      i += 3
    elif i + 1 < len(l) and ''.join([x for x,y in l[i:i + 2]]) in expressions_len2:
      res.append((''.join([x for x,y in l[i:i + 2]]), 'EXPRESSION'))
      i += 2
    else:
      res.append(l[i])
      i += 1
  return res


bseg = None 
def init_bseg(use_pos=False, use_ner=False):
  global bseg
  if bseg is None: 
    assert six.PY2, 'bseg must use python2'
    import nowarning 
    import libgezi 
    import libsegment 
    from libsegment import Segmentor 
    if use_pos:
      Segmentor.AddStrategy(libsegment.SEG_USE_POSTAG)
    if use_ner:
      Segmentor.AddStrategy(libsegment.SEG_USE_WORDNER)
    #Segmentor.Init(ner_maxterm_count=1024)
    Segmentor.Init()
    bseg = Segmentor
  return bseg

def to_gbk(text):
  return text.decode('utf8', 'ignore').encode('gbk', 'ignore')

def to_utf8(text):
  return text.decode('gbk', 'ignore').encode('utf8', 'ignore')

# sentence piece
sp = None
def init_sp(path=None):
  import sentencepiece as spm 
  global sp
  if sp is None: 
    sp = spm.SentencePieceProcessor()
    if path is None:
      path = './sp.model'
    sp.Load(path)

  assert sp

def word_cut(text):
  if gezi.env_has('SENTENCE_PIECE'):
    init_sp()
    l = sp.EncodeAsPieces(text)
    if l:
      if l[0] == '▁':
        l =  l[1:]
      elif l[0].startswith('▁'):
        l[0] = l[0][1:]
    return l

  if gezi.env_has('STANFORD_NLP'):
    import emoji
    init_stanford_nlp()
    try:
      l = stanford_nlp.word_tokenize(emoji.demojize(text))
      l = hack_emoji(l)
      return merge_expression(l)
    except Exception:
      print('stnaord error text: %s' % text, file=sys.stderr)
      l = list(jieba.cut(text))
      return merge_expression(l)

  if gezi.env_has('BSEG'):
    import emoji
    init_bseg()
    try:
      # NOTICE py2 need decode utf8 to get unicode as input to emoji
      l = bseg.Segment(to_gbk(emoji.demojize(text.decode('utf8'))))
      l = [to_utf8(x) for x in l]
      l = hack_emoji(l)
      l = merge_expression(l)
      return l
    except Exception:
      print('bseg error text: %s' % text, file=sys.stderr)
      l = list(jieba.cut(text))
      l = merge_expression(l)
      return l

  # TODO make a switch since jieba.posseg is much slower... then jieba.cut
  if gezi.env_has('JIEBA_POS'):
    l = jieba.posseg.cut(text)
    l = [word for word, tag in l]
    return merge_expression(l)
  
  return merge_expression(list(jieba.cut(text)))

pos_tags = ["None", "Ag", "Dg", "Ng", "Tg", "Vg", "a", "ad", "an", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "nr", "ns", "nt", "nx", "nz", "o", "p", "q", "r", "s", "t", "u", "v", "vd", "vn", "w", "y", "z" ]

#def pos_cut(text, same_with_ner=True):
def pos_cut(text):
  #print('------------', text, gezi.env_has('BSEG'))
  if gezi.env_has('STANFORD_NLP'):
    import emoji
    init_stanford_nlp()
    l = stanford_nlp.pos_tag(emoji.demojize(text))
    l = hack_emoji2(l)
    res = merge_expression2(l)
  elif gezi.env_has('BSEG'):
    import emoji
    init_bseg(use_pos=True)
    def bseg_(text):
      nodes = bseg.Cut(to_gbk(emoji.demojize(text.decode('utf8')))) 
      l = [(to_utf8(x.word), pos_tags[x.type]) for x in nodes]
      return l
    l = bseg_(text)

    # if not same_with_ner:
    #   l = bseg_(text)
    # else:
    #   # HACK to have same seg result as word ner
    #   MAX_LEN = 500
    #   text_len = len(text)
    #   if  text_len < MAX_LEN:
    #     l = bseg_(text)
    #   else:
    #     len_ = 0
    #     words = []
    #     l = []
    #     for word in jieba.cut(text):
    #       word = word.encode('utf-8')
    #       len_ += len(word)
    #       words.append(word)
    #       if len_ >= MAX_LEN:
    #         len_ = 0
    #         l += bseg_(''.join(words))
    #         #print(len(''.join(words)), len(l))
    #         words = []
    #     if words:
    #       l += bseg_(''.join(words))

    #assert l
    l = hack_emoji2(l)
    res = merge_expression2(l)
  else:
    res = merge_expression2(list(jieba.posseg.cut(text)))
  
  for i in range(len(res)):
    w, t = res[i]
    if w == '\x01' or w == '\x02' or w == '\x03':
      res[i] = (w, 'sep')
  return res

#x = 100000

def ner_cut(text):
  import emoji
  if gezi.env_has('STANFORD_NLP'):
    init_stanford_nlp()
    l = stanford_nlp.ner(emoji.demojize(text))
    l = merge_expression2(l)
  else:
    init_bseg(use_ner=True)
    def bseg_(text):
      bseg.Cut(to_gbk(emoji.demojize(text.decode('utf8'))))
      bseg.NerTag()
      if not gezi.env_has('BSEG_SUBNER'):
        nodes = bseg.GetNerNodes()
      else:
        nodes = bseg.GetSubNerNodes()
      l = [(to_utf8(x.word), x.name) for x in nodes]
      return l
    # have tested as 718 cause error 
    MAX_LEN = 500
    text_len = len(text)
    #l = bseg_(text)
    #print(text_len, len(l))

    # global x 
    # if len(l) == 0:
    #   if text_len < x:
    #     x = text_len
    #     #print('-------------------', x)

    #print('-----------------', x)
    
    # HACK bseg wordner could nout seg long text so workaround here is to cut it  
    # well still has some fail.. 
    jieba_cuts = None
    if  text_len < MAX_LEN:
      l = bseg_(text)
    else:
      len_ = 0
      words = []
      l = []
      jieba_cuts = [x.encode('utf8') for x in jieba.cut(text)]
      for word in jieba_cuts:
        len_ += len(word)
        words.append(word)
        if len_ >= MAX_LEN:
          len_ = 0
          l += bseg_(''.join(words))
          #print(len(''.join(words)), len(l))
          words = []

      if words:
        l += bseg_(''.join(words))
      
    #if len(l) < len(text) / 10:
    if len(''.join([x for x,y in l]).decode('utf8')) < len(text.decode('utf8')) * 0.8:
      print('warning, bad cut for turn back to use jieba cut:', text, file=sys.stderr)
      if not jieba_cuts:
        jieba_cuts = [x.encode('utf8') for x in jieba.cut(text)]
      l = [(x, 'NOR') for x in jieba_cuts]
     
    #exit(0)
  #assert l
  l = hack_emoji2(l)
  res = merge_expression2(l)

  for i in range(len(res)):
    w, t = res[i]
    if w == '\x01' or w == '\x02' or w == '\x03':
      res[i] = (w, 'sep')
  return res

#TODO hack how to better deal? now for c++ part must be str..
#TODO JiebaSegmentor can be Stanford nlp seg mentor if set STANFORD_NLP in env, so only to diff from BZegmentor
class JiebaSegmentor(object):
  def __init__(self):
    pass

  def segment_basic_single(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=True)]
    results += [word for word in cut(text)]
    return results  

  def segment_basic_single_all(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=False)]
    results += [word for word in cut(text)]
    return results  

  def segment_full_single(self, text):
    #results = [word for word in get_single_cns(text)]
    results = [word for word in segment_char(text, cn_only=True)]
    results += [word for word in cut_for_search(text)]
    return results  

  def Segment(self, text, method='basic'):
    """
    default means all level combine
    """
    method = method.replace('phrase', 'basic')

    words = None
    if method == 'default' or method == 'basic' or method == 'exact':
      #words = [x for x in cut(text)]
      words = cut(text)
    elif method == 'basic_digit':
      #words = [x for x in cut(text)]
      words = cut(text)
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
      words = cut_for_search(text)
    elif method == 'cut_all':
      words = cut(text, cut_all=True)
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
      return [x for x in cut(text, cut_all=False)] + ['<S>'] + segment_char(text)
    elif method == 'word_char_pinyin':
      from pypinyin import lazy_pinyin as pinyin
      return [x for x in cut(text, cut_all=False)] + ['<S>'] + segment_char(text) + ['<S>'] + [x.strip() for x in pinyin(text)]
    elif method == 'word_char_pinyin2':
      from pypinyin import lazy_pinyin as pinyin
      return [x for x in cut(text, cut_all=False)] + ['<S>'] + segment_char(text) + ['<S>'] + [''.join(pinyin(x)).strip() for x in text if x.strip()]
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
