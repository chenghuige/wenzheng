# -*- coding: utf-8 -*-
import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

from algos.config import NUM_CLASSES

import wenzheng
from wenzheng.utils import vocabulary

import melt
logging = melt.logging

import numpy as np
import lele

class Bow(nn.Module):
  def __init__(self):
    super(Bow, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)


    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))
    self.encode = nn.GRU(input_size=emb_dim, hidden_size=self.num_units, batch_first=True, bidirectional=True)

    #self.logits = nn.Linear(2 * self.num_units, NUM_CLASSES)
    self.logits = nn.Linear(emb_dim, NUM_CLASSES)

  def forward(self, input, training=False):
    x = input['rcontent'] if FLAGS.rcontent else input['content']
    #print(x.shape)

    x = self.embedding(x)
    
    x = torch.max(x, 1)[0]
    
    x = self.logits(x)    

    return x

class Gru(nn.Module):
  def __init__(self):
    super(Gru, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))
    
    #self.encode = nn.GRU(input_size=emb_dim, hidden_size=self.num_units, batch_first=True, bidirectional=True)
    self.encode = lele.layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=1 - FLAGS.keep_prob,
            dropout_output=False,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=FLAGS.rnn_padding,
        )    
    ## Support mask
    #self.pooling = lele.layers.MaxPooling() 

    self.pooling = lele.layers.Pooling(
                        FLAGS.encoder_output_method, 
                        input_size= 2 * self.num_units,
                        top_k=FLAGS.top_k, 
                        att_activation=getattr(F, FLAGS.att_activation))

    # input dim not as convinient as tf..
    pre_logits_dim = self.pooling.output_size
    
    if FLAGS.use_type:
      pre_logits_dim += 1

    num_types = 2
    if FLAGS.use_type_emb:
      type_emb_dim = 10
      self.type_embedding = nn.Embedding(num_types, type_emb_dim)
      pre_logits_dim += type_emb_dim

    if FLAGS.use_type_rnn:
      self.type_embedding = nn.Embedding(num_types, emb_dim)

    self.logits = nn.Linear(pre_logits_dim, NUM_CLASSES)
    #self.logits = nn.Linear(emb_dim, NUM_CLASSES)

  def forward(self, input, training=False):
    x = input['rcontent'] if FLAGS.rcontent else input['content']
    #print(x.shape)
    x_mask = x.eq(0)
    if not FLAGS.mask_pooling:
      x_mask = torch.zeros_like(x, dtype=torch.uint8)

    x = self.embedding(x)

    if FLAGS.use_type_rnn:
      t = self.type_embedding(input['type']).unsqueeze(1)
      x = torch.cat([t, x], 1)
    
      # TODO by default touch.zeros is cpu..
      x_mask = torch.cat([torch.zeros(x.size(0), 1, dtype=torch.uint8).cuda(), x_mask], 1)
    
    # # prefere to use class over function
    # #x = F.dropout(x,self.drop_out, training=self.training)
    # x = self.dropout(x)
    # #print('training', self.training)

    # x, _ = self.encode(x)

    x = self.encode(x, x_mask)

    
    #x = F.max_pool2d(x, kernel_size=x.size()[2:])
    #x = torch.max(x, 1)[0]

    x = self.pooling(x, x_mask)

    if FLAGS.use_type:
      x = torch.cat([x, input['type'].float().unsqueeze(1)], 1)

    if FLAGS.use_type_emb:
      x = torch.cat([x, self.type_embedding(input['type'])], 1)
    
    x = self.logits(x)    

    return x

# ai challenger baeline 
# https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/opinion_questions_machine_reading_comprehension2018_baseline
class MwAN(nn.Module):
    def __init__(self):
        super(MwAN, self).__init__()
        
        vocabulary.init()
        vocab_size = vocabulary.get_vocab_size()        
        embedding_size = FLAGS.emb_dim
        encoder_size = FLAGS.rnn_hidden_size
        self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))

        self.embedding = wenzheng.pyt.get_embedding(vocab_size, 
                                                emb_dim, 
                                                FLAGS.word_embedding_file, 
                                                FLAGS.finetune_word_embedding)

        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                bidirectional=True)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)

        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        self.gru_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
        """
        prediction layer
        """
        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.logits = nn.Linear(3, 3)
        self.logits2 = nn.Linear(3, 3)
        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, x, training=False):
        query = x['query']
        passage = x['passage']
        candidate_neg = x['candidate_neg']
        candidate_pos = x['candidate_pos']
        candidate_na = x['candidate_na']

        #print(passage.shape)
        
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        
        neg_embeddings = self.embedding(candidate_neg)
        pos_embeddings = self.embedding(candidate_pos)
        na_embeddings = self.embedding(candidate_na)

        neg_embedding, _ = self.a_encoder(neg_embeddings)
        pos_embedding, _ = self.a_encoder(pos_embeddings)
        na_embedding, _ = self.a_encoder(na_embeddings)

        neg_score = F.softmax(self.a_attention(neg_embedding), 1)
        neg_output = neg_score.transpose(2, 1).bmm(neg_embedding).squeeze()

        pos_score = F.softmax(self.a_attention(pos_embedding), 1)
        pos_output = pos_score.transpose(2, 1).bmm(pos_embedding).squeeze()

        na_score = F.softmax(self.a_attention(na_embedding), 1)
        na_output = na_score.transpose(2, 1).bmm(na_embedding).squeeze()

        a_embedding = torch.stack([neg_output, pos_output, na_output], dim=1)

        hq, _ = self.q_encoder(p_embedding)
        hq=self.dropout(hq)
        hp, _ = self.p_encoder(q_embedding)
        p=self.dropout(hp)
        _s1 = self.Wc1(hq).unsqueeze(1)
        _s2 = self.Wc2(hp).unsqueeze(2)
        # squeeze might cause batch size None TODO
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(hq)
        _s1 = self.Wb(hq).transpose(2, 1)
        sjt = hp.bmm(_s1)
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(hq)
        _s1 = hq.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(hq)
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(hq)
        _s1 = hp.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hp)
        aggregation = torch.cat([hp, qts, qtc, qtd, qtb, qtm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)
        sj = self.vq(torch.tanh(self.Wq(hq))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(hq)
        sj = F.softmax(self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(2, 1), 2)
        rp = sj.bmm(aggregation_representation)
        encoder_output = self.dropout(F.leaky_relu(self.prediction(rp)))
        logits = a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze()
        #logits = self.logits(logits)
        return logits

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  loss_fn = torch.nn.CrossEntropyLoss()

  # Do not need one hot
  #y = torch.zeros(y.size(0), NUM_CLASSES, dtype=torch.int64).scatter_(1, y.view(y.size(0), 1), 1)
  loss = loss_fn(y_, y)
  
  return loss


