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

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 

    emb_dim = FLAGS.emb_dim 

    self.embedding = nn.Embedding(vocab_size, embedding_dim=emb_dim)

    self.num_layers = FLAGS.num_layers
    self.num_units = FLAGS.rnn_hidden_size
    self.dropout = nn.Dropout(p=(1 - FLAGS.keep_prob))
    self.encode = nn.GRU(input_size=emb_dim, hidden_size=self.num_units, batch_first=True, bidirectional=True)

    self.logits = nn.Linear(2 * self.num_units, NUM_CLASSES, bias=True)
    #self.logits = nn.Linear(emb_dim, NUM_CLASSES, bias=True)

  def forward(self, input, training=False):
    x = input['rcontent'] if FLAGS.rcontent else input['content']
    #print(x.shape)

    x = self.embedding(x)
    
    # prefere to use class over function
    #x = F.dropout(x,self.drop_out, training=self.training)
    x = self.dropout(x)
    #print('training', self.training)

    x, _ = self.encode(x)
    
    #x = F.max_pool2d(x, kernel_size=x.size()[2:])
    x = torch.max(x, 1)[0]
    
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
        drop_out = 1 - FLAGS.keep_prob

        self.drop_out=drop_out
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
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
        hq=F.dropout(hq,self.drop_out)
        hp, _ = self.p_encoder(q_embedding)
        p=F.dropout(hp,self.drop_out)
        _s1 = self.Wc1(hq).unsqueeze(1)
        _s2 = self.Wc2(hp).unsqueeze(2)
        #print(_s1.shape, _s2.shape)
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
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)
        logits = a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze()
        return logits

def criterion(model, x, y, training=False):
  y_ = model(x, training=training)
  loss_fn = torch.nn.CrossEntropyLoss()

  # Do not need one hot
  #y = torch.zeros(y.size(0), NUM_CLASSES, dtype=torch.int64).scatter_(1, y.view(y.size(0), 1), 1)
  loss = loss_fn(y_, y)
  
  return loss


