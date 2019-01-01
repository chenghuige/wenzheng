#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random

import melt
logging = melt.logging

import lele

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

class CudnnRnn(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    for hkust tf rent, reccruent dropout, bw drop out, concat layers and rnn padding
    WELL TOO SLOW 0.3hour/per epoch -> 1.5 hour..
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False,  
                 recurrent_dropout=True,
                 bw_dropout=False,
                 padding=False):
        super(CudnnRnn, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.recurrent_dropout = recurrent_dropout
        self.bw_dropout = bw_dropout
        self.fws = nn.ModuleList()
        self.bws = nn.ModuleList()
        
        if type(rnn_type) == str:
            rnn_type=nn.GRU if rnn_type == 'gru' else nn.LSTM

        self.num_units = []
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.num_units.append(input_size)
            self.fws.append(rnn_type(input_size, hidden_size,
                                     num_layers=1,
                                     bidirectional=False))
            self.bws.append(rnn_type(input_size, hidden_size,
                                     num_layers=1,
                                     bidirectional=False))

    def forward(self, x, x_mask, fw_masks=None, bw_masks=None):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        # Transpose batch and sequence dims
        batch_size = x.size(0)
        x = x.transpose(0, 1)

        length = (1 - x_mask).sum(1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            mask = None
            if self.dropout_rate > 0:
                if not self.recurrent_dropout:              
                    rnn_input_ = F.dropout(rnn_input,
                                           p=self.dropout_rate,
                                           training=self.training)
                else:
                    if fw_masks is None:
                        mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                                         p=self.dropout_rate,
                                         training=self.training)
                    else:
                        mask = fw_masks[i]
                    rnn_input_ = rnn_input * mask
                
           
            # Forward
            fw_output = self.fws[i](rnn_input_)[0]

            mask = None
            if self.dropout_rate > 0 and self.bw_dropout:
                if not self.recurrent_dropout:
                    rnn_input_ = F.dropout(rnn_input,
                                            p=self.dropout_rate,
                                            training=self.training)
                else:
                    if bw_masks is None:
                        mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                                                    p=self.dropout_rate,
                                                    training=self.training)
                    else:
                        mask = bw_masks[i]
                    rnn_input_ = rnn_input * mask

            rnn_input_ = lele.reverse_padded_sequence(rnn_input_, length)

            bw_output = self.bws[i](rnn_input_)[0]
            
            bw_output = lele.reverse_padded_sequence(bw_output, length)

            outputs.append(torch.cat([fw_output, bw_output], 2))

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output.contiguous()



# TODO support recurrent dropout! to be similar as tf version, support shared dropout
class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, recurrent_dropout=False,
                 dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        self.recurrent_dropout = recurrent_dropout
        if type(rnn_type) is str:
            rnn_type=nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.num_units = []
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.num_units.append(input_size)
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        batch_size = x.size(1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                if not self.recurrent_dropout:
                    rnn_input = F.dropout(rnn_input,
                                        p=self.dropout_rate,
                                        training=self.training)
                else:
                    mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                                     p=self.dropout_rate,
                                     training=self.training)
                    rnn_input = rnn_input * mask

            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        #chg I think do not need padded if you not want to use last state, last output
        """
        # Compute sorted sequence lengths
        #lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        # TODO seems not need squeeze wich might cause error when batch_size is 1
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        batch_size = x.size(1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                if not self.recurrent_dropout:
                    dropout_input = F.dropout(rnn_input.data,
                                            p=self.dropout_rate,
                                            training=self.training)
                else:
                    mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                                     p=self.dropout_rate,
                                     training=self.training)
                    dropout_input = rnn_input.data * mask
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output



class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj
            

class PointerNetwork(nn.Module):
    def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, cell_type=nn.GRUCell, normalize=True):
        super(PointerNetwork, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.linear = nn.Linear(x_size+y_size, hidden_size, bias=False)
        self.weights = nn.Linear(hidden_size, 1, bias=False)
        self.self_attn = NonLinearSeqAttn(y_size, hidden_size)
        self.cell = cell_type(x_size, y_size)

    def init_hiddens(self, y, y_mask):
        attn = self.self_attn(y, y_mask)
        res = attn.unsqueeze(1).bmm(y).squeeze(1) # [B, I]
        return res
    
    def pointer(self, x, state, x_mask):
        x_ = torch.cat([x, state.unsqueeze(1).repeat(1,x.size(1),1)], 2)
        s0 = torch.tanh(self.linear(x_))
        s = self.weights(s0).view(x.size(0), x.size(1))
        s.data.masked_fill_(x_mask.data, -float('inf'))
        a = F.softmax(s)
        res = a.unsqueeze(1).bmm(x).squeeze(1)
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                scores = F.log_softmax(s)
            else:
                # ...Otherwise 0-1 probabilities
                scores = F.softmax(s)
        else:
            scores = a.exp()
        return res, scores


    def forward(self, x, y, x_mask, y_mask):
        hiddens = self.init_hiddens(y, y_mask)
        c, start_scores = self.pointer(x, hiddens, x_mask)
        c_ = F.dropout(c, p=self.dropout_rate, training=self.training)
        hiddens = self.cell(c_, hiddens)
        c, end_scores = self.pointer(x, hiddens, x_mask)
        return start_scores, end_scores

class MemoryAnsPointer(nn.Module):
    def __init__(self, x_size, y_size, hidden_size, hop=1, dropout_rate=0, normalize=True):
        super(MemoryAnsPointer, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.hop = hop
        self.dropout_rate = dropout_rate
        self.FFNs_start = nn.ModuleList()
        self.SFUs_start = nn.ModuleList()
        self.FFNs_end = nn.ModuleList()
        self.SFUs_end = nn.ModuleList()
        for i in range(self.hop):
            self.FFNs_start.append(FeedForwardNetwork(x_size+y_size+2*hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_start.append(SFU(y_size, 2*hidden_size))
            self.FFNs_end.append(FeedForwardNetwork(x_size+y_size+2*hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_end.append(SFU(y_size, 2*hidden_size))
    
    def forward(self, x, y, x_mask, y_mask):
        z_s = y[:,-1,:].unsqueeze(1) # [B, 1, I]
        z_e = None
        s = None
        e = None
        p_s = None
        p_e = None
        
        for i in range(self.hop):
            z_s_ = z_s.repeat(1,x.size(1),1) # [B, S, I]
            s = self.FFNs_start[i](torch.cat([x, z_s_, x*z_s_], 2)).squeeze(2)
            s.data.masked_fill_(x_mask.data, -float('inf'))
            p_s = F.softmax(s, dim=1) # [B, S]
            u_s = p_s.unsqueeze(1).bmm(x) # [B, 1, I]
            z_e = self.SFUs_start[i](z_s, u_s) # [B, 1, I]
            z_e_ = z_e.repeat(1,x.size(1),1) # [B, S, I]
            e = self.FFNs_end[i](torch.cat([x, z_e_, x*z_e_], 2)).squeeze(2)
            e.data.masked_fill_(x_mask.data, -float('inf'))
            p_e = F.softmax(e, dim=1) # [B, S]
            u_e = p_e.unsqueeze(1).bmm(x) # [B, 1, I]
            z_s = self.SFUs_end[i](z_e, u_e)
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                p_s = F.log_softmax(s, dim=1) # [B, S]
                p_e = F.log_softmax(e, dim=1) # [B, S]
            else:
                # ...Otherwise 0-1 probabilities
                p_s = F.softmax(s, dim=1) # [B, S]
                p_e = F.softmax(e, dim=1) # [B, S]
        else:
            p_s = s.exp()
            p_e = e.exp()
        return p_s, p_e


# ------------------------------------------------------------------------------
# Attentions
# ------------------------------------------------------------------------------

class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq

# like tf version DotAttention is rnet
class DotAttention(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, 
                 input_size,  
                 input_size2,
                 hidden,
                 dropout_rate=0.,
                 combiner='gate'):
        super(DotAttention, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(input_size2, hidden)

        self.hidden = hidden
        self.combiner = combiner
        if combiner is not None:
            if combiner == 'gate': 
                self.combine = Gate(input_size + input_size2, dropout_rate=dropout_rate)
                self.output_size = input_size + input_size2
            elif combiner == 'sfu':
                if input_size != input_size2:
                    self.sfu_linear = nn.Linear(input_size2, input_size)
                else:
                    self.sfu_linear = None
                self.combine = SFU(input_size, 3 * input_size, dropout_rate=dropout_rate)
                self.output_size = input_size
            else:
                raise ValueError(f'not support {combiner}')

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        x_proj = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1), self.hidden)
        x_proj = F.relu(x_proj)
        y_proj = self.linear2(y.view(-1, y.size(2))).view(y.size(0), y.size(1), self.hidden)
        y_proj = F.relu(y_proj)

        #print(x_proj.shape, y_proj.shape)
        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        #print(scores.shape)

        #print(y_mask.shape)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(y)

        if self.combiner is None:
            return matched_seq
        else:
            # TODO FIXME... why is False == True  ???
            #print(self.combiner, self.combiner is 'sfu', self.combiner == 'sfu')
            if self.combiner == 'gate':
                return self.combine(torch.cat([x, matched_seq], 2))
            elif self.combiner == 'sfu':
                if self.sfu_linear is not None:
                    matched_seq = self.sfu_linear(matched_seq)

                return self.combine(x,  torch.cat([matched_seq, x * matched_seq, x - matched_seq], 2))

class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diag = diag

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
        else:
            x_proj = x

        # Compute scores
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        if not self.diag:
            x_len = x.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0

        # Mask padding
        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)

        #print(scores.shape, ','.join([str(x) for x in alpha.view(-1).detach().cpu().numpy()]), sep='\t')

        # Take weighted average
        matched_seq = alpha.bmm(x)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class NonLinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, hidden_size=128):
        super(NonLinearSeqAttn, self).__init__()
        self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class MaxPooling(nn.Module):
    def forward(self, x, x_mask):
        if x_mask is None or x_mask.data.sum() == 0:
            return torch.max(x, 1)[0]
        else:
            # x_mask = x_mask.unsqueeze(-1).expand(x.size())
            # x.data.masked_fill_(x_mask.data, -float('inf'))
            # return torch.max(x, 1)[0]
            # # https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
            # # shold be same as above or use F.adaptive_max_pool1d and slightly slower
            lengths = (1 - x_mask).sum(1)
            return torch.cat([torch.max(i[:l], dim=0)[0].view(1,-1) for i,l in zip(x, lengths)], dim=0) 

class TopKPooling(nn.Module):
    def __init__(self, top_k=2):
        super(TopKPooling, self).__init__()
        self.top_k = top_k

    def forward(self, x, x_mask):
        if x_mask is None or x_mask.data.sum() == 0:
            return torch.topk(x, k=self.top_k, dim=1)[0].view(x.size(0), -1)
        else:
            # x_mask = x_mask.unsqueeze(-1).expand(x.size())
            # x.data.masked_fill_(x_mask.data, -float('inf'))
            # return torch.topk(x, k=self.top_k, dim=1)[0].view(x.size(0), -1)
            #FIXME below not ok
            #return F.adaptive_max_pool1d(x, self.top_k).view(x.size(0), -1)
            # https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
            lengths = (1 - x_mask).sum(1)
            return torch.cat([torch.topk(i[:l], k=self.top_k, dim=0)[0].view(1,-1) for i,l in zip(x, lengths)], dim=0).view(x.size(0), -1)

class LastPooling(nn.Module):
    def __init__(self):
        super(LastPooling, self).__init__()

    def forward(self, x, x_mask):
        if x_mask is None or x_mask.data.sum() == 0:
            return x[:,-1,:]
        else:
            lengths = (1 - x_mask).sum(1)
            return torch.cat([i[l - 1,:] for i,l in zip(x, lengths)], dim=0).view(x.size(0), -1)

class LinearSeqAttnPooling(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttnPooling, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        # TODO why need contiguous
        x = x.contiguous() 
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        self.alpha = alpha
        return alpha.unsqueeze(1).bmm(x).squeeze(1)

class LinearSeqAttnPoolings(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, num_poolings):
        super(LinearSeqAttnPoolings, self).__init__()
        self.num_poolings = num_poolings
        self.linear = nn.Linear(input_size, num_poolings)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        # TODO why need contiguous
        x_mask = lele.tile(x_mask.unsqueeze(-1), -1, self.num_poolings)
        x = x.contiguous() 
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1), self.num_poolings)
        scores = scores.transpose(-2, -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        self.alpha = alpha
        return alpha.bmm(x)

class NonLinearSeqAttnPooling(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, hidden_size=128):
        super(NonLinearSeqAttnPooling, self).__init__()
        self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        self.alpha = alpha
        return alpha.unsqueeze(1).bmm(x).squeeze(1)

class NonLinearSeqAttnPoolings(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, num_poolings, hidden_size=128):
        super(NonLinearSeqAttnPoolings, self).__init__()
        self.num_poolings = num_poolings
        self.FFN = FeedForwardNetwork(input_size, hidden_size, num_poolings)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x)
        x_mask = lele.tile(x_mask.unsqueeze(-1), -1, self.num_poolings)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        scores = scores.transpose(-2, -1)
        alpha = F.softmax(scores, dim=-1)
        self.alpha = alpha
        return alpha.bmm(x)

class Pooling(nn.Module):
  def __init__(self,  
               name,
               input_size=0,
               top_k=2,
               att_activation=F.relu,
               **kwargs):
    super(Pooling, self).__init__(**kwargs)

    self.top_k = top_k

    self.poolings = nn.ModuleList()
    def get_pooling(name):
      if name == 'max':
        return MaxPooling()
      elif name == 'mean':
        return MeanPooling()
      elif name == 'attention' or name == 'att':
        return NonLinearSeqAttnPooling(input_size)
      elif name == 'linear_attention' or name == 'linear_att' or name == 'latt':
        return LinearSeqAttnPooling(input_size)
      elif name == 'topk' or name == 'top_k':
        return TopKPooling(top_k=top_k)
      elif name =='first':
        return FirstPooling()
      elif name == 'last':
        return LastPooling()
      else:
        raise f'Unsupport pooling now:{name}'

    self.output_size = 0
    self.names = name.split(',')
    for name in self.names:
      self.poolings.append(get_pooling(name))
      if name == 'topk' or name == 'top_k':
        self.output_size += input_size * top_k
      else:
        self.output_size += input_size

    #logging.info('poolings:', self.poolings)
  
  def forward(self, x, mask=None, calc_word_scores=False):
    results = []
    self.word_scores = []
    for i, pooling in enumerate(self.poolings):
      results.append(pooling(x, mask))
      if calc_word_scores:
        self.word_scores.append(melt.get_words_importance(outputs, sequence_length, top_k=self.top_k, method=self.names[i]))
    
    return torch.cat(results, -1)

class Poolings(nn.Module):
  def __init__(self,  
               name,
               input_size=0,
               num_poolings=1,
               top_k=2,
               att_activation=F.relu,
               **kwargs):
    super(Poolings, self).__init__(**kwargs)

    self.top_k = top_k
    self.num_poolings = num_poolings

    self.poolings = nn.ModuleList()
    self.is_poolings_list = []
    def get_pooling(name):
      if name == 'max':
        # TODO actually only support attention based..
        return MaxPooling(), False
      elif name == 'mean':
        return MeanPooling(), False
      elif name == 'attention' or name == 'att':
        return NonLinearSeqAttnPoolings(input_size, num_poolings), True
      elif name == 'linear_attention' or name == 'linear_att' or name == 'latt':
        return LinearSeqAttnPoolings(input_size, num_poolings), True
      elif name == 'topk' or name == 'top_k':
        return TopKPooling(top_k=top_k), False
      elif name =='first':
        return FirstPooling(), False
      elif name == 'last':
        return LastPooling(), False
      else:
        raise f'Unsupport pooling now:{name}'

    self.output_size = 0
    self.names = name.split(',')
    for name in self.names:
      pooling, is_poolings = get_pooling(name)
      self.poolings.append(pooling) 
      self.is_poolings_list.append(is_poolings)
      if name == 'topk' or name == 'top_k':
        self.output_size += input_size * top_k
      else:
        self.output_size += input_size

    #logging.info('poolings:', self.poolings)
  
  def forward(self, x, mask=None, calc_word_scores=False):
    results = []
    self.word_scores = []
    for i, pooling in enumerate(self.poolings):
      result = pooling(x, mask)
      if not self.is_poolings_list[i]:
          result = lele.tile(result.unsqueeze(1), 1, self.num_poolings)
      results.append(result)
      if calc_word_scores:
        self.word_scores.append(melt.get_words_importance(outputs, sequence_length, top_k=self.top_k, method=self.names[i]))

    result = torch.cat(results, -1)
    self.encode = result
    return result

# TODO can we do multiple exclusive linear simultaneously ?
class Linears(nn.Module):
  def __init__(self,  
               input_size,
               output_size,
               num,
               **kwargs):
    super(Linears, self).__init__(**kwargs)
    self.num = num
    self.linears = lele.clones(nn.Linear(input_size, output_size), num)

  def forward(self, x):
    inputs = x.split(1, 1)
    results = []
    for linear, x in zip(self.linears, inputs):
      result = linear(x)
      results.append(result)
    result = torch.cat(results, 1)
    return result
    

# ------------------------------------------------------------------------------
# Functional Units
# ------------------------------------------------------------------------------

class Gate(nn.Module):
    """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """
    def __init__(self, input_size, dropout_rate=0.):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            res: batch * len * dim
        """
        if self.dropout_rate:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_proj = self.linear(x)
        gate = torch.sigmoid(x)
        return x_proj * gate


class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size, dropout_rate=0.):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)
        self.dropout_rate = dropout_rate

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        if self.dropout_rate:
            r_f = F.dropout(r_f, p=self.dropout_rate, training=self.training)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o

class SFUCombiner(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size, dropout_rate=0.):
        super(SFUCombiner, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)
        self.dropout_rate = dropout_rate

    def forward(self, x, y):
        r_f = torch.cat([x, y, x * y, x - y], 2)
        if self.dropout_rate:
            r_f = F.dropout(r_f, p=self.dropout_rate, training=self.training)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o
        

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
