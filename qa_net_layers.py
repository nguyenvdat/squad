"""Assortment of layers for use in models.py.

Author:
    Dat Nguyen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import math
import copy

class CharEmbeddings(nn.Module): 
    """
    Class that converts input words (represented as characters) to their CNN-based embeddings.
    """
    def __init__(self, d_char, char2idx, dropout):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        """
        super(CharEmbeddings, self).__init__()
        pad_token_idx = 0
        self.embeddings = nn.Embedding(len(char2idx), d_char, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        batch_size, sentence_length, max_word_length = input.size()
        output = input.contiguous().view(-1, max_word_length)
        output = self.embeddings(output) # (word_batch_size, max_word_length, d_char)
        output = torch.max(output, 1, keepdim=False)[0] # (word_batch_size, d_char)
        output = output.view(batch_size, sentence_length, -1) # (batch, max_word, d_char)
        output = self.dropout(output)
        return output

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size, dropout):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        for transform, gate in zip(self.transforms, self.gates):
            nn.init.kaiming_normal_(transform.weight)
            nn.init.kaiming_normal_(gate.weight)

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
            x = self.dropout(x)
        return x

class InputEmbedding(nn.Module):
    """Input Embedding layer used by QANet, which combine pretrained GloVe and character level embeddings to form word-level embeddings.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout):
        super(InputEmbedding, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.word_embed.weight.requires_grad = False
        self.char_embed = CharEmbeddings(d_char, char2idx, char_dropout)
        self.hwy = HighwayEncoder(2, hidden_size, highway_dropout)
        self.word_dropout = nn.Dropout(word_dropout)

    def forward(self, word_inputs, char_inputs):
        word_emb = self.word_embed(word_inputs)   # (batch_size, seq_len, word_embed_size)
        word_emb = self.word_dropout(word_emb)
        char_emb = self.char_embed(char_inputs)   # (batch_size, seq_len, char_embed_size)
        emb = torch.cat((word_emb, char_emb), dim=2) # (batch_size, seq_len, word_embed_size + char_embed_size)
        # print(emb.size())
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class NormLayer(nn.Module):
    """ Perform layer normalization with learnable weights
    """
    def __init__(self, input_size, eps=1e-6, dim=-1):
        super(NormLayer, self).__init__()
        input_size = (1, 1, input_size) if dim == -1 else (1, input_size, 1)
        self.g = nn.Parameter(torch.ones(input_size))
        self.b = nn.Parameter(torch.zeros(input_size))
        self.eps = eps  
        self.dim = dim
        nn.init.xavier_uniform_(self.g)
        nn.init.xavier_uniform_(self.b)
    
    def forward(self, x):
        mean = torch.mean(x, self.dim, keepdim=True)
        std = torch.std(x, self.dim, keepdim=True)
        return self.g * (x - mean) / (std + self.eps) + self.b

class ResNormLayer(nn.Module):
    """ for an input x add x to the output of applying consecutively n times f_layer to normalized x
    """
    def __init__(self, input_size, f_layer, n=1, dim=-1):
        super(ResNormLayer, self).__init__()
        self.f_layers = clones(f_layer, n)
        self.norm_layer = NormLayer(input_size, dim=dim)
    
    def forward(self, x, **kwargs):
        old_x = x
        x = self.norm_layer(x)
        for l in self.f_layers:
            if len(kwargs) > 1:
                x = l(x, kwargs['y'], kwargs['mask'])
            elif len(kwargs) > 0:
                x = l(x, kwargs['mask'])
            else:
                x = l(x)
        return x + old_x

class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, d_word, k, d_conv):
        super(DepthwiseSeparableConvLayer, self).__init__()
        self.depthwise_conv = nn.Conv1d(d_word, d_word, k, groups=d_word, padding=k//2,bias=False)  
        self.pointwise_conv = nn.Conv1d(d_word, d_conv, 1, bias=True)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
    
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

class AttentionLayer(nn.Module):
    def __init__(self, d_conv, d_attention):
        super(AttentionLayer, self).__init__()
        assert d_conv % d_attention == 0
        self.d_attention = d_attention
        self.query_linear = nn.Linear(d_conv, d_attention)
        self.key_linear = nn.Linear(d_conv, d_attention)
        self.value_linear = nn.Linear(d_conv, d_attention)
        nn.init.xavier_uniform_(self.query_linear.weight)
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)

    def forward(self, x, y, mask):
        query = self.query_linear(x) # (batch_size, len_sentence_x, d_attention)
        key = self.key_linear(y) # (batch_size, len_sentence_y, d_attention)
        value = self.value_linear(y) # (batch_size, len_sentence_y, d_attention)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt (self.d_attention) # (bs, len_x, len_y)
        if mask is not None:
            mask = torch.unsqueeze(mask, 1)
            # print('mask: {}'.format(mask.size()))
            scores = masked_softmax(scores, mask, -1) # (batch_size, len_sentence_x, len_sentence_y)
        else:
            scores = F.softmax(scores, -1)
        return torch.matmul(scores, value) # (batch_size, len_sentence_x,  d_attention)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_head, d_conv, d_attention):
        super(MultiHeadAttentionLayer, self).__init__()
        attention_layer = AttentionLayer(d_conv, d_attention)
        self.attention_layers = clones(attention_layer, n_head)

    def forward(self, x, y, mask):
        attention_heads = [attention_layer(x, y, mask) for attention_layer in self.attention_layers]
        return torch.cat(attention_heads, 2) # (batch_size, len_sentence_x, d_attention * n_head)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        return F.relu(self.linear(x))

class EncoderBlock(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head):
        super(EncoderBlock, self).__init__()
        self.pos_enc = PositionalEncoder(d_word)
        if d_word != d_conv:
            self.input_conv = nn.Conv1d(d_word, d_conv, 1)
            nn.init.xavier_uniform_(self.input_conv.weight)
            self.input_same_shape = False
        else:
            self.input_same_shape = True
        depthwise_conv = DepthwiseSeparableConvLayer(d_conv, kernel_size, d_conv)
        self.depthwise_conv_layers = clones(ResNormLayer(d_conv, depthwise_conv, dim=-2), n_conv)
        multihead_attention = MultiHeadAttentionLayer(n_head, d_conv, d_attention)
        self.multihead_attention_layer = ResNormLayer(d_conv, multihead_attention)
        feed_forward_layer = FeedForwardLayer(d_attention * n_head, d_out)
        self.feed_forward_layer = ResNormLayer(d_conv, feed_forward_layer)

    def forward(self, x, mask):
        x = self.pos_enc(x)
        x = x.transpose(-1, -2)
        if not self.input_same_shape:
            x = self.input_conv(x)
        for l in self.depthwise_conv_layers:
            x = l(x)
        x = x.transpose(-1, -2)
        x = self.multihead_attention_layer(x, y=x, mask=mask)
        x = self.feed_forward_layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head,   dropout, n_block=1):
        super(EncoderLayer, self).__init__()
        blocks = []
        for i in range(n_block):
            if i == 0:
                encoder_block = EncoderBlock(d_word, d_conv, kernel_size,  d_attention, d_out, n_conv, n_head)
            else:
                encoder_block = EncoderBlock(d_conv, d_conv, kernel_size,  d_attention, d_out, n_conv, n_head)
            blocks.append(encoder_block)
        self.encoder_blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        for i, encoder_block in enumerate(self.encoder_blocks):
            if i == 0:
                x = encoder_block(x, mask)
            else: 
                # x = encoder_block(x, mask=None)
                x = encoder_block(x, mask)
        return self.dropout(x)

class ContextQueryAttentionLayer(nn.Module):
    """Context Query Attention Layer (mainly Stanford cs224 course's code)

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, dropout):
        super(ContextQueryAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.c_weight = nn.Parameter(torch.zeros(input_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(input_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, input_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, input_size) => (bs, c_len, input_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, input_size) => (bs, c_len, input_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * input_size)

        return self.dropout(x)

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query.
        """
        c_len, q_len = c.size(1), q.size(1)
        # c = self.dropout(c)  # (bs, c_len,input_size)
        # q = self.dropout(q)  # (bs, q_len, input_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class ModelEncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout, n_block):
        super(ModelEncoderLayer, self).__init__()
        self.input_conv = nn.Conv1d(d_word, d_conv, 1)
        self.encoder_layer = EncoderLayer(d_conv, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout, n_block)
        nn.init.xavier_uniform_(self.input_conv.weight)

    def forward(self, x, mask):
        x = x.transpose(-1, -2)
        x = self.input_conv(x)
        x = x.transpose(-1, -2)
        m0 = self.encoder_layer(x, mask) # (batch_size, n_context, dout)
        m1 = self.encoder_layer(m0, mask) # (batch_size, n_context, dout)
        m2 = self.encoder_layer(m1, mask) # (batch_size, n_context, dout)
        # return m0, m1, m2
        return m0, m1, m2

class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super(OutputLayer, self).__init__()
        self.linear1 = nn.Linear(input_size * 2, 1)
        self.linear2 = nn.Linear(input_size * 2, 1)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, m0, m1, m2, mask):
        logits_1 = self.linear1(torch.cat((m0, m1), 2)) # (batch_size, n_context, 1)
        logits_2 = self.linear2(torch.cat((m0, m2), 2)) # (batch_size, n_context, 1)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 800):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        # print('pe: {}'.format(self.pe[:, :seq_len].size()))
        return x

class SegmentRecurrentHead(nn.Module):
    def __init__(self, memory_len, seg_len, d_in, d_out, device):
        super(SegmentRecurrentHead, self).__init__()
        M = memory_len
        L = seg_len
        self.R = self.get_relative_position_encoder(d_in, M + 2*L - 1, device) # (1, M + 2*L, d_in)
        self.R_center = M + L - 1
        self.Wq = nn.Linear(d_in, d_out)
        self.Wke = nn.Linear(d_in, d_out)
        self.Wkr = nn.Linear(d_in, d_out)
        self.Wv = nn.Linear(d_in, d_out)
        self.u = nn.Parameter(torch.ones(d_out, 1))
        self.v = nn.Parameter(torch.ones(d_out, 1))
        self.device = device
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wke.weight)
        nn.init.xavier_uniform_(self.Wkr.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, h_prev, h, mask):
        """
        h_prev: (bs, M, d_in)
        h: (bs, L, d_in)
        """
        bs, M, _ = h_prev.size()
        _, L, d_out = h.size()
        h_tile = torch.cat((h_prev.detach(), h), dim=1) # (bs, M + L, d_in)
        q = self.Wq(h) # (bs, L, d_out)
        k = self.Wke(h_tile) # (bs, M + L, d_out)
        value = self.Wv(h_tile) # (bs, M + L, d_out)

        # transformerxl paper p6
        first_term = torch.bmm(q, k.transpose(1, 2)) # (bs, L, M + L)

        third_term  = torch.matmul(k, self.u.repeat(1, L)) # (bs, M + L, L)
        third_term = third_term.transpose(1, 2) # (bs, L, M + L)

        # transformerxl paper p14
        R = self.R[:, self.R_center - M - L + 1: self.R_center + L]
        Q = self.Wkr(R.repeat(bs, 1, 1)) # (bs, M + 2*L - 1, d_out)
        B_tile = torch.bmm(q, Q.transpose(1, 2)) # (bs, L, M + 2*L - 1)
        c = torch.ones((L, M + 2*L - 1))
        c_top_left = torch.triu(torch.ones((L - 1, L - 1))).transpose(1, 0)[range(L - 2, -1, -1)]
        c_low_right = torch.triu(torch.ones((L - 1, L - 1)))[range(L - 2, -1, -1)].transpose(1, 0)
        if c_top_left.size()[0] > 0 and c_low_right.size()[0] > 0:
            c[:L - 1, :L - 1] -= c_top_left
            c[-L + 1:, -L + 1:] -= c_low_right
        c = c.to(self.device)
        B = torch.masked_select(B_tile, c.byte()).view(bs, L, M + L)
        # B = torch.zeros((bs, L, L + M), device=self.device) # (bs, L, M + L)
        # for i in range(L):
        #     B[:, i] = B_tile[:, i, L - i - 1: M + 2 * L - i - 1]
        
        D_tile = torch.matmul(Q, self.v.repeat(1, L)) # (bs, M + 2*L - 1, L)
        D_tile = D_tile.transpose(1, 2) # (bs, L, M + 2*L)
        D = torch.masked_select(D_tile, c.byte()).view(bs, L, M + L)
        # for i in range(L):
        #     D[:, i] = D_tile[:, i, L - i - 1: M + 2 * L - i - 1]
        
        scores = (first_term + B + third_term + D) / math.sqrt(d_out)
        mask = torch.unsqueeze(mask, 1) # (bs, 1, L)
        # print(mask.size())
        scores = masked_softmax(scores, mask, -1) # (bs, L, M + L)
        # print(scores.size())
        return torch.matmul(scores, value) # (bs, L, d_out)

    def get_relative_position_encoder(self, d, max_length, device):
        pe = torch.zeros((max_length, d), device=device)
        for pos in range(max_length):
            for i in range(0, d, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d)))
        pe = pe.unsqueeze(0) # (1, max_length, d)
        return pe

class MultiHeadSegmentRecurrent(nn.Module):
    def __init__(self, memory_len, seg_len, d_in, d_out, n_head, device):
        super(MultiHeadSegmentRecurrent, self).__init__()
        segment_recurrent_head = SegmentRecurrentHead(memory_len, seg_len, d_in, d_out, device)
        self.heads = clones(segment_recurrent_head, n_head)

    def forward(self, h_prev, h, mask):
        sr_heads = [head(h_prev, h, mask) for head in self.heads]
        return torch.cat(sr_heads, 2) # (bs, L, d_out * n_head)

class SegmentRecurrent(nn.Module):
    def __init__(self, memory_len, seg_len, d_in, d_out, n_head, device):
        super(SegmentRecurrent, self).__init__()
        assert memory_len % seg_len == 0
        self.memory_len = memory_len
        self.seg_len = seg_len
        self.d_in = d_in
        self.d_out = d_out
        self.n_head = n_head
        self.multi_heads = MultiHeadSegmentRecurrent(memory_len, seg_len, d_in, d_out, n_head, device)
        self.device = device

    def forward(self, x, mask):
        bs, s_len, _ = x.size()
        start_idx = 0
        out = torch.zeros((bs, s_len, self.d_out * self.n_head), device=self.device)
        h_prev = torch.zeros((bs, self.memory_len, self.d_in), dtype=torch.float32, device=self.device)
        mask_prev = torch.zeros((bs, self.memory_len), dtype=torch.uint8, device=self.device)
        while start_idx + self.seg_len < s_len:
            h = x[:, start_idx: start_idx + self.seg_len]
            mask_current = mask[:, start_idx: start_idx + self.seg_len]
            mask_combine = torch.cat((mask_prev, mask_current), dim=1)
            o = self.multi_heads(h_prev, h, mask_combine) # (bs, L, d_out * n_head)
            out[:, start_idx: start_idx + self.seg_len] = o
            h_prev[:, 0: -self.seg_len] = h_prev[:, self.seg_len:]
            h_prev[:, -self.seg_len:] = h
            mask_prev[:, 0: -self.seg_len] = mask_prev[:, self.seg_len:]
            mask_prev[:, -self.seg_len:] = mask_current
            start_idx += self.seg_len
        # the last segment        
        h = x[:, start_idx:]
        mask_combine = torch.cat((mask_prev, mask[:, start_idx:]), dim=1)
        o = self.multi_heads(h_prev, h, mask_combine)
        out[:, start_idx:] = o 
        return out # (bs, sentence_len, d_out * n_head)

class TXEncoderBlock(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, device):
        super(TXEncoderBlock, self).__init__()
        if d_word != d_conv:
            self.input_conv = nn.Conv1d(d_word, d_conv, 1)
            nn.init.xavier_uniform_(self.input_conv.weight)
            self.input_same_shape = False
        else:
            self.input_same_shape = True
        depthwise_conv = DepthwiseSeparableConvLayer(d_conv, kernel_size, d_conv)
        self.depthwise_conv_layers = clones(ResNormLayer(d_conv, depthwise_conv, dim=-2), n_conv)
        segment_recurrent = SegmentRecurrent(memory_len, seg_len, d_conv, d_attention, n_head, device=device)
        # multihead_attention = MultiHeadAttentionLayer(n_head, d_conv, d_attention)
        self.segment_recurrent_layer = ResNormLayer(d_conv, segment_recurrent)
        feed_forward_layer = FeedForwardLayer(d_attention * n_head, d_out)
        self.feed_forward_layer = ResNormLayer(d_conv, feed_forward_layer)

    def forward(self, x, mask):
        x = x.transpose(-1, -2)
        if not self.input_same_shape:
            x = self.input_conv(x)
        for l in self.depthwise_conv_layers:
            x = l(x)
        x = x.transpose(-1, -2)
        x = self.segment_recurrent_layer(x, mask=mask)
        x = self.feed_forward_layer(x)
        return x

class TXEncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, dropout, n_block, device):
        super(TXEncoderLayer, self).__init__()
        blocks = []
        for i in range(n_block):
            if i == 0:
                tx_encoder_block = TXEncoderBlock(d_word, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, device)
            else:
                tx_encoder_block = TXEncoderBlock(d_conv, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, device)
            blocks.append(tx_encoder_block)
        self.tx_encoder_blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        for i, tx_encoder_block in enumerate(self.tx_encoder_blocks):
            if i == 0:
                x = tx_encoder_block(x, mask)
            else: 
                # x = encoder_block(x, mask=None)
                x = tx_encoder_block(x, mask)
        return self.dropout(x)

class TXModelEncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, dropout, n_block, device):
        super(TXModelEncoderLayer, self).__init__()
        self.input_conv = nn.Conv1d(d_word, d_conv, 1)
        self.tx_encoder_layer = TXEncoderLayer(d_conv, d_conv, kernel_size, memory_len, seg_len, d_attention, d_out, n_conv, n_head, dropout, n_block, device)
        nn.init.xavier_uniform_(self.input_conv.weight)

    def forward(self, x, mask):
        x = x.transpose(-1, -2)
        x = self.input_conv(x)
        x = x.transpose(-1, -2)
        m0 = self.tx_encoder_layer(x, mask) # (batch_size, n_context, dout)
        m1 = self.tx_encoder_layer(m0, mask) # (batch_size, n_context, dout)
        m2 = self.tx_encoder_layer(m1, mask) # (batch_size, n_context, dout)
        # return m0, m1, m2
        return m0, m1, m2