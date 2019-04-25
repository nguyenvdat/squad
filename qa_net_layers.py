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
    
    def forward(self, x):
        mean = torch.mean(x, self.dim, keepdim=True)
        std = torch.std(x, self.dim, keepdim=True)
        return self.g * (x - mean) / (std + self.eps) + self.b

class ResNormLayer(nn.Module):
    """ for an input x add x to the output of applying consecutively n times f_layer to normalized x
    """
    def __init__(self, input_size, f_layer, dropout, n=None, dim=-1):
        super(ResNormLayer, self).__init__()
        if n is None:
            n = 1
        self.f_layers = clones(f_layer, n)
        self.norm_layer = NormLayer(input_size, dim=dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, **kwargs):
        old_x = x
        x = self.norm_layer(x)
        for l in self.f_layers:
            if len(kwargs) > 0:
                x = l(x, kwargs['y'], kwargs['mask'])
            else:
                x = l(x)
        return self.dropout(x + old_x)

class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, d_word, k, d_conv, dropout):
        super(DepthwiseSeparableConvLayer, self).__init__()
        self.depthwise_conv = nn.Conv1d(d_word, d_word, k, groups=d_word, padding=k//2,bias=False)  
        self.pointwise_conv = nn.Conv1d(d_word, d_conv, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
    
    def forward(self, x):
        return self.dropout(F.relu(self.pointwise_conv(self.depthwise_conv(x))))

class AttentionLayer(nn.Module):
    def __init__(self, d_conv, d_attention, dropout=None):
        super(AttentionLayer, self).__init__()
        assert d_conv % d_attention == 0
        self.d_attention = d_attention
        self.dropout = dropout
        self.query_linear = nn.Linear(d_conv, d_attention)
        self.key_linear = nn.Linear(d_conv, d_attention)
        self.value_linear = nn.Linear(d_conv, d_attention)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask):
        query = self.query_linear(x) # (batch_size, len_sentence_x, d_attention)
        key = self.key_linear(y) # (batch_size, len_sentence_y, d_attention)
        value = self.value_linear(y) # (batch_size, len_sentence_y, d_attention)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_attention)
        mask = torch.unsqueeze(mask, 1)
        scores = masked_softmax(scores, mask, -1) # (batch_size, len_sentence_x, len_sentence_y)
        return torch.matmul(scores, value) # (batch_size, len_sentence_x,  d_attention)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_head, d_conv, d_attention, dropout=None):
        super(MultiHeadAttentionLayer, self).__init__()
        attention_layer = AttentionLayer(d_conv, d_attention, dropout)
        self.attention_layers = clones(attention_layer, n_head)

    def forward(self, x, y, mask):
        attention_heads = [attention_layer(x, y, mask) for attention_layer in self.attention_layers]
        return torch.cat(attention_heads, 2) # (batch_size, len_sentence_x, d_attention * n_head)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        nn.init.kaiming_normal_(self.linear.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.linear(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.input_conv = nn.Conv1d(d_word, d_conv, 1)
        depthwise_conv = DepthwiseSeparableConvLayer(d_conv, 7, d_conv, dropout)
        self.depthwise_conv_layers = ResNormLayer(d_conv, depthwise_conv, dropout, n_conv, dim=-2)
        multihead_attention = MultiHeadAttentionLayer(n_head, d_conv, d_attention, dropout)
        self.multihead_attention_layer = ResNormLayer(d_conv, multihead_attention, dropout)
        feed_forward_layer = FeedForwardLayer(d_attention * n_head, d_out, dropout)
        self.feed_forward_layer = ResNormLayer(d_conv, feed_forward_layer, dropout)

    def forward(self, x, mask):
        x = x.transpose(-1, -2)
        x = self.input_conv(x)
        x = self.depthwise_conv_layers(x)
        x = x.transpose(-1, -2)
        x = self.multihead_attention_layer(x, y=x, mask=mask)
        x = self.feed_forward_layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, d_attention, d_out, n_conv, n_head,   dropout, n_block=1):
        super(EncoderLayer, self).__init__()
        blocks = []
        for i in range(n_block):
            if i == 0:
                encoder_block = EncoderBlock(d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout)
            else:
                encoder_block = EncoderBlock(d_conv, d_conv, d_attention, d_out, n_conv, n_head, dropout)
            blocks.append(encoder_block)
        self.encoder_blocks = nn.ModuleList(blocks)
    
    def forward(self, x, mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x

class ContextQueryAttentionLayer(nn.Module):
    """Context Query Attention Layer (mainly Stanford cs224 course's code)

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, drop_prob=0.1):
        super(ContextQueryAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
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

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query.
        """
        c_len, q_len = c.size(1), q.size(1)
        c = self.dropout(c)  # (bs, c_len,input_size)
        q = self.dropout(q)  # (bs, q_len, input_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class ModelEncoderLayer(nn.Module):
    def __init__(self, d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout, n_block):
        super(ModelEncoderLayer, self).__init__()
        self.encoder_layer = EncoderLayer(d_word,d_conv, d_attention, d_out, n_conv, n_head, dropout, n_block)

    def forward(self, x, mask):
        m0 = self.encoder_layer(x, mask) # (batch_size, n_context, dout)
        m1 = self.encoder_layer(m0, mask) # (batch_size, n_context, dout)
        m2 = self.encoder_layer(m1, mask) # (batch_size, n_context, dout)
        return m0, m1, m2

class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super(OutputLayer, self).__init__()
        self.linear1 = nn.Linear(input_size * 2, 1)
        self.linear2 = nn.Linear(input_size * 2, 1)
    
    def forward(self, m0, m1, m2, mask):
        logits_1 = self.linear1(torch.cat((m0, m1), 2)) # (batch_size, n_context, 1)
        print(logits_1.size())
        logits_2 = self.linear2(torch.cat((m0, m2), 2)) # (batch_size, n_context, 1)
        print(logits_2.size())
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2