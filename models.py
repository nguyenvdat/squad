"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import qa_net_layers


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    def __init__(self, word_vectors, char2idx):
        super(QANet, self).__init__()
        # input embedding layer
        d_char = 200
        # d_char = 20
        word_dropout = 0.1
        # word_dropout = 0
        char_dropout = 0.05
        # char_dropout = 0
        hidden_size = 500
        # hidden_size = 320
        highway_dropout = 0.1
        # highway_dropout = 0
        self.emb = qa_net_layers.InputEmbedding(word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout)
        # embedding encoder layer
        d_word = 500
        # d_word = 320
        d_conv = 128
        kernel_size = 7
        d_attention = 16
        d_out = 128
        n_conv = 4
        n_head = 8
        dropout = 0.1
        # dropout = 0
        self.emb_encoder = qa_net_layers.EncoderLayer(d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout)
        # context query attention layer
        dropout = 0.1
        self.att = qa_net_layers.ContextQueryAttentionLayer(d_out, dropout)        
        # model encoder layer
        d_word = 128 * 4
        d_conv = 128
        kernel_size = 5
        d_attention = 16
        d_out = 128
        n_conv = 2
        n_head = 8
        dropout = 0.1
        # dropout = 0
        n_block = 2
        self.model_encoder = qa_net_layers.ModelEncoderLayer(d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout, n_block)
        # output layer
        self.output_layer = qa_net_layers.OutputLayer(128)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # print(c_mask.size())
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs, cc_idxs) # (bs, context_len, hidden_size)
        # print('c_emb: {}'.format(c_emb.size()))
        q_emb = self.emb(qw_idxs, qc_idxs) # (bs, question_len, hidden_size)
        # print('q_emb: {}'.format(q_emb.size()))

        c_enc = self.emb_encoder(c_emb, c_mask) # (bs, context_len, dout)
        # print('c_enc: {}'.format(c_enc.size()))
        q_enc = self.emb_encoder(q_emb, q_mask) # (bs, context_len, dout)
        # print('q_enc: {}'.format(q_enc.size()))

        att = self.att(c_enc, q_enc, c_mask, q_mask) # (bs, context_len, 4*dout)
        # print('att: {}'.format(att.size()))

        # print('model encoder layer')
        m0, m1, m2 = self.model_encoder(att, mask=c_mask)  # (bs, context_len, dout)
        # print('m0: {}'.format(m0.size()))
        # print('m1: {}'.format(m1.size()))
        # print('m2: {}'.format(m2.size()))

        log_p1, log_p2 = self.output_layer(m0, m1, m2, c_mask) # (bs, context_len, 1)

        return log_p1, log_p2
