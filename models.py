"""Top-level model classes.

Author:
    Dat Nguyen (QANet and TransformerXL part)
"""

import layers
import torch
import torch.nn as nn
import qa_net_layers
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from util import masked_softmax
import numpy as np


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

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

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
        self.emb = qa_net_layers.InputEmbedding(
            word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout)
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
        self.emb_encoder = qa_net_layers.EncoderLayer(
            d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout)
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
        self.model_encoder = qa_net_layers.ModelEncoderLayer(
            d_word, d_conv, kernel_size, d_attention, d_out, n_conv, n_head, dropout, n_block)
        # output layer
        self.output_layer = qa_net_layers.OutputLayer(128)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # print(c_mask.size())
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs, cc_idxs)  # (bs, context_len, hidden_size)
        # print('c_emb: {}'.format(c_emb.size()))
        q_emb = self.emb(qw_idxs, qc_idxs)  # (bs, question_len, hidden_size)
        # print('q_emb: {}'.format(q_emb.size()))

        c_enc = self.emb_encoder(c_emb, c_mask)  # (bs, context_len, dout)
        # print('c_enc: {}'.format(c_enc.size()))
        q_enc = self.emb_encoder(q_emb, q_mask)  # (bs, context_len, dout)
        # print('q_enc: {}'.format(q_enc.size()))

        # (bs, context_len, 4*dout)
        att = self.att(c_enc, q_enc, c_mask, q_mask)
        # print('att: {}'.format(att.size()))

        # print('model encoder layer')
        m0, m1, m2 = self.model_encoder(
            att, mask=c_mask)  # (bs, context_len, dout)
        # print('m0: {}'.format(m0.size()))
        # print('m1: {}'.format(m1.size()))
        # print('m2: {}'.format(m2.size()))

        log_p1, log_p2 = self.output_layer(
            m0, m1, m2, c_mask)  # (bs, context_len, 1)

        return log_p1, log_p2


class TransformerXL(nn.Module):
    def __init__(self, word_vectors, char2idx, device):
        super(TransformerXL, self).__init__()
        self.device = device
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
        self.emb = qa_net_layers.InputEmbedding(
            word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout)
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
        memory_len_train = 64
        memory_len_eval = 128
        seg_len = 32
        # dropout = 0
        self.emb_encoder = qa_net_layers.TXEncoderLayer(
            d_word, d_conv, kernel_size, memory_len_train, memory_len_eval, seg_len, d_attention, d_out, n_conv, n_head, dropout, 1, device, 1)
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
        memory_len_train = 64
        memory_len_eval = 128
        seg_len = 32
        # dropout = 0
        n_block = 2
        self.model_encoder = qa_net_layers.TXModelEncoderLayer(
            d_word, d_conv, kernel_size, memory_len_train, memory_len_eval, seg_len, d_attention, d_out, n_conv, n_head, dropout, n_block, device, 1)
        # output layer
        self.output_layer = qa_net_layers.OutputLayer(128)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # print(c_mask.size())
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs, cc_idxs)  # (bs, context_len, hidden_size)
        # print('c_emb: {}'.format(c_emb.size()))
        q_emb = self.emb(qw_idxs, qc_idxs)  # (bs, question_len, hidden_size)
        # print('q_emb: {}'.format(q_emb.size()))

        c_enc = self.emb_encoder(c_emb, c_mask)  # (bs, context_len, dout)
        # print('c_enc: {}'.format(c_enc.size()))
        q_enc = self.emb_encoder(q_emb, q_mask)  # (bs, context_len, dout)
        # print('q_enc: {}'.format(q_enc.size()))

        # (bs, context_len, 4*dout)
        att = self.att(c_enc, q_enc, c_mask, q_mask)
        # print('att: {}'.format(att.size()))

        # print('model encoder layer')
        m0, m1, m2 = self.model_encoder(
            att, mask=c_mask)  # (bs, context_len, dout)
        # print('m0: {}'.format(m0.size()))
        # print('m1: {}'.format(m1.size()))
        # print('m2: {}'.format(m2.size()))

        log_p1, log_p2 = self.output_layer(
            m0, m1, m2, c_mask)  # (bs, context_len, 1)

        return log_p1, log_p2


class BertMd(nn.Module):
    def __init__(self, idx2word):
        super(BertMd, self).__init__()
        self.idx2word = idx2word
        # self.idx2word = {idx: word for word, idx in word2idx.items()}
        # self.idx2word[0] = "[PAD]"
        # self.idx2word[1] = "[UNK]"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to('cuda')
        self.W1 = nn.Linear(768, 1)
        self.W2 = nn.Linear(768, 1)

    def forward(self, cw_idxs, qw_idxs, y1, y2):
        indexed_tokens_batch = []
        segments_ids_batch = []
        # print('context size')
        # print(cw_idxs.size())
        # print('question size')
        # print(qw_idxs.size())
        q_batch = []
        c_batch = []
        if not self.training:
            y_maps = []
        for i in range(len(cw_idxs)):
            q = qw_idxs[i]
            c = cw_idxs[i]
            q_text = [self.idx2word[str(idx.item())].lower() if self.idx2word[str(
                idx.item())][0] != '[' else self.idx2word[str(idx.item())] for idx in q]
            c_text = [self.idx2word[str(idx.item())].lower() if self.idx2word[str(
                idx.item())][0] != '[' else self.idx2word[str(idx.item())] for idx in c]
            tokenized_q_text = []
            tokenized_c_text = []
            if not self.training:
                y_map = []
            for word in q_text:
                tokenized_q_text += self.tokenizer.tokenize(word)
            for j, word in enumerate(c_text):
                if j == y1[i]:
                    y1_index = len(tokenized_c_text)
                if j == y2[i]:
                    y2_index = len(tokenized_c_text)
                tokenized_c_text += self.tokenizer.tokenize(word)
                if not self.training:
                    y_map += [j] * (len(tokenized_c_text) - len(y_map))
            y1[i] = y1_index
            y2[i] = y2_index
            q_batch.append(tokenized_q_text)
            c_batch.append(tokenized_c_text)
            if not self.training:
                y_maps.append(y_map)

        cw_idxs = None
        qw_idxs = None
        max_len_c = max(len(c) for c in c_batch)
        max_len_q = max(len(q) for q in q_batch)

        c_mask = []
        for i in range(len(c_batch)):
            while len(c_batch[i]) < max_len_c:
                c_batch[i].append('[PAD]')
                if not self.training:
                    y_maps[i].append(max_len_c)
            c_batch[i] = c_batch[i] + ['[SEP]']
            if not self.training:
                y_maps[i].append(max_len_c)
            while len(q_batch[i]) < max_len_q:
                q_batch[i].append('[PAD]')
            q_batch[i] = ['[CLS]'] + q_batch[i] + ['[SEP]']
            c_mask.append(
                [1 if x != '[PAD]' and x != '[SEP]' else 0 for x in c_batch[i]])
            segments_ids_batch.append(
                [0] * len(q_batch[i]) + [1] * len(c_batch[i]))
            tokenized_text = q_batch[i] + c_batch[i]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                tokenized_text)
            indexed_tokens_batch.append(indexed_tokens)
            if not self.training:
                assert len(y_maps[i]) == len(c_batch[i])

        tokens_tensor = torch.tensor(indexed_tokens_batch)
        segments_tensors = torch.tensor(segments_ids_batch)
        c_mask = torch.tensor(c_mask)
        c_mask = c_mask.to('cuda')
        # print('token tensor size')
        # print(tokens_tensor.size())
        # print('segment tensor size')
        # print(segments_tensor.size())
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')

        encoded_layers, _ = self.model(
            tokens_tensor, segments_tensors, output_all_encoded_layers=False)

        # (bs, c_len, dbert)
        c_layer = encoded_layers[:, max_len_q + 2:, :]
        # print(c_layer.size())
        # print(c_mask.size())
        logits_1 = torch.squeeze(self.W1(c_layer))  # (bs, c_len)
        logits_2 = torch.squeeze(self.W2(c_layer))  # (bs, c_len)
        log_p1 = masked_softmax(logits_1, c_mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2, c_mask, log_softmax=True)
        if not self.training:
            return log_p1, log_p2, y1, y2, np.array(y_maps)
        return log_p1, log_p2, y1, y2
