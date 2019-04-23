import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args, get_setup_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
import layers
from qa_net_layers import *
import util
import pickle

def main():
    # test_input_embedding()
    # test_embedding_encoder()
    # test_context_query_attention()
    # test_model_encoder()
    test_output_layer()
    
def test_input_embedding():
    args = get_setup_args()
    d_char = 200
    word_dropout = 0.1
    char_dropout = 0.05
    with open(args.char2idx_file, "r") as f:
        char2idx = json_load(f)
    hidden_size = 500
    highway_dropout = 0.1
    word_vectors = util.torch_from_json(args.word_emb_file)
    input_embedding = InputEmbedding(word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout)

    word_inputs = torch.tensor([[1, 2, 0], [1, 2, 4]], dtype=torch.long)
    char_inputs = torch.tensor([[[1, 2, 2, 0], [1, 3, 2, 3], [0, 0, 0, 0]], [[1, 5, 2, 0], [1, 3, 6, 3], [3, 4, 2, 1]]], dtype=torch.long)
    emb = input_embedding(word_inputs, char_inputs)
    pickle_in = open('input_emb.pickle', 'wb')
    pickle.dump(emb, pickle_in)
    assert emb.size() == (2, 3, 500)
    return emb

def test_embedding_encoder():
    input_emb_pickle = open('input_emb.pickle', 'rb')
    x = pickle.load(input_emb_pickle)
    old_x = x
    d_word = 500
    d_conv = 128
    d_attention = 16
    d_out = 128
    n_conv = 4
    n_head = 8
    mask = None
    dropout = 0.1

    input_conv = nn.Conv1d(d_word, d_conv, 1)
    depthwise_conv = DepthwiseSeparableConvLayer(d_conv, 7, d_conv, dropout)
    depthwise_conv_layers = ResNormLayer(d_conv, depthwise_conv, dropout, n_conv, dim=-2)
    multihead_attention = MultiHeadAttentionLayer(n_head, d_conv, d_attention, mask, dropout)
    multihead_attention_layer = ResNormLayer(d_conv, multihead_attention, dropout)
    feed_forward_layer = FeedForwardLayer(d_attention * n_head, d_out, dropout)
    feed_forward_layer = ResNormLayer(d_conv, feed_forward_layer, dropout)

    x = x.transpose(-1, -2)
    x = input_conv(x)
    assert x.size() == (2, 128, 3)
    x = depthwise_conv_layers(x)
    x = x.transpose(-1, -2)
    assert x.size() == (2, 3, 128)
    x = multihead_attention_layer(x, y=x)
    assert x.size() == (2, 3, 128) 
    x = feed_forward_layer(x)
    assert x.size() == (2, 3, 128) 

    x = old_x
    embedding_encoder = EncoderLayer(d_word,d_conv, d_attention, d_out, n_conv, n_head, dropout)
    output = embedding_encoder(x)
    assert output.size() == (2, 3, 128)
    return output

def test_context_query_attention():
    context = torch.rand(2, 5, 4)
    query = torch.rand(2, 3, 4)
    c_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    q_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    context_query_attention = ContextQueryAttentionLayer(4)
    x = context_query_attention(context, query, c_mask, q_mask)
    assert x.size() == (2, 5, 16)
    return x

def test_model_encoder():
    d_word = 128 * 4
    d_conv = 128 * 4
    d_attention = 64
    d_out = 128 * 4
    n_conv = 2
    n_head = 8
    mask = None
    dropout = 0.1

    x = torch.rand(2, 5, 128 * 4)

    model_encoder = ModelEncoderLayer(d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout, n_block=7)

    m0, m1, m2 = model_encoder(x)
    assert m0.size() == (2, 5, 512)
    assert m1.size() == (2, 5, 512)
    assert m2.size() == (2, 5, 512)

def test_output_layer():
    m0 = torch.rand(2, 3, 5)
    m1 = torch.rand(2, 3, 5)
    m2 = torch.rand(2, 3, 5)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    output_layer = OutputLayer(5)
    log_p1, log_p2 = output_layer(m0, m1, m2, mask)
    print(log_p1)
    print(log_p2)

if __name__ == '__main__':
    main()
