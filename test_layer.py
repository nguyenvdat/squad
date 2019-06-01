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
from models import BiDAF, QANet, TransformerXL
# from tensorboardX import SummaryWriter
# from tqdm import tqdm
from ujson import load as json_load
from ujson import dump as json_dump
from util import collate_fn, SQuAD
import layers
from qa_net_layers import *
import util
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def main():
    # test_input_embedding()
    # test_embedding_encoder()
    # test_context_query_attention()
    # test_model_encoder()
    # test_output_layer()
    # test_model()
    # args = get_train_args()
    # outfile = '/Volumes/Blazing/data/train_light.npz'
    # infile = '/Volumes/Blazing/data/train.npz'
    # outfile = '/Volumes/Blazing/data/train_light.npz'
    # dataset = np.load(infile)
    # np.savez(outfile, context_idxs=dataset['context_idxs'][:70000],
    #                   context_char_idxs=dataset['context_char_idxs'][:70000],
    #                   ques_idxs=dataset['ques_idxs'][:70000],
    #                   ques_char_idxs=dataset['ques_char_idxs'][:70000],
    #                   y1s=dataset['y1s'][:70000],
    #                   y2s=dataset['y2s'][:70000],
    #                   ids=dataset['ids'][:70000])
    # test_segment_recurrent()
    test_transformer_xl_model()


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
    input_embedding = InputEmbedding(
        word_vectors, d_char, char2idx, hidden_size, word_dropout, char_dropout, highway_dropout)

    word_inputs = torch.tensor([[1, 2, 0], [1, 2, 4]], dtype=torch.long)
    char_inputs = torch.tensor([[[1, 2, 2, 0], [1, 3, 2, 3], [0, 0, 0, 0]], [
                               [1, 5, 2, 0], [1, 3, 6, 3], [3, 4, 2, 1]]], dtype=torch.long)
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

    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    input_conv = nn.Conv1d(d_word, d_conv, 1)
    depthwise_conv = DepthwiseSeparableConvLayer(d_conv, 7, d_conv, dropout)
    depthwise_conv_layers = ResNormLayer(
        d_conv, depthwise_conv, dropout, n_conv, dim=-2)
    multihead_attention = MultiHeadAttentionLayer(
        n_head, d_conv, d_attention, dropout)
    multihead_attention_layer = ResNormLayer(
        d_conv, multihead_attention, dropout)
    feed_forward_layer = FeedForwardLayer(d_attention * n_head, d_out, dropout)
    feed_forward_layer = ResNormLayer(d_conv, feed_forward_layer, dropout)

    x = x.transpose(-1, -2)
    x = input_conv(x)
    assert x.size() == (2, 128, 3)
    x = depthwise_conv_layers(x)
    x = x.transpose(-1, -2)
    assert x.size() == (2, 3, 128)
    x = multihead_attention_layer(x, y=x, mask=mask)
    assert x.size() == (2, 3, 128)
    x = feed_forward_layer(x)
    assert x.size() == (2, 3, 128)

    x = old_x
    embedding_encoder = EncoderLayer(
        d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout)
    output = embedding_encoder(x, mask)
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
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    dropout = 0.1

    x = torch.rand(2, 5, 128 * 4)

    model_encoder = ModelEncoderLayer(
        d_word, d_conv, d_attention, d_out, n_conv, n_head, dropout, n_block=7)

    m0, m1, m2 = model_encoder(x, mask)
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


def test_qa_net_model():
    args = get_setup_args()
    word_vectors = util.torch_from_json(args.word_emb_file)
    with open(args.char2idx_file, "r") as f:
        char2idx = json_load(f)
    model = QANet(word_vectors, char2idx)
    cw_idxs = torch.randint(2, 1000, (64, 374))
    cc_idxs = torch.randint(2, 50, (64, 374, 200))
    qw_idxs = torch.randint(2, 1000, (64, 70))
    qc_idxs = torch.randint(2, 50, (64, 70, 200))
    cw_idxs[:, 0] = 1
    cw_idxs[3, -1] = 0
    qw_idxs[:, 0] = 1
    qw_idxs[3, -1] = 0
    out = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
    print(out)


def test_segment_recurrent():
    memory_len = 64
    seg_len = 32
    d_in = 128
    d_out = 16
    n_head = 8
    sr = SegmentRecurrent(memory_len, seg_len, d_in, d_out, n_head)
    x = torch.rand(64, 300, 128)
    mask = torch.ones(64, 300)
    mask[1:20, 200:] = 0
    mask[80:100, 250:] = 0
    out = sr(x, mask)
    print(out.size())
    assert out.size() == (64, 300, 128)


def test_transformer_xl_model():
    args = get_setup_args()
    word_vectors = util.torch_from_json(args.word_emb_file)
    with open(args.char2idx_file, "r") as f:
        char2idx = json_load(f)
    model = TransformerXL(word_vectors, char2idx)
    cw_idxs = torch.randint(2, 1000, (64, 374))
    cc_idxs = torch.randint(2, 50, (64, 374, 200))
    qw_idxs = torch.randint(2, 1000, (64, 70))
    qc_idxs = torch.randint(2, 50, (64, 70, 200))
    cw_idxs[:, 0] = 1
    cw_idxs[3, -1] = 0
    qw_idxs[:, 0] = 1
    qw_idxs[3, -1] = 0
    out = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
    print(out.size())


def test_bert():
    args = get_setup_args()
    with open(args.word2idx_file, "r") as f:
        word2idx = json_load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}

    idx2word[0] = "[PAD]"
    idx2word[1] = "[UNK]"
    with open('idx2word.json', 'w') as f:
        json_dump(idx2word, f, indent=4, sort_keys=True)
    # print(idx2word)
    cw_idxs = torch.tensor([[1,   256,   128,  1486,  4580,  4186,  7723,     6,    59, 17032,
                             9,  1849,    39,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0],
                            [1,   191,   534,     7,  2505,   471,    12, 15309,    26,     4,
                             3158,  1392,    39,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0]])
    qw_idxs = torch.tensor([[1,   256,   128,  1486,  4580,  4186,  7723,     6,    59, 17032,
                             9,  1849,    39,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0],
                            [1,   191,   534,     7,  2505,   471,    12, 15309,    26,     4,
                             3158,  1392,    39,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0]])
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    indexed_tokens_batch = []
    segments_ids_batch = []

    for i in range(len(cw_idxs)):
        q = qw_idxs[i]
        c = cw_idxs[i]
        q_text = ' '.join([idx2word[idx.item()] for idx in q])
        q_text = '[CLS] ' + q_text + ' [SEP] '
        c_text = ' '.join([idx2word[idx.item()] for idx in c])
        c_text = c_text + ' [SEP]'
        text = q_text + c_text
        # print(text)
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens_batch.append(indexed_tokens)
        segments_ids_batch.append([0] * (len(q) + 2) + [1] * (len(c) + 1))
        print(i)
        print(len(indexed_tokens))
        print(len(segments_ids_batch[i]))

    model = BertModel.from_pretrained('bert-large-uncased')
    model.eval()

    tokens_tensor = torch.tensor(indexed_tokens_batch)
    segments_tensors = torch.tensor(segments_ids_batch)
    print('tensor size: ')
    print(tokens_tensor.size())
    print(segments_tensors.size())
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    print(len(encoded_layers))
    print(encoded_layers[-1].size())

    # text1 = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # text2 = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a really good puppeteer [SEP]"
    # text = [text1, text2]
    # tokenized_text = tokenizer.tokenize(text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print('indexed tokens')
    # print(indexed_tokens)
    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    # model = BertModel.from_pretrained('bert-large-uncased')
    # model.eval()
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')

    # # Predict hidden states features for each layer
    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # print(encoded_layers[-1].size())


if __name__ == '__main__':
    # main()
    test_bert()
