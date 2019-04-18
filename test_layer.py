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

def main():
    test_input_embedding()
    
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
    assert emb.size() == (2, 3, 500)
    return emb


if __name__ == '__main__':
    main()
