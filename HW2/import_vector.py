import argparse
import csv
import os
import os
import pickle
import random
import sys
import unittest

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_embedding(filename='glove.6B.50d.txt'):
    """
    Load embedding for the training and
    :return: dataframe, words
    """
    # creat column names
    num = np.arange(51)
    num_str = list(map(str, num))
    list_name = list(map(lambda x: "dim_" + x, num_str))
    df = pd.read_csv("glove.6B.50d.txt", sep=" ", quoting=csv.QUOTE_NONE,
                     header=None, encoding='utf-8',
                     names=list_name)
    df.rename({'dim_0': 'token'}, axis=1, inplace=True)
    words = df.token.to_list()
    # add padding embedding
    df.loc['<PAD>'] = np.zeros(50)
    df.set_index('token', inplace=True)
    df.to_pckle("glove.pkl")
    return df, words


def word_to_embedding(target_vocab, pre_train):
    """

    :param pre_train: pd.DataFrame pre-trained dataframe
    :param target_vocab: list/ array of tokens need to be transformed
    :return: transformed matrix, result dictionary for the unique tokens
    """
    matrix_len = len(target_vocab)
    weighted_matrix = np.zeros((matrix_len + 1, 50))
    words_found = 0
    for i, word in enumerate(target_vocab):
        try:
            weighted_matrix[i] = pre_train.loc[word]
            words_found += 1
        except KeyError:
            weighted_matrix[i] = np.random.normal(size=50)
        if i % 1000 == 0:
            print("Finished {}th words".format(i))
    return weighted_matrix


def create_emb_layer(weighted_matrix1, non_trainable=False):
    """

    :param weighted_matrix1: tensor matrix
    :param non_trainable:
    :return: emb_layer type embedding
    """
    input_shape, embedding_dim = weighted_matrix1.shape
    emb_layer = nn.Embedding.from_pretrained(weighted_matrix1,
                                             padding_idx=input_shape - 1)
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def split_text(text_file):
    """

    :param text_file: training file
    :return: DIC, TOKENS and TAGS

    """
    with open(text_file, mode="r") as file:
        text_f = file.read()
        text_f_lst = text_f.split()
        file.close()
    keys, values = text_f_lst[::2], text_f_lst[1::2]
    result_dic = dict(zip(keys, values))
    return result_dic, keys, values


def prepare_seq(seq_list, dictionary):
    """
    embedding and padded a sequence, given its relating dictionary
    :return: padded sequence in numerical numbers
    """
    embedded = []
    for batch in seq_list:
        empty_lst = [dictionary[tag] for tag in batch]
        embedded.append(empty_lst)
    embedded = [torch.tensor(seq) for seq in embedded]
    padded = nn.utils.rnn.pad_sequence(embedded,
                                       batch_first=True,
                                       padding_value=dictionary['<PAD>'])
    print(padded)
    return padded



<<<<<<< HEAD
class LSTM(nn.Module):
    def __init__(self, nb_layers, batch_size, nb_lstm_units, embedding_layer,
                 bidirectional=False,
                 dropout=0,
                 embedding_dim=50):
        super(LSTM, self).__init__()
=======


class ToyLSTM(nn.Module):
    def __init__(self, nb_layers, batch_size, nb_lstm_units, embedding_layer, embedding_dim=50):
        super(ToyLSTM, self).__init__()
>>>>>>> parent of 6a94ef8 (finish NLL loss function)
        self.hidden_layer = None
        self.result_dic, self.words_lst, self.tags_lst = split_text("wsj1-18.training")
        self.vocab = dict(zip(sorted(set(self.words_lst)),
                              np.arange(len(set(self.words_lst)))))
        self.tags = dict(zip(sorted(set(self.tags_lst)),
                             np.arange(len(set(self.tags_lst)))))
        self.vocab['<PAD>'] = len(set(self.words_lst))
        self.tags['<PAD>'] = len(set(self.tags_lst))
        self.padding_idx = self.vocab['<PAD>']
        self.nb_layers = nb_layers
        self.batch_size = batch_size
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.embedding_layer = embedding_layer
        self.bidirectional = bidirectional
        self.dropout = dropout if nb_layers > 1 else 0
        self.dropout_layer = nn.Dropout(self.dropout)
        # don't count the pad for the tags
        self.nb_tags = len(self.tags) - 1

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            batch_first=True,
            num_layers=self.nb_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )

        # output layer which project back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units * 2
                                       if self.bidirectional
                                       else self.nb_lstm_units
                                       , self.nb_tags)

    def forward(self, input):
        # init hidden layers and input sequence length
        h0 = torch.rand(self.nb_layers, input.size(0), self.nb_lstm_units)
        c0 = torch.rand(self.nb_layers, input.size(0), self.nb_lstm_units)
        input_lengths = torch.all(input != self.padding_idx, dim=2)\
            .sum(dim=1).flatten()

        # -------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len,
        # embedding_dim)
        input = self.dropout_layer(self.embedding_layer(input))
        input = input.squeeze(2)
        # -------------------
        # 2.  Run through LSTM
        # Dim transformation: (B,L, embedding_dim) -> (B, L, LSTM_units)
        input = torch.nn.utils.rnn.pack_padded_sequence(input,
                                                        input_lengths,
                                                        batch_first=True,
                                                        enforce_sorted=False)
        # now run through LSTM
        input = input.float()
        out, (h0, c0) = self.lstm(input, (h0, c0))  # undo the packing operation
        out, len_unpacked = nn.utils.rnn.pad_packed_sequence(out,
                                                             batch_first=True)
        # -------------------
        # 3.  Apply FC linear layer
        # linear layer
        out = out.view(-1,
                       out.size(-1))  # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        out = self.hidden_to_tag(out)  # (batch_size * seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_tags)

        # reshape into (batch_size,  seq_len, nb_lstm_units)
        out = out.view(self.batch_size, -1, self.nb_tags)
        # -------------------
        # 4.  softmax to transfer it to probability
        Y_hat = F.log_softmax(out.float(), dim=2)
        return Y_hat

    def loss(self, Y_hat, Y):
        # NLL(tensor log_softmax output, target index list)
        # flatten out all labels
        Y = prepare_seq(Y, self.tags)
<<<<<<< HEAD
        Y = Y.flatten()
        # flatten all predictions
        Y_hat = Y_hat.view(-1, len(self.tags) - 1)
        # create a mask that filter '<PAD>;
        tag_token = self.tags['<PAD>']
        mask = (Y < tag_token)
        mask_idx = torch.nonzero(mask.float())
        Y_hat = Y_hat[mask_idx].squeeze(1)
        Y = Y[mask_idx].squeeze(1)
        loss = nn.NLLLoss()
        result = loss(Y_hat, Y)
        return result



def unit_test(Y_hat, Y):
    # NLL(tensor log_softmax output, target index list)
    # flatten out all labels
    Y = prepare_seq(Y, tags)
    Y = Y.flatten()
    # flatten all predictions
    Y_hat = Y_hat.view(-1, len(tags) - 1)
    # create a mask that filter '<PAD>;
    tag_token = tags['<PAD>']
    mask = (Y < tag_token)
    mask_idx = torch.nonzero(mask.float())
    Y_hat = Y_hat[mask_idx].squeeze(1)
    Y = Y[mask_idx].squeeze(1)
    loss = nn.NLLLoss()
    result = loss(Y_hat, Y)
    print(result)
    return result
=======
        Y_length = Y_hat.size(1)  # because Y_hat is already padded

        Y = Y.flatten()
        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.nb_tags)
        # create a mask that filter '<PAD>;
        tag_token = self.tags['<PAD>']  # todo: 输入应该是padded过的文字 seq
        mask = (Y != tag_token).float().unsqueeze(1)
        Y_hat = mask * Y_hat  # (batch_size * seq_len, nb_tags) <==> (N*L, C)
        nllloss = nn.NLLLoss(Y_hat, Y)

        return nllloss


def unit_test(input, embedded=False):
    # specify hyperparameters
    num_layers = 2
    nb_tags = 10
    if embedded:
        input_size = 50
        input = embedding_layer_const(input)
        # need to squeeze pos 2 to make it 3D, after embedding
        input = input.squeeze(2)
        padding_idx = torch.zeros(input_size)
    else:
        input_size = input.size()[-1]
    h0 = torch.zeros(num_layers, batch_size, nb_lstm_units)
    c0 = torch.zeros(num_layers, batch_size, nb_lstm_units)
    lstm = nn.LSTM(input_size=input_size,
                   hidden_size=nb_lstm_units,
                   num_layers=num_layers,
                   batch_first=True)
    fc_linear = nn.Linear(nb_lstm_units, nb_tags)
    input = input.float()
    # pack the padded sequence
    input_lengths = torch.all(input != padding_idx, dim=2).sum(dim=1).flatten()
    input = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True, enforce_sorted=False)
    out, (h0, c0) = lstm(input, (h0, c0))
    # unpacking the padded sequence
    out, len_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    print(out.size())
    # linear layer
    out = out.view(-1, out.size(-1))  # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
    out = fc_linear(out)  # (batch_size * seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_tags)

    # reshape into (batch_size,  seq_len, nb_lstm_units)
    out = out.view(batch_size, -1, nb_tags)
    # softmax to get the result
    Y_hat = F.log_softmax(out.float(), dim=2)
    return Y_hat
>>>>>>> parent of 6a94ef8 (finish NLL loss function)


# todo: prepare dataloader for text


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    ### transform text into list of words
    df = pd.read_pickle('glove.pkl')
    _, words_lst, tags_lst = split_text("wsj1-18.training")
    tags = dict(zip(sorted(set(tags_lst)), np.arange(len(set(tags_lst)))))
    tags['<PAD>'] = 912344
    weighted_matrix = torch.load("weighed_matrix.pt")
    embedding_layer_const = create_emb_layer(weighted_matrix)
    nb_layers = 2
    nb_lstm_units = 32
    batch_size = 3
    seq_len = 4
    padding_idx = 912344
    toy_training = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    batch_in = torch.tensor([[[4],
                              [5],
                              [912344],
                              [912344]],

                             [[6],
                              [912344],
                              [912344],
                              [912344]],

                             [[7],
                              [8],
                              [9],
                              [10]]])
<<<<<<< HEAD
    Y = [["CC", "CD", "DT"], ["EX"], ["JJ", "IN", "JJ", "JJR"]]
    model = LSTM(nb_layers=nb_layers,
                 batch_size=batch_size,
                 nb_lstm_units=nb_lstm_units,
                 embedding_layer=embedding_layer_const)
=======
    model = ToyLSTM(nb_layers=nb_layers,
                    batch_size=batch_size,
                    nb_lstm_units=nb_lstm_units,
                    embedding_layer=embedding_layer_const)
>>>>>>> parent of 6a94ef8 (finish NLL loss function)
    out = model(batch_in)

    # out = unit_test(batch_in, embedded=True)
    # unittest.main()
