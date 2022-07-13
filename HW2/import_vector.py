import unittest
import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import nltk
import os
from torch.autograd import Variable


def load_embedding(filename='glove.6B.50d.txt'):
    """
    Load embedding for the training and
    :return: dataframe, words
    """
    # creat column names
    num = np.arange(51)
    num_str = list(map(str, num))
    list_name = list(map(lambda x: "dim_" + x, num_str))
    df = pd.read_csv("glove.6B.50d.txt", sep=" ", quoting=csv.QUOTE_NONE, header=None, encoding='utf-8',
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


class ToyLSTM(nn.Module):
    def __init__(self, nb_layers, batch_size, nb_lstm_units, embedding_layer, embedding_dim=50):
        super(ToyLSTM, self).__init__()
        self.hidden_layer = None
        self.result_dic, self.words_lst, self.tags_lst = split_text("wsj1-18.training")
        self.vocab = dict(zip(sorted(set(self.words_lst)), np.arange(1, len(set(self.words_lst)) + 1)))
        self.tags = dict(zip(sorted(set(self.tags_lst)), np.arange(1, len(set(self.tags_lst)) + 1)))
        self.vocab['<PAD>'] = 0
        self.tags['<PAD>'] = 0
        self.padding_idx = 912344
        self.nb_layers = nb_layers
        self.batch_size = batch_size
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.embedding_layer = embedding_layer

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
            num_layers=self.nb_layers
        )

        # output layer which project back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise LSTM will treat a batch as a
        # continuation of a sequence
        h0 = torch.rand(self.nb_layers, X.size(0), self.nb_lstm_units)
        c0 = torch.rand(self.nb_layers, X.size(0), self.nb_lstm_units)

        # -------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.embedding_layer(X)

        # -------------------
        # 2.  Run through LSTM
        ### ASSUMPTION: already padded
        # Dim transformation: (B,L, embedding_dim) -> (B, L, LSTM_units)
        # pack padded items so that they are not shown to the LSTM
        ### creating a mask that filter all non-zeros(<PAD>) in the tensor
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        # now run through LSTM
        print("X is PackedSequence: {}".format(isinstance(X, torch.nn.utils.rnn.PackedSequence)))
        X = X.float()
        X, _ = self.lstm(X, (h0, c0))  # undo the packing operation
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # # -------------------
        # # 3. Project to the tag space
        # # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size, seq_len, tag_nb)
        # X = X.contiguous()
        # # reshape data so that it goes to the linear layer
        # X = X.view(-1, X.shape[2])
        # X = self.hidden_to_tag(X)
        #
        # # -------------------
        # # 4. Create softmax activation bc we are doing classification
        # # Dim transformation: (batch_size * seq_len, tag_nb, batch_size, seq_len, tag_nb)
        # X = F.log_softmax(X, dim=1)
        # X = X.view(batch_size, seq_len, self.nb_tags)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # the simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all labels
        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view_as(Y)

        # create a mask by filtering out all the tokens that are not the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the value for the label and zero out the rest with the mask
        Y_hat = Y_hat * mask

        # compute the cross entropy loss which ignore all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens
        return ce_loss


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
    input_lengths =torch.all(input != padding_idx, dim=2).sum(dim=1).flatten()
    input = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True, enforce_sorted=False)
    out, (h0, c0) = lstm(input, (h0, c0))
    # unpacking the padded sequence
    out, len_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

    # linear layer
    out = out.view(-1, out.size(-1))  # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
    out = fc_linear(out)  # (batch_size * seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_tags)

    # reshape into (batch_size,  seq_len, nb_lstm_units)
    out = out.view(batch_size, -1, nb_tags)
    # softmax to get the result
    Y_hat = F.log_softmax(out.float(), dim=2)

    return Y_hat


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    ### transform text into list of words
    df = pd.read_pickle('glove.pkl')
    _, words_lst, __ = split_text("wsj1-18.training")
    weighted_matrix = torch.load("weighed_matrix.pt")
    embedding_layer_const = create_emb_layer(weighted_matrix)
    nb_layers = 2
    nb_lstm_units = 32
    batch_size = 3
    seq_len = 4
    padding_idx = 912344
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
    # model = ToyLSTM(nb_layers=nb_layers,
    #                 batch_size=batch_size,
    #                 nb_lstm_units=nb_lstm_units,
    #                 embedding_layer=embedding_layer_const)

    out = unit_test(batch_in, embedded=True)
    # unittest.main()
