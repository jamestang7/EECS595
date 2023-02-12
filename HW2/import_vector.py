import argparse
import csv
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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
    df.set_index('token', inplace=True)
    df.loc['<PAD>'] = np.zeros(50)
    df.to_pickle("glove.pkl")
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
    if type(weighted_matrix1) == np.ndarray:
        weighted_matrix1 = torch.from_numpy(weighted_matrix1)
    emb_layer = nn.Embedding.from_pretrained(weighted_matrix1,
                                             padding_idx=input_shape - 1)
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def split_text(text_file, by_line=False):
    """

    :param text_file: training file
    :return: DIC, TOKENS and TAGS

    """
    if by_line == False:
        with open(text_file, mode="r") as file:
            text_f = file.read()
            text_f_lst = text_f.split()
            file.close()
        keys, values = text_f_lst[::2], text_f_lst[1::2]
        result_dic = dict(zip(keys, values))
        return result_dic, keys, values
    else:
        with open(text_file, mode="r") as file:
            text_f = file.read()
            text_f_lst = text_f.splitlines()
            file.close()
        keys = [line.split()[::2] for line in text_f_lst]
        values = [line.split()[1::2] for line in text_f_lst]
        # result_dic = dict(zip(keys, values))
        return keys, values


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


def get_length_tensor(batch, padding_idx=912344):
    index = batch.size(0)
    result = []
    for i in range(index):
        sentence = batch[i]
        length = len(
            torch.nonzero(sentence != padding_idx))  # grab ith tensor's length
        result.append(length)
    return result


class LSTM(nn.Module):
    def __init__(self, nb_layers, batch_size, nb_lstm_units, embedding_layer,
                 bidirectional=False,
                 dropout=0,
                 embedding_dim=50):
        super(LSTM, self).__init__()
        self.hidden_layer = None
        self.result_dic, self.words_lst, self.tags_lst = split_text(
            "wsj1-18.training")
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

    def forward(self, input, *args, **kwargs):
        # init hidden layers and input sequence length
        input.to(device)
        h0 = torch.rand(self.nb_layers, input.size(0), self.nb_lstm_units).to(device)
        c0 = torch.rand(self.nb_layers, input.size(0), self.nb_lstm_units).to(device)
        input_lengths = get_length_tensor(input)

        # -------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len,
        # embedding_dim)
        input = self.dropout_layer(self.embedding_layer(input.long()))
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
        out = out.view(-1, out.size(
            -1))  # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        out = self.hidden_to_tag(
            out)  # (batch_size * seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_tags)

        # reshape into (batch_size,  seq_len, nb_lstm_units)
        out = out.view(self.batch_size, -1, self.nb_tags)
        # -------------------
        # 4.  softmax to transfer it to probability
        # Y_hat = F.log_softmax(out.float(), dim=2)
        return out

    def loss(self, Y_hat, Y):
        # NLL(tensor log_softmax output, target index list)
        # flatten out all labels
        ## next line deprecated because Y is already padded in data loader
        # Y = prepare_seq(Y, self.tags)  # convert labels into number by tag dict
        # Y = Y.flatten()
        # # flatten all predictions
        # Y_hat = Y_hat.view(-1, len(self.tags) - 1)
        # # create a mask that filter '<PAD>;
        tag_token = self.tags['<PAD>']
        # mask = (Y < tag_token)
        # mask_idx = torch.nonzero(mask.float())
        # Y_hat = Y_hat[mask_idx].squeeze(1)
        # Y = Y[mask_idx].squeeze(1)
        # loss = nn.NLLLoss()
        # result = loss(Y_hat, Y)

        ### second approac using ignore_idx = 45
        ### flatten Y_hat and apply log_softmax
        Y_hat = Y_hat.view(-1, tag_token).float()
        Y_hat = F.log_softmax(Y_hat, dim=1).double()
        Y = Y.flatten().long()
        loss = nn.NLLLoss(ignore_index=tag_token)
        result = loss(Y_hat, Y)
        return result


class TagDataset(Dataset):
    def __init__(self, train=True):
        """

        :param train: bool, if True read training data, else read testing data
        """
        self.train = train
        if self.train:

            self.trainX, self.trainY = split_text(
                "wsj1-18.training", by_line=True)
        else:
            self.testX, self.testY = split_text("wsj19-21.truth",
                                                by_line=True)

    def __len__(self):
        return len(self.trainY) if self.train else len(self.testY)

    def __getitem__(self, item):
        if self.train:
            return self.trainX[item], self.trainY[item]
        else:
            return self.testX[item], self.testY[item]


#  customized collate_fn to get to equal size
def collate_fn(batch):
    def helper(target, batch):
        if target == 'X':
            dic = vocab
            list_sentence = [item[0] for item in batch]
        else:
            dic = tags
            list_sentence = [item[1] for item in batch]
        # if test not in training dic
        try:
            list_sentence = [[dic[word] for word in sentence] for sentence in
                             list_sentence]
        except:
            list_sentence_temp = []
            for sentence in list_sentence:
                _ = []
                for word in sentence:
                    try:
                        temp = dic[word]
                    except:
                        temp = dic['James']
                    _.append(temp)
                list_sentence_temp.append(_)
                list_sentence = list_sentence_temp
        length_list = [len(sentence) for sentence in list_sentence]
        max_length = max(length_list)
        batch_size = len(list_sentence)
        pad_token = dic['<PAD>']
        # init tensors of ones with batch_size * max_length
        result = np.ones((batch_size, max_length)) * pad_token
        # populate the result
        for i, length in enumerate(length_list):
            sequence = list_sentence[i]
            result[i][0:length] = sequence
        return torch.from_numpy(result)

    return [helper('X', batch), helper('Y', batch)]


batch_size = 8
train_loader = DataLoader(TagDataset(train=True),
                          batch_size=batch_size,
                          # num_workers=8, Mac M1 cannot use this
                          collate_fn=collate_fn,
                          shuffle=False,
                          drop_last=True)
test_loader = DataLoader(TagDataset(train=False),
                         batch_size=batch_size,
                         # num_workers=8, Mac M1 cannot use this
                         collate_fn=collate_fn,
                         shuffle=False,
                         drop_last=True)


# done: accuracy computation
def get_accuracy(model, train):
    data = train_loader if train else test_loader
    correct, total = 0, 0
    model = model.to(device)
    for X, labels in data:
        X, labels = X.to(device), labels.to(device)
        tag_padding_token = tags['<PAD>']
        y_pred = model(X)
        y_pred = F.softmax(y_pred, dim=2)  # change into probability
        # select the maximum probablilty in dim2, out of all tags
        y_pred = y_pred.max(dim=2)[1]
        # flatten all prediction
        y_pred, labels = y_pred.flatten(), labels.flatten()
        mask = (labels < tag_padding_token)
        y_pred, labels = y_pred[mask], labels[mask]
        correct += labels.eq(y_pred).sum().item()
        total += len(mask)
    return correct / total


def train(model, lr, momemtum, num_epoch=1):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momemtum)
    iters, losses, train_acc, test_acc = [], [], [], []
    model = model.to(device=device)
    # training
    n = 0  # number of iterations
    for epoch in range(num_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            print("Batch:{}".format(batch_idx+1))
            out = model(data)  # forward pass
            loss = model.loss(out, target)  # compute the loss
            loss.backward()  # backwardpass (compute parameter updates)
            optimizer.step()  # make the update to each parameter
            optimizer.zero_grad()

            # save the current training log
            iters.append(n)
            losses.append(float(loss) / batch_size)
            train_acc.append(get_accuracy(model, train=True))
            test_acc.append(get_accuracy(model, train=False))
            n += 1
            # print result
            print(f"Epoch: {epoch + 1}; Batch: {batch_idx + 1}; "
                  f"Loss: {float(loss) / batch_size};"
                  f"Training Acc:{train_acc[-1]};"
                  f"Testing Acc:{test_acc[-1]}")

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training/Testing Curve with Accuracy")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, train_acc, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()

    print("Final training Accuracy: {:.2%}".format(train_acc[-1]))
    print("Final testing Accuracy: {:.2%}".format(test_acc[-1]))



if __name__ == '__main__':
    ### transform text into list of words

    df = pd.read_pickle('glove.pkl')
    _, words_lst, tags_lst = split_text("wsj1-18.training")
    tags = dict(zip(sorted(set(tags_lst)), np.arange(len(set(tags_lst)))))
    tags['<PAD>'] = 45
    vocab = dict(zip(sorted(set(words_lst)), np.arange(len(set(words_lst)))))
    vocab['<PAD>'] = 912344
    weighted_matrix = torch.load("weighed_matrix.pt")
    embedding_layer_const = create_emb_layer(weighted_matrix)
    nb_layers = 2
    nb_lstm_units = 64
    batch_size = 8
    seq_len = 30
    padding_idx = 912344
    model = LSTM(nb_layers=nb_layers,
                 batch_size=batch_size,
                 nb_lstm_units=nb_lstm_units,
                 embedding_layer=embedding_layer_const,
                 bidirectional=False)
    lr = 1e-3
    momentum = 0.9
    num_epoch = 5
    train(model=model, lr=lr, momemtum=momentum, num_epoch=num_epoch)

    # toy_training = [
    #     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    #     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    # ]
    # toy_training_1 = [
    #     "The dog ate the apple".split(),
    #     "Everybody read that book".split()]
    # x, y = next(iter(train_loader))
    # # batch_in = torch.tensor([[[4],
    # #                           [5],
    # #                           [912344],
    # #                           [912344]],
    # #
    # #                          [[6],
    # #                           [912344],
    # #                           [912344],
    # #                           [912344]],
    # #
    # #                          [[7],
    # #                           [8],
    # #                           [9],
    # #                           [10]]])
    # # Y = [["CC", "CD", "DT"], ["EX"], ["JJ", "IN", "JJ", "JJR"]]
    # model = LSTM(nb_layers=nb_layers,
    #              batch_size=batch_size,
    #              nb_lstm_units=nb_lstm_units,
    #              embedding_layer=embedding_layer_const,
    #              bidirectional=False)
    # out = model(x)
    # print(f"out dim {out.size()} \n Out: {out}")
    # # out dim torch.Size([32, 52, 45])
    # # y.shape : [32, 52]
    # loss = model.loss(out, y)
    # print(f"loss: ".format(loss.item()))
