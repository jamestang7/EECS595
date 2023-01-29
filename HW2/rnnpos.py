import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F


def load_embedding(filename='glove.6B.50d.txt'):
    """
    Load embedding for training
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
    df.set_index('token', inplace=True)
    df.to_pickle("glove.pkl")
    return df, words


def word_to_embedding(target_vocab, pre_train):
    """

    :param pre_train: pd.DataFrame pre-trained dataframe
    :param target_vocab: list/ array of tokens need to be transformed
    :return: transformed matrix
    """
    matrix_len = len(target_vocab)
    weighted_matrix = np.zeros((matrix_len, 50))
    words_found = 0
    for i, word in enumerate(target_vocab):
        try:
            weighted_matrix[i] = pre_train.loc[word]
            words_found += 1
        except KeyError:
            weighted_matrix[i] = np.random.normal(size=50)
    print("{0:d} words found, {1:d} words randomized".format(words_found,
                                                             matrix_len - words_found))
    return weighted_matrix


def create_emb_layer(weighted_matrix, non_trainable=False):
    """

    :param weighted_matrix: tensor matrix
    :param non_trainable:
    :return:
    """
    input_shape, embedding_dim = weighted_matrix.size()
    emb_layer = nn.Embedding(input_shape, embedding_dim)
    emb_layer.load_state_dict({'weight': weighted_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, input_shape, embedding_dim


def split_text(text_file):
    """

    :param text_file: training file
    :return: dictionary with tokens and its corresponding tags
    """
    with open(text_file,mode = "r") as file:
        text_f = file.read()
        text_f_lst = text_f.split()
        file.close()
    keys, values = text_f_lst[::2], text_f_lst[1::2]
    result_dic = dict(zip(keys,values))
    return result_dic



class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores


def train(training_file):
    assert os.path.isfile(training_file), 'Training file does not exist'

    # Your code starts here

    model = RNNTagger(1, 1, 10)  # Replace with code that actually trains a model

    # Your code ends here

    return model


def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), 'Model file does not exist'
    assert os.path.isfile(data_file), 'Data file does not exist'
    assert os.path.isfile(label_file), 'Label file does not exist'

    # Your code starts here

    model = RNNTagger(1, 1, 10)
    model.load_state_dict(torch.load(model_file))

    prediction = model(torch.rand(1000, 1, 1))  # replace with inference from the loaded model
    prediction = torch.argmax(prediction, -1).cpu().numpy()

    ground_truth = [random.randint(0, 10) for _ in range(1000)]  # replace with actual labels from the data files

    # Your code ends here

    print(f'The accuracy of the model is {100 * accuracy_score(prediction, ground_truth):6.2f}%')


def main(params):
    if params.train:
        model = train(params.training_file)
        torch.save(model.state_dict(), params.model_file)
    else:
        test(params.model_file, params.data_file, params.label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument('--model_file', type=str, default='model.torch')
    parser.add_argument('--training_file', type=str, default='')
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')

    main(parser.parse_args())
