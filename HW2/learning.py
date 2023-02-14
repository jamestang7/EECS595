#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. [Imports](#Imports)
# 2. [Data Read In](#Data-Read-in)

# ## Imports
# [back to top](#Table-of-Contents)

# In[1]:


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

random.seed(SEED
            )
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device('cuda')
# elif torch.has_mps:
#     device = torch.device("mps")
else:
    device = torch.device("cpu")


# In[2]:


# Part 1
def prepare_sequence(seq, to_ix):
    """Input: takes in a list of words, and a dictionary containing the index of the words
    Output: a tensor containing the indexes of the word"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
# This is the example training data
training_data = [
    ("the dog happily ate the big apple".split(), ["DET", "NN", "ADV", "V", "DET", "ADJ", "NN"]),
    ("everybody read that good book quietly in the hall".split(), ["NN", "V", "DET", "ADJ", "NN", "ADV", "PRP", "DET", "NN"]),
    ("the old head master sternly scolded the naughty children for \
     being very loud".split(), ["DET", "ADJ", "ADJ", "NN", "ADV", "V", "DET", "ADJ",  "NN", "PRP", "V", "ADJ", "NN"]),
    ("i love you loads".split(), ["PRN", "V", "PRN", "ADV"])
]
#  These are other words which we would like to predict (within sentences) using the model
other_words = ["area", "book", "business", "case", "child", "company", "country",
               "day", "eye", "fact", "family", "government", "group", "hand", "home",
               "job", "life", "lot", "man", "money", "month", "mother", "food", "night",
               "number", "part", "people", "place", "point", "problem", "program",
               "question", "right", "room", "school", "state", "story", "student",
               "study", "system", "thing", "time", "water", "way", "week", "woman",
               "word", "work", "world", "year", "ask", "be", "become", "begin", "can",
               "come", "do", "find", "get", "go", "have", "hear", "keep", "know", "let",
               "like", "look", "make", "may", "mean", "might", "move", "play", "put",
               "run", "say", "see", "seem", "should", "start", "think", "try", "turn",
               "use", "want", "will", "work", "would", "asked", "was", "became", "began",
               "can", "come", "do", "did", "found", "got", "went", "had", "heard", "kept",
               "knew", "let", "liked", "looked", "made", "might", "meant", "might", "moved",
               "played", "put", "ran", "said", "saw", "seemed", "should", "started",
               "thought", "tried", "turned", "used", "wanted" "worked", "would", "able",
               "bad", "best", "better", "big", "black", "certain", "clear", "different",
               "early", "easy", "economic", "federal", "free", "full", "good", "great",
               "hard", "high", "human", "important", "international", "large", "late",
               "little", "local", "long", "low", "major", "military", "national", "new",
               "old", "only", "other", "political", "possible", "public", "real", "recent",
               "right", "small", "social", "special", "strong", "sure", "true", "white",
               "whole", "young", "he", "she", "it", "they", "i", "my", "mine", "your", "his",
               "her", "father", "mother", "dog", "cat", "cow", "tiger", "a", "about", "all",
               "also", "and", "as", "at", "be", "because", "but", "by", "can", "come", "could",
               "day", "do", "even", "find", "first", "for", "from", "get", "give", "go",
               "have", "he", "her", "here", "him", "his", "how", "I", "if", "in", "into",
               "it", "its", "just", "know", "like", "look", "make", "man", "many", "me",
               "more", "my", "new", "no", "not", "now", "of", "on", "one", "only", "or",
               "other", "our", "out", "people", "say", "see", "she", "so", "some", "take",
               "tell", "than", "that", "the", "their", "them", "then", "there", "these",
               "they", "thing", "think", "this", "those", "time", "to", "two", "up", "use",
               "very", "want", "way", "we", "well", "what", "when", "which", "who", "will",
               "with", "would", "year", "you", "your"]


# In[4]:


word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix.keys():
            word_to_ix[word] = len(word_to_ix)
for word in other_words:
    if word not in word_to_ix.keys():
        word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2, "ADJ": 3, "ADV": 4, "PRP": 5, "PRN": 6}
EMBEDDING_DIM = 64
HIDDEN_DIM = 64


# In[26]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(device)
        self.hidden2tag = nn.Linear(hidden_dim, target_size).to(device)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        sentence.to(device)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out.to(device)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_score = F.log_softmax(tag_space, dim = 1)
        return tag_score


# In[6]:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[11]:


# test a sentence
seq1 = "everybody read the book and ate the food".split()
seq2 = "she like my dog".split()
print("Running a sample tenset \n Sentence:\n {} \n {}".format(" ".join(seq1),
                                                               " ".join(seq2)))
with torch.no_grad():
    for seq in [seq1, seq2]:
        model.to(device)
        inputs = prepare_sequence(seq, word_to_ix).to(device)
        tag_score = model(inputs)
        max_indices = tag_score.max(dim=1)[1]
        ret = []
        # reverse tag_to_ix
        reverse_tag_index = {v: k for k, v in tag_to_ix.items()}
        for i in range(len(max_indices)):
            idx = int(max_indices[i])
            ret.append((seq[i], reverse_tag_index[idx]))
        print(ret)


# In[13]:


# Train
losses = []
model.to(device)
for epoch in range(300):
    count = 0
    sum_loss = 0
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
        targets = prepare_sequence(tags, tag_to_ix).to(device)
        out = model(sentence_in)
        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        count += 1
        sum_loss += loss
        losses.append(sum_loss / count)
    print("Epoch: {}, Loss {}".format(epoch, losses[-1]))
print("Train Finished")


# In[16]:


# predict function
def predict_seq(seq_list, model):
    """

    :param seq_list: list of sequences
    :param model: NN model
    :return: tuple predictions
    """
    # model.to(device)
    with torch.no_grad():
        for seq in seq_list:
            inputs = prepare_sequence(seq, word_to_ix).to(device)
            tags_score = model(inputs)
            max_indices = tags_score.max(dim=1)[1]
            pred = []
            reverse_tag_index = {v: k for k, v in tag_to_ix.items()}
            for i in range(len(max_indices)):
                idx = int(max_indices[i])
                pred.append(reverse_tag_index[idx])
            print("Sequence: {} \n"
              "Tag Prediction: {}\n".format(seq, pred))


# In[17]:


# test on unkown data
predict_seq([seq1, seq2], model)


# ## Data Read in
# [back to top](#Table-of-Contents)

# In[18]:


def split_text(text_file, by_line=False):
    """

    :param by_line: bool, whether to split by lines; if False, split by word
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
# create a list of list of tuples for training data
def combine_lists(vocab_list, tags_list):
    """

    :param vocab_list: list of sentence
    :param tags_list:
    :return: list of list of sentence of words tuples e.g. [[('Pierre', 'NOUN'), ('Vinken', 'NOUN'), (',', '.')]]
    """
    result = []
    for i in range(len(vocab_list)):
        sentence, tags = vocab_list[i], tags_list[i]
        zipped = zip(sentence, tags)
        result.append(list(zipped))
    return result


# In[19]:


vocab_list, tags_list = split_text("wsj1-18.training", by_line=True)
train_list = combine_lists(vocab_list, tags_list)
test_vocab_list, test_tags_list = split_text("wsj19-21.truth", by_line=True)
test_list = combine_lists(test_vocab_list, test_tags_list)


# ### Construct dictionary
# 1. A word/tag dictionary
# 2. A letter/character dictionary
# 3. A POS tag dictionary
# 

# In[20]:


def sequence_to_idx(words, dic_ix):
    """

    :param words: list of words
    :param dic_ix: dictionary with the index as values, word as keys
    :return: list of indices
    """
    return torch.tensor([dic_ix[word] for word in words], dtype=torch.long)


# In[21]:


word_to_idx = {}
tag_to_idx = {}
char_to_idx = {}
for sentence in train_list:
    for word, tag in sentence:
        if word not in word_to_idx.keys():
            word_to_idx[word] = len(word_to_idx)
        if tag not in tag_to_idx.keys():
            tag_to_idx[tag] = len(tag_to_idx)
        for char in word:
            if char not in char_to_idx.keys():
                char_to_idx[char] = len(char_to_idx)
word_vocab_size = len(word_to_idx)
tag_vocab_size = len(tag_to_idx)
char_vocab_size = len(char_to_idx)
for sentence in test_vocab_list:
    for word in sentence:
        if word not in word_to_idx.keys():
            word_to_idx[word] = len(word_to_idx)
print("Unique words: {}".format(len(word_to_idx)))
print("Unique tags: {}".format(len(tag_to_idx)))
print("Unique characters: {}".format(len(char_to_idx)))


# ### Specify hyperparamters

# In[22]:


def get_accuracy(model, if_train):
    if if_train:
        data = list(zip(vocab_list, tags_list))
    else:
        data = list(zip(test_vocab_list, test_tags_list))
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_tuple in data:
            # get X in a list by unzipping the input tuple list
            X = input_tuple[0]
            # get Y similarly
            y = input_tuple[1]
            # convert into index
            X = prepare_sequence(X, word_to_idx).to(device)
            y = prepare_sequence(y, tag_to_idx).to(device)
            # forward model
            out = model(X)
            max_indices = out.max(dim=1)[1]
            total += len(y)
            # because prepare sequence output long type tensor
            correct = torch.eq(max_indices, y).sum().item()
        return correct / total


# In[23]:


WORD_EMBEDDING_DIM = 1024
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 1024
CHAR_HIDDEN_DIM = 1024
EPOCHS = 100
lr = 1e-3


# In[31]:


debug = True
def train(model, lr, epochs=EPOCHS):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.NLLLoss()
    data = list(zip(vocab_list, tags_list))
    # init losses,
    losses, train_acc, test_acc = [], [], []
    for epoch in range(epochs):
        iters = 0
        sum_loss = 0
        for batch_id, (X, y) in enumerate(data):
            if debug: print('Beginning Reading Batch: {}'.format(batch_id))
            X = prepare_sequence(X, word_to_idx).to(device)
            y = prepare_sequence(y, tag_to_idx).to(device)
            model.to(device)
            # forward model pass
            out = model(X)
            loss = loss_function(out, y) # compute the loss
            loss.backward() # backward pass
            optimizer.step() # make the update to each parameter
            optimizer.zero_grad()

            # save result
            if debug: print('Calculated Loss')
            sum_loss += loss
            iters += 1
            train_acc.append(get_accuracy(model, True))
            if debug: print('Calculated Train Acc')
            test_acc.append(get_accuracy(model, False))
            if debug: print('Calculated Test Acc')
            losses.append(sum_loss / iters)
            # if batch_id % 100 == 0:
            print("Epoch: {}, Batch: {} \n"
                      "Loss{}, Train Accuracy{:.2%}, Test Accuracy:{:.2%}".format(
                    epoch, batch_id, losses[-1], train_acc[-1], test_acc[-1]
                ) )
        # plotting
        plt.title("Training Curve")
        plt.plot(np.arange(len(losses)), losses, label="Loss")
        plt.plot(np.arange(len(losses)), train_acc, linestyle='-.', label="Train")
        plt.plot(np.arange(len(losses)), test_acc, linestyle='-.', label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/ Accuracy")
        plt.legend(loc="best")
        plt.show()
    return model
    print("Finished Training\n" + "-" * 50)


# In[ ]:


model = LSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, word_vocab_size, tag_vocab_size)
out_model = train(model, lr)


# In[29]:




