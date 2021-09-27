#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:45:03 2021

@author: jamestang1
"""

import numpy as np
import pandas as pd
import os
import re

print(os.getcwd())
path = '/Users/jamestang1/Dropbox (University of Michigan)/UM James/um/EECS 595'
training_path = path + '/HW1_data/training'
def train():
    # 1) when word occurs and c = 1
    # 2) when word doesn't occur and c=1
    # 3) when word occurs and c = -1
    # 4) when word doesn't occur and c=-1
    # TODO: implement data loading, preprocessing, and model training
    # Set training and testing path
    training_path = path + '/HW1_data/training'
    for file in os.listdir(training_path):
        if file == 'pos':
            pos_train_file = [txts for txts in os.listdir(training_path+ '/'+file)]
            
        elif file =='neg':
            neg_train_file = [txts for txts in os.listdir(training_path+ '/'+file)]
    data = ''
    os.chdir(training_path+'/'+'pos')
    for txt_name in pos_train_file:
        with open(txt_name,encoding = 'windows-1252') as infile:
            data += infile.read()
            data += "\n"
    with open ('_.txt', 'w') as fp:
        fp.write(data)
    file = open(training_path+'/pos/'+'_.txt',"r")
    pos_txt = file.read()

    #chaning directory to negative folder
    os.chdir(training_path+'/'+'neg')
    data = ''
    for txt_name in neg_train_file:
        with open(txt_name,encoding = 'windows-1252') as infile:
            data += infile.read()
            data += "\n"
    with open ('_.txt', 'w') as fp:
        fp.write(data)
    file = open(training_path+'/neg/'+'_.txt',"r")
    neg_txt = file.read()
    return pos_txt, neg_txt
#b
def count_word(word,pos_txt,neg_txt):
    word_1 = pos_txt.count(word)
    pos_total = len(re.findall(r'\w+',pos_txt))
    word_2 = pos_total - word_1
    word_3 = neg_txt.count(word)
    neg_total = len(re.findall(r'\w+',neg_txt))
    word_4 = neg_total - word_3
    return word_1, word_2, word_3, word_4
dic = {}
pos_txt, neg_txt = train()
for word in ['the','like','good','movie']:
    dic[word] = tuple(count_word(word,pos_txt,neg_txt))
print(dic)
# c)
# calculate H(w)
def H_I(word):
    """
    

    Parameters
    ----------
    word : str
        word

    Returns
    -------
    Entropy H(w) , mutual information I(w|c)

    """
    tup = dic[word]
    total_occurence = tup[0] + tup[2]
    total = np.sum(tup)
    p = total_occurence/total
    H_w =   - p * np.log2(p) - (1-p)* np.log2(1-p)
    p_c = (tup[0] + tup[1])/total
    H_c = -p_c * np.log2(p_c) - (1-p_c) * np.log2(1-p_c)
    p_array = np.array(tup)/np.sum(tup)
    H_joint = np.sum(p_array * np.log2(1/p_array))
    return H_w + H_c - H_joint
for word in ['the','like','good','movie']:
    print(word, 'mutual information :', np.around(H_I(word),10))
def H_I_new(word,dic):
    """
    

    Parameters
    ----------
    word : str
        word

    Returns
    -------
    Entropy H(w) , mutual information I(w|c)

    """
    tup = dic[word]
    total_occurence = tup[0] + tup[2]
    total = np.sum(tup)
    p = total_occurence/total
    H_w =   - p * np.log2(p) - (1-p)* np.log2(1-p)
    p_c = (tup[0] + tup[1])/total
    H_c = -p_c * np.log2(p_c) - (1-p_c) * np.log2(1-p_c)
    p_array = np.array(tup)/np.sum(tup)
    H_joint = np.sum(p_array * np.log2(1/p_array))
    return H_w + H_c - H_joint
unique = []
total_txt = pos_txt + neg_txt
for word in total_txt.split():
    if word not in unique:
        unique.append(word)
unique.sort()
dic_total = {}
df = pd.DataFrame()
for word in unique:
    dic_total[word] = tuple(count_word(word,pos_txt,neg_txt))
df['word'] = unique
df['w1'] = [x[0] for x in dic_total.values()]
df['w2'] = [x[1] for x in dic_total.values()]  
df['w3'] = [x[2] for x in dic_total.values()]
df['w4'] = [x[3] for x in dic_total.values()]  
pos_total = len(re.findall(r'\w+',pos_txt))  
neg_total = len(re.findall(r'\w+',neg_txt))
total_count = pos_total + neg_total
p_series = (df['w1'] + df['w3'])/total_count
H_w = - p_series * np.log2(p_series) - (1-p_series)* np.log2(1-p_series)
df['H_w'] = H_w
p_series = (df['w1'] + df['w2'])/total_count
H_c = - p_series * np.log2(p_series) - (1-p_series)* np.log2(1-p_series)
df['H_c'] = H_c

df.head()
p_array = np.array(df.loc[:,'w1':'w4'])/total_count
log_array = np.log2(1/p_array)
df['I(w,c)'] = np.array(df['H_w']) + np.array(df['H_c']) - np.sum(p_array * log_array, axis = 1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
