import pandas as pd
import numpy as np
import torch.nn as nn
import torch

##
COMMENT = '0'
train_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_comments.csv')[COMMENT].to_numpy()
test_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_comments.csv')[COMMENT].to_numpy()

train_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_labels.csv').to_numpy()[:,1:]
test_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_labels.csv').to_numpy()[:,1:]


##

def tokenize(s): return s.split()


##


train_comments_split = [tokenize(com) for com in train_comments]
test_comments_split = [tokenize(com) for com in test_comments]

##


vocab = dict()
pos = 0

for comment in train_comments_split:
    for w in comment:
        if w not in vocab:
            vocab[w] = pos
            pos +=1

for comment in test_comments_split:
    for w in comment:
        if w not in vocab:
            vocab[w] = pos
            pos +=1
            
##

vocab_size = len(vocab)
vector_size = 100

embed = nn.Embedding(vocab_size, vector_size)

##
lookup_tensor = torch.tensor([vocab["hello"]], dtype=torch.long)
hello_embed = embed(lookup_tensor)
print(hello_embed)

