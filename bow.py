import pandas as pd
import numpy as np
COMMENT = '0'
train_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_comments.csv')[COMMENT].to_numpy()
test_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_comments.csv')[COMMENT].to_numpy()

train_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_labels.csv').to_numpy()[:,1:]
test_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_labels.csv').to_numpy()[:,1:]

##

for comments in train_comments,test_comments:
    # remove '\\n'
    comments = comments['comment_text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
    # remove any text starting with User... 
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
    # remove IP addresses or user IDs
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
    # lower uppercase letters
    comments['comment_text'] = comments['comment_text'].map(lambda x: str(x).lower())
    
    #remove http links in the text
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
    
    #remove all punctuation except for apostrophe (')
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','',str(x)))


##

import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


##
#This Tokenize the text


train_comments = [tokenize(com) for com in train_comments]
test_comments = [tokenize(com) for com in test_comments]
##



##

import torch.nn as nn

embed = nn.Embedding(vocab_size, vector_size)

# intialize the word vectors, pretrained_weights is a 
# numpy array of size (vocab_size, vector_size) and 
# pretrained_weights[i] retrieves the word vector of
# i-th word in the vocabulary
embed.weight.data.copy_(torch.fromnumpy(pretrained_weights))

# Then turn the word index into actual word vector
vocab = {"some": 0, "words": 1}
word_indexes = [vocab[w] for w in ["some", "words"]] 
word_vectors = embed(word_indexes)