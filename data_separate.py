import pandas as pd
import numpy as np
import torch.nn as nn
import torch

##
COMMENT = 'comment_text'
comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\comments.csv')[COMMENT].to_numpy()

labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\labels.csv').to_numpy()[:,1:]

##

good_comments = []
bad_comments = []
bad_labels = []

for i in range(len(comments)):
    if sum(labels[i]) == 0:
        good_comments.append(comments[i])
    else:
        bad_comments.append(comments[i])
        bad_labels.append(labels[i])

good_comments = pd.DataFrame(good_comments,columns=["comment_text"])
bad_comments = pd.DataFrame(bad_comments,columns=["comment_text"])
bad_labels = pd.DataFrame(bad_labels,columns= ['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
##


good_comments.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\good_comments.csv')
bad_comments.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\bad_comments.csv')
bad_labels.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\bad_labels.csv')