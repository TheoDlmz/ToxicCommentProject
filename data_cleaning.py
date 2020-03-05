import pandas as pd
import numpy as np

import re
##

data = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\unclean\\train.csv')

##

    data['comment_text'] = data['comment_text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
    # remove any text starting with User... 
    data['comment_text'] = data['comment_text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
    # remove IP addresses or user IDs
    data['comment_text'] = data['comment_text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
    # lower uppercase letters
    data['comment_text'] = data['comment_text'].map(lambda x: str(x).lower())
    
    #remove http links in the text
    data['comment_text'] = data['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
    
    #remove all punctuation except for apostrophe (')
    data['comment_text'] = data['comment_text'].map(lambda x: re.sub('["#$%&\()*+,-.!?/:;<=>@[\\]^_`{|}~]','',str(x)))


##

data['comment_text'].replace('', np.nan, inplace=True)

##

data.dropna(inplace=True)

##

COMMENT = 'comment_text'
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
comments = data[[COMMENT]]
labels = data[LABELS]

assert(len(comments) == len(labels))


##

n = len(comments)
n_train =int(2*n/3)
a = np.arange(n)
np.random.shuffle(a)
train_index,test_index = np.split(a,[n_train])
train_comments =  pd.DataFrame(comments.to_numpy()[train_index])
test_comments = pd.DataFrame(comments.to_numpy()[test_index])
train_labels = pd.DataFrame(labels.to_numpy()[train_index])
test_labels = pd.DataFrame(labels.to_numpy()[test_index])


##


test_comments.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_comments.csv')

train_comments.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_comments.csv')

test_labels.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_labels.csv')
train_labels.to_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_labels.csv')