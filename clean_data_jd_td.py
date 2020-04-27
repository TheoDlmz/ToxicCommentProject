import collections
import numpy as np
import pandas as pd
import re

##
from argparse import Namespace

args = Namespace(
    path_csv="C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\unclean\\",
    train_proportion=0.7,
    val_proportion=0.2,
    test_proportion=0.1,
    path_out="C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\",
    seed=1337
)
##


def clean_data(input_path):
    data = pd.read_csv(input_path + 'train.csv')

    # Remove return to line
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub('\\n', ' ', str(x)))

    # Remove any text starting with user
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub('\[\[User.*', '', str(x)))

    # convert to lowercase
    data['comment_text'] = data['comment_text'].map(lambda x: str(x).lower())

    # remove http links in the text
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub("(http://.*?\s)|(http://.*)", '', str(x)))

    # Flag empty comments with na and remove from data
    data['comment_text'].replace('', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Other
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'s", " \'s", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'m", " \'m", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'ve", " \'ve", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"n\'t", " n\'t", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'re", " \'re", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'d", " \'d", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\'ll", " \'ll", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r",", " , ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"!", " ! ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\(", " ( ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\)", " ) ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\?", " ? ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub(r"\s{2,}", " ", str(x)))
    data['comment_text'] = data['comment_text'].map(
        lambda x: x.replace('"',""))
    data['comment_text'] = data['comment_text'].map(
        lambda x: x.strip('\"'))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub('"', '', str(x)))
   
    data = data.drop('id',1)

    COMMENT = 'comment_text'
    LABELS = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    
    data['clean'] = (data['toxic'] + data['severe_toxic']+data['obscene']+data['threat'] +data['insult']+data['identity_hate']) == 0
    '''
    comments = data[[COMMENT]]
    labels = data[LABELS]

    data['extract'] = data[LABELS].astype('str').agg(' '.join, axis=1)
    data['extract'] = data['extract'] + '\t' + data['comment_text']

    data['extract'].to_csv(input_path + '/train_edited.csv',
                           header=False,
                           index=False)
    
    data['overall'] = sum(data[LABELS])
    assert (len(comments)) == len(labels)
    comments.to_csv(input_path + '/train_comments.csv', header=True)
    labels.to_csv(input_path + '/train_labels.csv')
    '''
    by_toxicity = collections.defaultdict(list)
    for _, row in data.iterrows():
        by_toxicity[row.clean].append(row.to_dict())
        
        
    # Create split data
    final_list = []
    np.random.seed(args.seed)
    
    for _, item_list in sorted(by_toxicity.items()):
    
        np.random.shuffle(item_list)
        
        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)
        n_test = int(args.test_proportion * n_total)
        
        # Give data point a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'
        
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        
        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
            
        # Add to final list
        final_list.extend(item_list)
    
    data = pd.DataFrame(final_list)
    data.to_csv(args.path_out+"toxic_comments.csv", index=False)
    return data



data = clean_data(args.path_csv)