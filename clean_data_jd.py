import pandas as pd
import numpy as np
import re


def clean_data(input_path):
    data = pd.read_csv(input_path + '/train.csv')

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
        lambda x: x = x.replace('"', ''))
    data['comment_text'] = data['comment_text'].map(
        lambda x: x = x.strip('\"'))
    data['comment_text'] = data['comment_text'].map(
        lambda x: re.sub('"', '', str(x)))

    COMMENT = 'comment_text'
    LABELS = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    comments = data[[COMMENT]]
    labels = data[LABELS]

    data['extract'] = data[LABELS].astype('str').agg(' '.join, axis=1)
    data['extract'] = data['extract'] + '\t' + data['comment_text']

    data['extract'].to_csv(input_path + '/train_edited.csv',
                           header=False,
                           index=False)

    assert (len(comments)) == len(labels)
    comments.to_csv(input_path + '/train_comments.csv', header=True)
    labels.to_csv(input_path + '/train_labels.csv')
