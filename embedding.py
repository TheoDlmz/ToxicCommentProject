import torch
import numpy as np


def get_glove_embeddings():
    embedding_path = './data/embeddings/glove.6B.200d.txt'
    lines = []
    with open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        if not len(emb) == 200:
            continue
        vector = [float(x) for x in emb]
        if indx == 0:
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word_to_indx[word] = indx + 1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)

    return embedding_tensor, word_to_indx


def get_indices_tensor(text_arr, word_to_indx, max_length):
    '''
    -text_arr: array of word tokens
    -word_to_indx: mapping of word -> index
    -max length of return tokens
    returns tensor of same size as text with each words corresponding
    index
    '''
    nil_indx = 0
    text_indx = [
        word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr
    ][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend(
            [nil_indx for _ in range(max_length - len(text_indx))])

    x = torch.LongTensor([text_indx])

    return x
