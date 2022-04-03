import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class LM_Dataset(Dataset):
    '''
    Initiating Variables
    df: the training dataframe
    source_column : the name of source text column in the dataframe
    target_columns : the name of target text column in the dataframe
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    source_vocab_max_size : max source vocab size
    target_vocab_max_size : max target vocab size
    '''

    def __init__(self, path, pad_token='<pad>'):

        self.Ind2word = None
        self.pad_token = pad_token
        self.indexed_data = []
        self.create_mapping(path)
        self.indices = np.arange(len(self.Ind2word))

    def __len__(self):
        return len(self.indexed_data)

    def create_mapping(self, path):
        corpus_words = [self.pad_token] + ['<s>'] + ['</s>']
        num_corpus_words = 3
        df = pd.read_csv(path)
        lines = df.values
        for line in lines:
            tokenized_line = line[0].strip().split()
            for word in tokenized_line:
                if word not in corpus_words:
                    corpus_words.append(word)
                    num_corpus_words += 1
        Ind2word = dict(enumerate(corpus_words))
        word2Ind = {word: i for i, word in Ind2word.items()}
        for line in lines:
            tokenized_line = line[0].strip().split()
            tokenized_line = ['<s>'] + tokenized_line + ['</s>']
            indexed_tokenized_line = [word2Ind[word] for word in tokenized_line]
            self.indexed_data.append(indexed_tokenized_line)

        self.Ind2word = Ind2word

    def __getitem__(self, index):
        '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values using the vocabulary objects we created in __init__
        '''
        sent_idxs = self.indexed_data[index]
        return sent_idxs