import numpy as np
import torch
from torch.utils.data import Dataset


class Train_Dataset(Dataset):
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

    def __init__(self, path, window_size, k=5):

        self.x_data = None
        self.y_data = None
        self.Ind2word = None
        self.create_mapping(path, window_size)
        self.k = k
        self.indices = np.arange(len(self.Ind2word))

    def __len__(self):
        return len(self.Ind2word)

    def create_mapping(self, path, window_size):
        mapping = []
        corpus_words = ['<Start>'] + ['<End>'] + []
        num_corpus_words = 2
        with open(path, 'r') as f:
            tokenized_line = f.read().split()
            for word in tokenized_line:
                if word not in corpus_words:
                    corpus_words.append(word)
                    num_corpus_words += 1
            Ind2word = dict(enumerate(corpus_words))
            word2Ind = {word: i for i, word in Ind2word.items()}
            tokenized_line = ['<Start>'] + tokenized_line + ['<End>']
            for i, c_word in enumerate(tokenized_line):
                other_words = tokenized_line[max(i - window_size, 0): i] + tokenized_line[i + 1: i + 1 + window_size]
                for o_word in other_words:
                    mapping.append(tuple([word2Ind[c_word], word2Ind[o_word]]))

        self.x_data, self.y_data = zip(*mapping)
        self.Ind2word = Ind2word

    def get_negative_samples(self, outsideWordIdx):
        """ Samples K indexes which are not the outsideWordIdx """
        negSampleWordIndices = [None] * self.k
        for i in range(self.k):
            newidx = np.random.choice(self.indices)
            while newidx in [outsideWordIdx, 0, 1]:
                newidx = np.random.choice(self.indices)
            negSampleWordIndices[i] = newidx
        return negSampleWordIndices

    def __getitem__(self, index):
        '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values using the vocabulary objects we created in __init__
        '''
        c_idx = self.x_data[index]
        o_idx = self.y_data[index]
        neg_idx_samples = self.get_negative_samples(o_idx)

        return c_idx, o_idx, torch.LongTensor(neg_idx_samples)