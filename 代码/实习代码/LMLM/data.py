#!/usr/bin/python3

from __future__ import unicode_literals, division
import numpy as np
import os
import re
import sys
import codecs
import random
import collections
import operator
import numpy
from vocab import Vocab

BEGIN_TOKEN = "<s>"
END_TOKEN = "</s>"


class TextLoader:
    def __init__(self, args, train=True):
        self.batch_size = args['batch_size']
        self.num_steps = args['num_steps']
        self.vocab = Vocab(args['vocab_file'])
        if train:
            self._i = 0
            self.train_data = self.read_dataset(args['train_file'], 0)
            self._nids_i = len(self.train_data)
            self._d = 0
            self.dev_data = self.read_dataset(args['dev_file'], 0)
            self._nids_d = len(self.dev_data)

    def read_dataset(self, filename, flag):
        """
        Read dataset from a file
        """
        print("read dataset from {}...".format(filename))
        data = list()
        with open(filename, 'r') as f:
            tmp = f.readlines()
            if flag != 0:
                print("shuffle")
                random.shuffle(tmp)
        for line in tmp:
            line = line.strip()
            data.append(self.vocab.word2id(BEGIN_TOKEN))
            for word in line.split()[1:]:
                word = word.lower()
                data.append(self.vocab.word2id(word))
            data.append(self.vocab.word2id(END_TOKEN))
        print("read data size %d" % len(data))
        return data

    def read_nbestdata(self, filename):
        """
        Read nbest data from a file
        """
        data = list()
        with open(filename, 'r') as f:
            tmp = f.readlines()
        for line in tmp:
            
            sent = list()
            sent.append(BEGIN_TOKEN)  #add sentence start flag
            line = line.strip()
            for word in line.split()[1:-1]:
                sent.append(word)
            sent.append(END_TOKEN)
            
            data.append(self.vocab.data_to_word_ids(sent))
        print("read nbest data size: ", len(data))
        return data

    def read_data(self, filename):
        data = list()
        with open(filename, 'r') as f:
            tmp = f.readlines()
        for line in tmp:
            line = line.strip()
            data.append(self.vocab.word2id(BEGIN_TOKEN))
            for word in line.split()[1:-1]:
                word = word.lower()
                data.append(self.vocab.word2id(word))
            data.append(self.vocab.word2id(END_TOKEN))
        print("read nbest data size %d" % len(data))
        return data
            
        return data
    def get_test_sentence(self):
        while True:
            if self._d == self._nids_d:
                raise StopIteration
            ret = self.dev_data[self._d]
            self._d += 1
            yield ret

    def get_train_sentence(self):
        while True:
            if self._i == self._nids_i:
               raise StopIteration 
            ret = self.train_data[self._i]
            self._i += 1
            yield ret

    def data_iterator(self, raw_data, batch_size, num_steps, biderectional=False):
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = []
        for i in range(batch_size):
            x = raw_data[batch_len * i:batch_len * (i + 1)]
            data.append(x)
        if biderectional:
            epoch_size = (batch_len - 2) // num_steps
        else:
            epoch_size = (batch_len - 1) // num_steps
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
        for i in range(epoch_size):
            xs = list()
            ys = list()
            zs = list()
            for j in range(batch_size):
                x = data[j][i*num_steps: (i+1)*num_steps]
                y = data[j][i*num_steps+1: (i+1)*num_steps+1]
                xs.append(x)
                ys.append(y)
                if biderectional:
                    z = data[j][i*num_steps+2: (i+1)*num_steps+2]
                    zs.append(z)
            sample = (xs, ys, zs)
            yield sample

    def nbest_data_iterator(self, data, batch_size, biderectional=False):
        data_size = len(data)
        epoch_size = data_size // batch_size
        i = data_size % batch_size
        if i > 0:
            epoch_size += 1
            for j in range(batch_size - i):
                index = j % data_size
                sent = list(data[index])
                data.append(sent)
        for i in range(epoch_size):
            xs = list()
            ys = list()
            for j in range(batch_size):
                if biderectional:
                    x = data[j + i * batch_size][:-2]
                    y = data[j + i * batch_size][2:]
                    xs.append(x)
                    ys.append(y)
                else:
                    x = data[j + i * batch_size][:-1]
                    xs.append(x)
            sample = (xs, ys)
            yield sample
            