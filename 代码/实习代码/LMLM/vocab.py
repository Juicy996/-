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


UNK_TOKEN = "<unk>"
BEGIN_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"


class Vocab:
    def __init__(self, vocab_file):
        self.word_to_id, self.id_to_word = self.read_vocab(vocab_file)
        self.size = len(self.word_to_id)

    def read_vocab(self, vocab_file):
        """
        read vocabulary from vocab_file
        :param vocab_file:
        :return: dict(word to id)
        """
        id_to_word = [PAD_TOKEN]
        item_to_id = dict()
        if BEGIN_TOKEN not in item_to_id:
            item_to_id[BEGIN_TOKEN] = len(item_to_id) + 1
            id_to_word.append(BEGIN_TOKEN)
        if END_TOKEN not in item_to_id:
            item_to_id[END_TOKEN] = len(item_to_id) + 1
            id_to_word.append(END_TOKEN)
        with open(vocab_file, 'r') as fin:
            for line in fin:
                word = line.strip()
                if word not in item_to_id:
                    item_to_id[word] = len(item_to_id) + 1
                    id_to_word.append(word)
        if UNK_TOKEN not in item_to_id:
            item_to_id[UNK_TOKEN] = len(item_to_id) + 1
            id_to_word.append(UNK_TOKEN)
        return item_to_id, id_to_word

    def word2id(self, word):
        """
        convert word into its id
        """
        if word in self.word_to_id:
            return self.word_to_id[word]
        else:
            return self.word_to_id[UNK_TOKEN]

    def data_to_word_ids(self, input_data):
        """
        Given a list of words, convert each word into it's word id
        :param input_data: a list of words
        :return: a list of word ids
        """
        _buffer = list()
        for word in input_data:
            _buffer.append(self.word2id(word))
        return _buffer

    def data_to_ids_word(self, input_data):
        """
        Given a list of ids, convert each id into it's word
        :param input_data: a list of ids
        :return: a list of words
        """
        _buffer = list()
        for idx in input_data:
            _buffer.append(self.id_to_word[idx])
        return _buffer

