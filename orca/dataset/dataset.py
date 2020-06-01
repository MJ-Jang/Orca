#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer import CharacterTokenizer
from orca.utils.noise_functions import noise_maker

from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random
import re


class TypoDetectionSentenceLevelDataset(Dataset):

    def __init__(self,
                 sents: list,
                 typo_num: int = 2,
                 max_word_len: int = 20,
                 max_sent_len: int = 20,
                 ignore_idx: int = 2
                 ) -> None:

        self.tokenizer = CharacterTokenizer()
        self.vocab_size = len(self.tokenizer)
        self.data = sents

        self.typo_nu = typo_num
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        self.pad_id = self.tokenizer.pad_id
        self.ignore_idx = ignore_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sent = self.data[item]
        words = sent.split(' ')
        inputs, targets = [], []

        for word in words:
            y = 0
            prob_ = random.random()
            if prob_ <= 0.5:
                if prob_ <= 0.3:
                    _, word = noise_maker(word, 1.0, 1, method='ct')
                else:
                    _, word = noise_maker(word, 1.0, self.typo_nu, method='g')
                y = 1
            token = self.tokenizer.text_to_idx(word)
            if len(token) <= self.max_word_len:
                token += [self.pad_id] * (self.max_word_len - len(token))
            else:
                token = token[:self.max_word_len]
            inputs.append(token)
            targets.append(y)
        if len(inputs) < self.max_sent_len:
            inputs += [[self.pad_id] * self.max_word_len] * (self.max_sent_len - len(inputs))
            targets += [self.ignore_idx] * (self.max_sent_len - len(targets))
        else:
            inputs = inputs[:self.max_sent_len]
            targets = targets[:self.max_sent_len]

        return np.array(inputs), np.array(targets)
