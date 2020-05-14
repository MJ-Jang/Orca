#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer import CharacterTokenizer
from orca.utils.noise_functions import noise_maker

from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np


class CBOWTypoDataset(Dataset):

    def __init__(self,
                 sents: list,
                 window_size: int
                 ) -> None:
        assert window_size > 0

        self.tokenizer = CharacterTokenizer()
        self.vocab_size = len(self.tokenizer)
        self.data = []

        for s in tqdm(sents):
            token = self.tokenizer.text_to_idx(s)
            token = [self.tokenizer.pad_id] * window_size + token + [self.tokenizer.pad_id] * window_size
            for i in range(window_size, len(token) - window_size):
                context = [token[i - 2], token[i - 1],
                           token[i + 1], token[i + 2]]
                target = token[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, tgt = self.data[idx][0], self.data[idx][1]
        return np.array(inp), tgt


class TextCNNDataset(Dataset):

    def __init__(self,
                 sents: list,
                 max_len: int = 150,
                 threshold: float = 0.3,
                 noise_char_ratio: float = 0.05):

        self.tokenizer = CharacterTokenizer()
        self.vocab_size = len(self.tokenizer)
        self.data = sents

        self.max_len = max_len
        self.threshold = threshold
        self.noise_char_ratio = noise_char_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sent = self.data[item]

        targets = self.tokenizer.text_to_idx(sent)
        inputs, _ = noise_maker(sent, self.threshold, self.noise_char_ratio)
        tok_to_id_dict = self.tokenizer.tok_to_id_dict
        inputs = [tok_to_id_dict[v] if v in list(tok_to_id_dict.keys()) else self.tokenizer.unk_id for v in inputs]

        assert len(targets) == len(inputs)

        if len(targets) <= self.max_len:
            targets += [self.tokenizer.pad_id] * (self.max_len - len(targets))
            inputs += [self.tokenizer.pad_id] * (self.max_len - len(inputs))
        else:
            print(self.max_len)
            targets = targets[:self.max_len]
            inputs = inputs[:self.max_len]

        return np.array(inputs), np.array(targets)
