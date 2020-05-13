#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer import CharacterTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np


class KORTypoDataset(Dataset):

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
