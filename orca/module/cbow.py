#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, fc_dim):
        super(CBOW, self).__init__()
        self.pe = PositionalEncoding(embedding_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.proj_1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj_2 = nn.Linear(embedding_dim, embedding_dim)
        self.proj_3 = nn.Linear(embedding_dim, fc_dim)

        self.output = nn.Linear(fc_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs) # (N * S * E)
        embeds += self.pe(embeds) # (N * S * E)
        embeds = embeds.sum(dim=1)

        out = F.relu(self.proj_1(embeds))
        out = F.relu(self.proj_2(out))
        out = F.relu(self.proj_3(out))

        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob
