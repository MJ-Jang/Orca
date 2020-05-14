# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.module import CBOW, TextCNN
from orca.tokenizer import CharacterTokenizer
from orca.dataset import CBOWTypoDataset, TextCNNDataset
from orca.abstract import TypoCorrecter

import torch
import torch.nn as nn
import torch.optim as optim
import os
import dill
import numpy as np
import math

from torch.utils.data import DataLoader
from tqdm import tqdm


class TextCNNTypoCorrector(TypoCorrecter):

    def __init__(self, embedding_dim: int, hidden_dim: int, use_gpu: bool = True, **kwargs):
        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

        self.tokenizer = CharacterTokenizer()
        vocab_size = len(self.tokenizer)

        self.model_conf = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim
        }
        self.model = TextCNN(**self.model_conf).to(self.device)

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str,
              **kwargs
              ):

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = TextCNNDataset(sents, kwargs['max_len'], kwargs['threshold'], kwargs['noise_char_ratio'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        loss_function = nn.NLLLoss()
        best_loss = 1e5

        for epoch in range(num_epochs):
            total_loss = 0
            for context, target in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                context = context.to(self.device)
                target = target.to(self.device)

                nll_prob = self.model(context)
                loss = loss_function(nll_prob, target)

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item() / batch_size
                if total_loss <= best_loss:
                    best_loss = total_loss
                    self.save_dict(save_path=save_path, model_prefix=model_prefix)
            print("| Epochs: {} | Training loss: {} |".format(epoch + 1, round(total_loss, 4)))

    def infer(self, sent: list, **kwargs):
        pass

###
class CBOWTypoCorrector:
    def __init__(self, embedding_dim: int, fc_dim: int, use_gpu: bool = False):
        super(CBOWTypoCorrector, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

        self.tokenizer = CharacterTokenizer()
        vocab_size = len(self.tokenizer)

        self.model_conf = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'fc_dim': fc_dim
        }
        self.model = CBOW(**self.model_conf).to(self.device)

    def train(self,
              sents: list,
              window_size: int,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str):

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        dataset = CBOWTypoDataset(sents, window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        loss_function = nn.NLLLoss()
        best_loss = 1e5

        for epoch in range(num_epochs):
            total_loss = 0
            for context, target in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                context = context.to(self.device)
                target = target.to(self.device)

                nll_prob = self.model(context)
                loss = loss_function(nll_prob, target)

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item() / batch_size
                if total_loss <= best_loss:
                    best_loss = total_loss
                    self.save_dict(save_path=save_path, model_prefix=model_prefix)
            print("| Epochs: {} | Training loss: {} |".format(epoch + 1, round(total_loss, 4)))

    def save_dict(self, save_path: str, model_prefix: str):
        os.makedirs(save_path, exist_ok=True)

        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'model_params': self.model.cpu().state_dict(),
            'model_conf': self.model_conf,
            'model_type': 'pytorch'
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)
        self.model.to(self.device)

    def load_model(self, model_path: str):
        if model_path:
            with open(model_path, 'rb') as modelFile:
                model_dict = dill.load(modelFile)
            self.model_conf = model_dict['model_conf']
            self.model = CBOW(**self.model_conf)
            self.model.load_state_dict(model_dict["model_params"])
            self.model.to(self.device)

    def infer(self, text: str, window_size: int = 5, threshold: float = 0.5):
        self.model.eval()
        softmax = torch.nn.Softmax(dim=-1)

        token = self.tokenizer.text_to_idx(text)
        token = [self.tokenizer.pad_id] * window_size + token + [self.tokenizer.pad_id] * window_size
        token_reversed = token[::-1]

        # target = self.tokenizer.text_to_token(text)
        # target = [self.tokenizer.pad_id] * window_size + target + [self.tokenizer.pad_id] * window_size

        max_i = math.ceil(np.median(range(window_size, len(token) - window_size)))
        for i in range(window_size, len(token) - window_size):
            # forward
            context = [token[i - 2], token[i - 1],
                       token[i + 1], token[i + 2]]
            context = torch.LongTensor([context]).to(self.device)

            logits = self.model(context)
            probs = softmax(logits)
            pred, prob = int(probs.argmax()), float(probs[0][int(probs.argmax())])
            target = token[i]

            if pred != target:
                if prob >= threshold:
                    token[i] = pred
                    token_reversed[-i] = pred
            #
            # # backward
            # context_re = [token_reversed[i - 2], token_reversed[i - 1],
            #               token_reversed[i + 1], token_reversed[i + 2]]
            # context_re = torch.LongTensor([context_re]).to(self.device)
            #
            # logits = self.model(context_re)
            # probs = softmax(logits)
            # pred_re, prob_re = int(probs.argmax()), float(probs[0][int(probs.argmax())])
            # target_re = token_reversed[i]
            #
            # if pred_re != target_re:
            #     if prob_re >= threshold:
            #         token_reversed[i] = pred_re
            #         token[-i] = pred_re
            # if i == max_i:
            #     break
        token = token[window_size:-window_size]
        return self.tokenizer.idx_to_text(token)
