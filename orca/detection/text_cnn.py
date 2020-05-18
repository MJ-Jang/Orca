# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.module import TextCNN
from orca.tokenizer import JasoTokenizer, CharacterTokenizer
from orca.dataset import TypoDetectionDataset
from orca.abstract import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill


from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict


class TextCNNTypoDetector(Module):

    def __init__(self, embedding_dim: int, hidden_dim: int, n_class: int,
                 use_gpu: bool = True, **kwargs):
        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.tokenizer = CharacterTokenizer()
        vocab_size = len(self.tokenizer)
        self.pad_id = self.tokenizer.pad_id

        self.model_conf = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'n_class': n_class
        }
        self.model = TextCNN(**self.model_conf).to(self.device)
        if self.n_gpu == 1:
            pass
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str,
              **kwargs
              ):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = TypoDetectionDataset(sents, kwargs['max_len'], kwargs['typo_num'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        best_loss = 1e5

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = []
            for context, target in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                context = context.to(self.device)
                target = target.to(self.device)

                logits = self.model(context)
                loss = F.cross_entropy(logits, target, ignore_index=self.pad_id)
                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()

                _, pred = logits.max(dim=1)
                acc = [1 if pred[i] == target[i] else 0 for i in range(len(pred))]
                acc = sum(acc) / len(acc)
                total_acc.append(acc)

            if total_loss <= best_loss:
                best_loss = total_loss
                self.save_dict(save_path=save_path, model_prefix=model_prefix)
            print("| Epochs: {} | Training loss: {} | Acc : {} |".format(epoch + 1,
                                                                         round(total_loss, 4),
                                                                         round(acc, 4)))

    def infer(self, sent: str, **kwargs):
        softmax = torch.nn.Softmax(dim=1)

        inputs = self.tokenizer.text_to_idx(sent)
        inputs = torch.LongTensor([inputs]).to(self.device)

        logits = self.model(inputs)
        logits = softmax(logits)

        probs, pred = logits.max(dim=1)
        probs = probs.cpu().detach()
        print(probs, pred)
        # outp = self.decode_outp(inputs[0], pred[0], probs[0], kwargs['threshold'])
        # outp = self.tokenizer.idx_to_text(outp)
        return probs

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = TextCNN(**model_conf)
        try:
            self.model.load_state_dict(model_dict["model_params"])
        except:
            new_dict = OrderedDict()
            for key in model_dict["model_params"].keys():
                new_dict[key.replace('module.', '')] = model_dict["model_params"][key]
            self.model.load_state_dict(new_dict)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def decode_outp(inputs, pred, probs, threshold):
        outp = []
        for i, pre, pro in zip(inputs, pred, probs):
            i, pre, pro = i.item(), pre.item(), pro.item()

            if pro < threshold:
                outp.append(i)
            else:
                if i != pre:
                    outp.append(pre)
                else:
                    outp.append(i)
        return outp
