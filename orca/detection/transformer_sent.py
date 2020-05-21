# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.module import TransformerHierachiSeqTagger
from orca.tokenizer import CharacterTokenizer
from orca.dataset import TypoDetectionSentenceLevelDataset
from orca.abstract import Module
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill


from torch.utils.data import DataLoader
from tqdm import tqdm


class TransformerSentTypoDetector(Module):

    def __init__(self,
                 word_dim: int,
                 d_model: int,
                 n_head: int,
                 n_layers: int,
                 dim_ff: int,
                 dropout: float,
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
            'word_dim': word_dim,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_head': n_head,
            'n_layers': n_layers,
            'dim_ff': dim_ff,
            'dropout': dropout,
            'pad_id': self.pad_id,
            'n_class': 2
        }
        self.model = TransformerHierachiSeqTagger(**self.model_conf).to(self.device)
        if self.n_gpu == 1:
            pass
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

    @staticmethod
    def calculate_acc(pred, target):
        acc = []
        for pre, tgt in zip(pred, target):
            for p, t in zip(pre, tgt):
                if t.item() == 2:
                    continue
                else:
                    if p.item() == t.item():
                        acc.append(1)
                    else:
                        acc.append(0)
        acc = sum(acc) / len(acc)
        return round(acc, 4)

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

        dataset = TypoDetectionSentenceLevelDataset(sents,
                                                    typo_num=kwargs['typo_num'],
                                                    max_sent_len=kwargs['max_sent_len'],
                                                    max_word_len=kwargs['max_word_len'],
                                                    ignore_idx=kwargs['ignore_index'])

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
                # loss = F.cross_entropy(logits, target, ignore_index=kwargs['ignore_index'])
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1),
                                       ignore_index=kwargs['ignore_index'])

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()

                # Acc
                _, pred = logits.max(dim=-1)
                total_acc.append(self.calculate_acc(pred, target))
            acc = sum(total_acc) / len(total_acc)
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
        # outp = self.decode_outp(inputs[0], pred[0], probs[0], kwargs['threshold'])
        # outp = self.tokenizer.idx_to_text(outp)
        return probs, pred

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = TransformerClassifier(**model_conf)
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
