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


class TransformerTypoDetector(Module):

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
        tp, tn, fp, fn = 0, 0, 0, 0
        for pre, tgt in zip(pred, target):
            for p, t in zip(pre, tgt):
                p, t = p.item(), t.item()
                if t == 2:
                    continue
                else:
                    if p == 1 and t == 1:
                        tp += 1
                    elif p == 1 and t == 0:
                        fp += 1
                    elif p == 0 and t == 1:
                        fn += 1
                    elif p == 0 and t == 0:
                        tn += 1

        acc = round((tp + tn) / (tp + tn + fp + fn), 4)
        precision = round(tp/(tp + fp), 4)
        recall = round(tp/(tp + fn), 4)
        f1 = round(2 * recall * precision / (recall + precision), 4)
        return acc, precision, recall, f1

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
            total_acc, total_precision, total_recall, total_f1 = [], [], [], []
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
                acc, precision, recall, f1 = self.calculate_acc(pred, target)

                total_acc.append(acc)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(f1)

            acc = sum(total_acc) / len(total_acc)
            precision = sum(total_precision) / len(total_precision)
            recall = sum(total_recall) / len(total_recall)
            f1 = sum(total_f1) / len(total_f1)

            if total_loss <= best_loss:
                best_loss = total_loss
                self.save_dict(save_path=save_path, model_prefix=model_prefix)
            print("| Epochs: {} | Training loss: {} |".format(epoch + 1,
                                                              round(total_loss, 4)))
            print("| Acc : {} | Precision: {} | Recall : {} | F1: {} |".format(round(acc, 4),
                                                                               round(precision, 4),
                                                                               round(recall, 4),
                                                                               round(f1, 4)))

    def infer(self, sent: str, **kwargs):
        softmax = torch.nn.Softmax(dim=-1)
        tokens = sent.split(' ')
        tokens = [self.tokenizer.text_to_idx(t) for t in tokens]
        tokens = [t + [self.pad_id] * (kwargs['max_word_len']-len(t)) if len(t) < kwargs['max_word_len']
                  else t[:kwargs['max_word_len']] for t in tokens]

        inputs = torch.LongTensor([tokens]).to(self.device)

        logits = self.model(inputs)
        logits = softmax(logits)

        probs, pred = logits.max(dim=-1)
        probs = probs.cpu().detach()
        return probs.tolist(), pred.tolist()

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = TransformerHierachiSeqTagger(**model_conf)
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
