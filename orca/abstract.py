#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import torch
import os
import dill

from orca.tokenizer import CharacterTokenizer


class TypoCorrecter(object):
    """
    Abstract class for TypoCorrector
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str,
              **kwargs
              ):
        pass

    @abc.abstractmethod
    def infer(self, sent: list, **kwargs):
        pass

    @abc.abstractmethod
    def load_model(self, model_path: str):
        if model_path:
            with open(model_path, 'rb') as modelFile:
                model_dict = dill.load(modelFile)
            self.model_conf = model_dict['model_conf']
            self.model = self.model_class(**self.model_conf)
            self.model.load_state_dict(model_dict["model_params"])
            self.model.to(self.device)

    @abc.abstractmethod
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
