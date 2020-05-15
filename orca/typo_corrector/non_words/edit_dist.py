# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.abstract import TypoCorrecter
from orca.tokenizer import CharacterTokenizer
from konlpy.tag import Okt, Kkma, Komoran, Hannanum, Mecab
from tqdm import tqdm
from typing import Text
from collections import Counter

import dill
import editdistance
import os
import numpy as np
import difflib
import re


class EditDistTypoCorrecter(TypoCorrecter):
    def __init__(self, word_dict: dict = None, tokenizer: Text = 'Okt'):
        # use konlpy Okt tokenzer as base
        if tokenizer not in ['Okt', 'Kkma', 'Komoran', 'Hannanum']:
            raise ValueError("Specified tokenizer is not supported")

        self.tokenizer = eval(tokenizer + "()")
        self.tokenizer.morphs('test') # slow at first inference

        self.chartok = CharacterTokenizer()

        self.word_dict = Counter(word_dict)
        self.word_dict_by_length = dict()

    def train(self,
              sents: list,
              **kwargs
              ):
        # For now, use Whitespace tokenization as a default
        for text in tqdm(sents):
            # add whiespacing
            words = self._whitespacing(text)
            self.word_dict.update(words)
            for w in words:
                if len(w) > 1:
                    if not self.word_dict_by_length.get(len(w)):
                        self.word_dict_by_length[len(w)] = set()
                    self.word_dict_by_length[len(w)].add(w)

            # add kor tokenization
            words = self._tokenize(text)
            self.word_dict.update(words)
            for w in words:
                if len(w) > 1:
                    if not self.word_dict_by_length.get(len(w)):
                        self.word_dict_by_length[len(w)] = set()
                    self.word_dict_by_length[len(w)].add(w)

    def load_model(self, model_path: str):
        with open(model_path, "rb") as file:
            model = dill.load(file)
        self.word_dict = model['word_dict']
        self.word_dict_by_length = model['word_dict_by_length']

    def save_dict(self, save_path: str, model_prefix: str):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, model_prefix+'.vocab')

        outp_dict = {
            'word_dict': self.word_dict,
            'word_dict_by_length': self.word_dict_by_length
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)

    def infer(self, sent: Text, **kwargs):
        words = self._whitespacing(sent)
        for idx, w in enumerate(words):
            if not self._is_word_in_dict(w) and len(w) > 1:
                replace_word = self._return_correct_word(w, kwargs['threshold'])
                words[idx] = replace_word
        return ' '.join(words)

    def _tokenize(self, text: Text) -> list:
        return self.tokenizer.morphs(text)

    @staticmethod
    def _whitespacing(text: Text) -> list:
        return text.split(' ')

    @staticmethod
    def _decode_text(token: list) -> Text:
        return ' '.join(token)

    def _return_correct_word(self, word: Text, threshold: int):
        def extract_close_words(word, word_list):
            # extract 50 maximum close words from dictionary
            return difflib.get_close_matches(word, word_list, n=50, cutoff=0.5)

        if len(word) <= 1:
            return word

        freq = self.word_dict.get(word)
        if freq:
            return word

        length = len(word)
        candidates = set()
        for i in range(length-1, length+2):
            w_set = self.word_dict_by_length.get(i)
            if w_set:
                candidates.update(w_set)
        candidates = extract_close_words(word, candidates)

        if candidates:
            dist = [editdistance.eval(self.chartok.text_to_token(word), self.chartok.text_to_token(w))
                    for w in candidates]
            min_id = np.argmin(dist)
            if dist[min_id] <= threshold:
                return candidates[min_id]
        return word

    def _is_word_in_dict(self, word):
        return self.word_dict.get(word)

    def __len__(self):
        return len(self.word_dict)
