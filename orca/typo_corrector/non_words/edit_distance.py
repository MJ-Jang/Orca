# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.abstract import TypoCorrecter
from orca.tokenizer import CharacterTokenizer
from konlpy.tag import Okt, Kkma, Komoran, Hannanum, Mecab
from tqdm import tqdm
from typing import Text

import dill
import editdistance
import os
import numpy as np
import difflib
import re


class EditDistTypoCorrecter(TypoCorrecter):
    def __init__(self, tokenizer: Text = 'Komoran'):
        # use konlpy Komoran tokenzer as base
        if tokenizer not in ['Okt', 'Kkma', 'Komoran', 'Hannanum']:
            raise ValueError("Specified tokenizer is not supported")

        self.tokenizer = eval(tokenizer + "()")
        self.tokenizer.morphs('test') # slow at first inference

        self.chartok = CharacterTokenizer()

        self.word_set = set()
        self.word_list = list()
        self.word_dict = dict()

    def train(self,
              sents: list,
              **kwargs
              ):
        # For now, use Whitespace tokenization as a default
        for text in tqdm(sents):
            # add whiespacing
            words = self._whitespacing(text)
            for w in words:
                self.word_set.add(w)

            words = self._tokenize(text)
            for w in words:
                self.word_set.add(w)
        self.word_list = sorted(list(self.word_set))
        self.word_dict = self._build_word_dict(self.word_list)

    def load_model(self, model_path: str):
        with open(model_path, "rb") as file:
            model = dill.load(file)
        self.word_list = model['word_list']
        self.word_set = model['word_set']
        self.word_dict = model['word_dict']

    def save_dict(self, save_path: str, model_prefix: str):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, model_prefix+'.vocab')

        outp_dict = {
            'word_list': self.word_list,
            'word_set': self.word_set,
            'word_dict': self.word_dict
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)

    def infer(self, sent: Text, **kwargs):
        words = self._whitespacing(sent)
        for idx, w in enumerate(words):
            if not self._is_word_in_dict(w) and len(w) > 1:
                replace_word, flag = self._return_correct_word(w, kwargs['threshold'])
                if flag:
                    s, e = re.search(pattern=w, string=sent).span()
                    sent = sent[:s] + replace_word + sent[e:]
        return sent

    def _tokenize(self, text: Text) -> list:
        return self.tokenizer.morphs(text)

    @staticmethod
    def _whitespacing(text: Text) -> list:
        return text.split(' ')

    @staticmethod
    def _decode_text(token: list) -> Text:
        return ' '.join(token)

    @staticmethod
    def _build_word_dict(word_list):
        word_list = word_list
        word_dict = {}
        for w in word_list:
            length = len(w)
            if length >= 12:
                length = 'else'
            else:
                length = str(length)
            if length not in list(word_dict.keys()):
                word_dict[length] = []
            word_dict[length].append(w)
        return word_dict

    def _return_correct_word(self, word: Text, threshold: int):
        def extract_close_words(word, word_list):
            # extract 50 maximum close words from dictionary
            return difflib.get_close_matches(word, word_list, n=100, cutoff=0.5)

        outp = word
        length = len(word)
        if length >= 12:
            candidates = extract_close_words(word, self.word_dict['else']) + extract_close_words(word, self.word_dict['11'])
        elif length == 11:
            candidates = extract_close_words(word, self.word_dict['else']) + \
                         extract_close_words(word, self.word_dict['11']) + \
                         extract_close_words(word, self.word_dict['10'])
        else:
            candidates = extract_close_words(word, self.word_dict[str(length+1)]) + \
                         extract_close_words(word, self.word_dict[str(length)]) + \
                         extract_close_words(word, self.word_dict[str(length-1)])

        if candidates:
            dist = [editdistance.eval(self.chartok.text_to_token(word), self.chartok.text_to_token(w))
                    for w in candidates]

            min_id = np.argmin(dist)
            if dist[min_id] <= threshold:
                outp = candidates[min_id]
        return outp, outp != word

    def _is_word_in_dict(self, word):
        length = len(word)
        return word in self.word_dict[str(length)]

    def __len__(self):
        return len(self.word_set)
