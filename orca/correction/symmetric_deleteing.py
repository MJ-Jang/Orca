# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from symspellpy.symspellpy import SymSpell, Verbosity
from orca.abstract import Module
from typing import Text
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul
from collections import Counter
from itertools import islice
from tqdm import tqdm

import os


class SymDeletingTypoCorrecter(Module):

    def __init__(self, max_edit_dist: int = 2, prefix_length: int = 10):
        self.symspell = SymSpell(max_dictionary_edit_distance=max_edit_dist, prefix_length=prefix_length)
        self.max_edit_dist = max_edit_dist

    def train(self,
              corpus_path: str,
              save_path: str,
              unigram_dict_prefix: str,
              bigram_dict_prefix: str = None,
              **kwargs
              ):
        self.symspell.create_dictionary(corpus_path)
        # 1) Unigram dict
        worddict = ''
        for key, count in self.symspell.words.items():
            worddict += '{} {}\n'.format(''.join(flat_hangeul(key)), count)

        unigram_save_path = os.path.join(save_path, unigram_dict_prefix + '.txt')
        with open(unigram_save_path, 'w', encoding='utf-8') as file:
            for line in worddict:
                file.write(line)
            file.close()
        print("Total {} Unigrams are saved!".format(len(self.symspell.words.items())))

        if bigram_dict_prefix:
            # 2) Bigram dict
            with open(corpus_path, 'r', encoding='utf-8') as file:
                corpus = file.readlines()
            corpus = [s.strip() for s in corpus]

            bi_count = self.count_bigrams(corpus, min_count=5)

            bi_dict = ''
            for key, count in bi_count.items():
                s1, s2 = key.split(' ')
                bi_dict += '{} {} {}\n'.format(''.join(flat_hangeul(s1)),
                                               ''.join(flat_hangeul(s2)),
                                               count)

            bigram_save_path = os.path.join(save_path, bigram_dict_prefix + '.txt')
            with open(bigram_save_path, 'w', encoding='utf-8') as biFile:
                for line in bi_dict:
                    biFile.write(line)
                biFile.close()
            print("Total {} bigrams are saved!".format(len(bi_count)))

    def load_model(self, unigram_dict_path: str, bigram_dict_path: str = None, **kwargs):
        try:
            here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
            default_path = os.path.join(here, "resources", 'default_uni_dict.txt')

            self.symspell.load_dictionary(default_path, term_index=0, count_index=1)
            self.symspell.load_dictionary(unigram_dict_path, term_index=0, count_index=1)
        except ValueError:
            raise ValueError("Specified unigram dictionary path not exist")

        if bigram_dict_path:
            try:
                self.symspell.load_bigram_dictionary(unigram_dict_path, term_index=0, count_index=1)
            except ValueError:
                raise ValueError("Specified bigram dictionary path not exist")

    def infer(self, word: Text, **kwargs):
        suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
        suggestions = self.symspell.lookup(''.join(flat_hangeul(word)), suggestion_verbosity, self.max_edit_dist)
        if suggestions:
            word = list(suggestions[0].term)
            return merge_flatted_hangeul(word)
        return word

    @staticmethod
    def count_bigrams(corpus: list, min_count: int):
        bigrams = []
        for t in tqdm(corpus):
            if t.__class__ != str:
                continue
            else:
                text = t.split(' ')
                _bigrams = zip(*[text[i:] for i in range(2)])
                bigrams += [' '.join(s) for s in list(_bigrams)]

        count = Counter(bigrams)
        new_dict = {}
        for key, value in count.items():
            if value >= min_count:
                new_dict[key] = value
        return new_dict
