# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from symspellpy.symspellpy import SymSpell, Verbosity
from orca.abstract import Module
from typing import Text
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul

import os


class SymDeletingTypoCorrecter(Module):

    def __init__(self, max_edit_dist: int = 2, prefix_length: int = 10):
        self.symspell = SymSpell(max_dictionary_edit_distance=max_edit_dist, prefix_length=prefix_length)
        self.max_edit_dist = max_edit_dist

    def train(self,
              corpus_path: str,
              save_path: str,
              save_dict_prefix: str,
              **kwargs
              ):
        self.symspell.create_dictionary(corpus_path)
        worddict = ''
        for key, count in self.symspell.words.items():
            worddict += '{} {}\n'.format(''.join(flat_hangeul(key)), count)

        save_path = os.path.join(save_path, save_dict_prefix + '.txt')
        with open(save_path, 'w', encoding='utf-8') as file:
            for line in worddict:
                file.write(line)
        print("Total {} words are saved!".format(len(self.symspell.words.items())))

    def load_model(self, word_dict_path: str):
        try:
            self.symspell.load_dictionary(word_dict_path, term_index=0, count_index=1)
        except ValueError:
            raise ValueError("Specified dictionary path not exist")

    def infer(self, word: Text, **kwargs):
        suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
        suggestions = self.symspell.lookup(''.join(flat_hangeul(word)), suggestion_verbosity, self.max_edit_dist)
        if suggestions:
            word = list(suggestions[0].term)
            return merge_flatted_hangeul(word)
        return word

aa = SymDeletingTypoCorrecter()
aa.load_model('./data/testdict.txt')
aa.infer('ㅠㅡ랑스로')