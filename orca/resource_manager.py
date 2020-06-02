# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from typing import Text
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul


class UnigramDictManager:
    def __init__(self, file_path: str):
        self.path = file_path

        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        data = [s.strip() for s in data]
        self.uni_dict = {}

        for line in data:
            key, cnt = line.split(' ')
            cnt = int(cnt)
            self.uni_dict[key] = cnt

    def create(self, word: Text, count: int = 1000):
        word = ''.join(flat_hangeul(word))
        value = self.uni_dict.get(word)
        if not value:
            self.uni_dict[word] = count

    def read(self):
        word_list = list(self.uni_dict.keys())
        word_list = [merge_flatted_hangeul(w) for w in word_list]
        word_list = sorted(word_list, reverse=False)
        return word_list

    def update(self, word: Text, repl_word: Text):
        word = ''.join(flat_hangeul(word))
        value = self.uni_dict.get(word)

        repl_word = ''.join(flat_hangeul(repl_word))
        if value:
            self.uni_dict[repl_word] = self.uni_dict.pop(word)

        else:
            raise ValueError("Specified word not exist in default unigram dictionary")

    def delete(self, word: Text):
        word = ''.join(flat_hangeul(word))
        value = self.uni_dict.get(word)
        if value:
            self.uni_dict.pop(word)
        else:
            raise ValueError("Specified word not exist in default unigram dictionary")

    def search(self, word: Text):
        word = ''.join(flat_hangeul(word))
        value = self.uni_dict.get(word)
        if value:
            return True
        else:
            return False

    def save_dict(self, local_path: str = None):
        worddict = ''
        for key, count in self.uni_dict.items():
            worddict += '{} {}\n'.format(''.join(flat_hangeul(key)), count)

        if local_path:
            save_path = local_path
        else:
            save_path = self.path
        with open(save_path, 'w', encoding='utf-8') as file:
            for line in worddict:
                file.write(line)
            file.close()
