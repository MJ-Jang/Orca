#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer.pattern import *
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul
from orca.utils.pattern import typo_pattern

import random


noise_dict = {}

for pattern in [KOR_CHAR_JA_BOTH_PATTERN, KOR_CHAR_JA_UNDER_PATTERN, KOR_CHAR_MO_PATTERN,
                ENG_CHAR_UP_PATTERN, ENG_CHAR_LOW_PATTERN]:
    for i, s in enumerate(pattern):
        noise_dict[s] = pattern[:i] + pattern[i+1:]


def noise_maker(word: str, threshold: float = 0.3, noise_char_num: int = 2, method: str = 'g'):
    """
    :param word: single word
    :param threshold: probability of generating noise
    :param noise_char_num: maximum number of noise character
    :param method: generation method, ("g": general, "ct": close typo)
    """
    noise_dict_key = {
        'g': noise_dict,
        'ct': typo_pattern
    }
    if method not in ['g', 'ct']:
        raise ValueError("specified generation method is not supported")
    n_dict = noise_dict_key[method]

    assert threshold >= 0
    assert threshold <= 1

    text_tokenized = flat_hangeul(word)

    rand = random.random()
    if rand >= threshold:
        return text_tokenized, word
    else:
        sample_num_noise = random.choice(range(1, noise_char_num+1))
        rand_idx = random.sample(range(len(text_tokenized)), min(len(text_tokenized), sample_num_noise))
        for idx in rand_idx:
            if text_tokenized[idx] in list(n_dict.keys()):
                text_tokenized[idx] = random.choice(n_dict[text_tokenized[idx]])
        return text_tokenized, merge_flatted_hangeul(text_tokenized)
