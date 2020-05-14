#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer.pattern import *
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul

import random


noise_dict = {}

for pattern in [KOR_CHAR_JA_BOTH_PATTERN, KOR_CHAR_JA_UNDER_PATTERN, KOR_CHAR_MO_PATTERN,
                ENG_CHAR_UP_PATTERN, ENG_CHAR_LOW_PATTERN]:
    for i, s in enumerate(pattern):
        noise_dict[s] = pattern[:i] + pattern[i+1:]


def noise_maker(text: str, threshold: float = 0.3, noise_char_ratio: float = 0.05):
    assert threshold >= 0
    assert threshold <= 1

    text_tokenized = flat_hangeul(text)

    rand = random.random()
    if rand >= threshold:
        return text_tokenized, text
    else:
        rand_idx = random.sample(range(len(text_tokenized)), int(len(text_tokenized) * noise_char_ratio))
        for idx in rand_idx:
            if text_tokenized[idx] in list(noise_dict.keys()):
                text_tokenized[idx] = random.choice(noise_dict[text_tokenized[idx]])
        return text_tokenized, merge_flatted_hangeul(text_tokenized)
