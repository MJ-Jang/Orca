# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from orca.tokenizer.pattern import KOR_CHAR_MO_PATTERN, KOR_CHAR_JA_UNDER_PATTERN, KOR_CHAR_JA_BOTH_PATTERN
from orca.utils.hangeul import merge_flatted_hangeul, flat_hangeul

import re

mo_p = ''.join(KOR_CHAR_MO_PATTERN)
ko_un_p = ''.join(KOR_CHAR_JA_BOTH_PATTERN+KOR_CHAR_JA_UNDER_PATTERN)

rule_typo_patterns = {
    'ㅠ[{}][가-힣]'.format(mo_p): ['ㅠ', 'ㅍ'],
    'ㅠ[{}][{}]'.format(mo_p, ko_un_p): ['ㅠ', 'ㅍ'],
    'ㅗ[{}][가-힣]'.format(mo_p): ['ㅗ', 'ㅎ'],
    'ㅗ[{}][{}]'.format(mo_p, ko_un_p): ['ㅗ', 'ㅎ'],
    'ㅛ[{}][가-힣]'.format(mo_p): ['ㅛ', 'ㅅ'],
    'ㅛ[{}][{}]'.format(mo_p, ko_un_p): ['ㅛ', 'ㅅ'],
}


sent = 'ㅛㅏ랑해'
for p, repl in rule_typo_patterns.items():
    match = re.search(pattern=p, string=sent)
    if match:
        s, e = match.span()
        value = sent[s:e]
        value = value.replace(repl[0], repl[1])
        value = merge_flatted_hangeul(flat_hangeul(value))
        sent = sent[:s] + value + sent[e:]

