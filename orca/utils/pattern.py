#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division


typo_pattern = {
    "ㄱ": ["ㄷ", "ㅅ"],
    "ㄲ": ["ㄸ", "ㅆ"],
    "ㄴ": ["ㅁ", "ㅇ"],
    "ㄷ": ["ㅈ", "ㄱ"],
    "ㄸ": ["ㅉ", "ㄲ"],
    "ㄹ": ["ㅇ", "ㅎ"],
    "ㅁ": ["ㄴ"],
    "ㅂ": ["ㅈ"],
    "ㅃ": ["ㅉ"],
    "ㅅ": ["ㄱ", "ㅛ"],
    "ㅆ": ["ㄲ"],
    "ㅇ": ["ㄴ", "ㄹ"],
    "ㅈ": ["ㅂ", "ㄷ"],
    "ㅉ": ["ㅃ" "ㄸ"],
    "ㅊ": ["ㅌ", "ㅍ"],
    "ㅋ": ["ㅌ"],
    "ㅌ": ["ㅋ", "ㅊ"],
    "ㅍ": ["ㅊ", "ㅠ"],
    "ㅎ": ["ㄹ", "ㅗ"],
    "ㅏ": ["ㅣ", "ㅓ"],
    "ㅑ": ["ㅕ", "ㅐ"],
    "ㅓ": ["ㅗ", "ㅏ"],
    "ㅕ": ["ㅛ", "ㅑ"],
    "ㅗ": ["ㅎ", "ㅓ"],
    "ㅛ": ["ㅅ", "ㅕ"],
    "ㅜ": ["ㅠ", "ㅡ"],
    "ㅠ": ["ㅍ", "ㅜ"],
    "ㅡ": ["ㅜ"],
    "ㅣ": ["ㅏ"],
    "ㅐ": ["ㅑ", "ㅔ"],
    "ㅔ": ["ㅐ"],
    "ㅒ": ["ㅑ", "ㅖ"],
    "ㅖ": ["ㅒ"],
}