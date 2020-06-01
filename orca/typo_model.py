# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from typing import Text

from orca.detection.transformer import TransformerTypoDetector
from orca.correction import SymDeletingTypoCorrecter

import os


class OrcaTypoProcessor:
    def __init__(self,
                 unigram_dict_path: Text,
                 bigram_dict_path: Text = None,
                 detection_model_path: Text = None,
                 word_max_len: int = 10) -> None:
        self.detector = TransformerTypoDetector(word_dim=128,
                                                d_model=128,
                                                n_head=4,
                                                n_layers=1,
                                                dim_ff=128,
                                                dropout=0.5)
        self.corrector = SymDeletingTypoCorrecter()
        self.corrector.load_model(unigram_dict_path=unigram_dict_path, bigram_dict_path=bigram_dict_path)

        if detection_model_path:
            self.detector.load_model(detection_model_path)
            self.word_max_len = word_max_len
        else:
            # use default model
            filename = 'transformer_detector_basic.modeldict'
            here = os.path.dirname(__file__)
            full_filename = os.path.join(here, "resources", filename)
            self.detector.load_model(full_filename)
            self.word_max_len = 10

    def process(self, sent: Text, non_typo_threshold: float = None):
        sent_splitted = sent.split(' ')
        probs, preds = self.detector.infer(sent=sent, max_word_len=self.word_max_len)
        probs, preds = probs[0], preds[0]

        if non_typo_threshold:
            probs, preds = self._prepro_prediction(preds, probs, non_typo_threshold)

        outp = []
        for i, (pr, pred, token) in enumerate(zip(probs, preds, sent_splitted)):
            if len(token) == 1:
                outp.append(sent_splitted[i])
                continue
            if pred == 1 or pred == 2:
                # 띄어쓰기로 인해 분리된 case
                if i < len(preds)-1:
                    repl = self.corrector.infer(sent_splitted[i])
                    repl_nextToken = repl + sent_splitted[i+1]
                    repl_bi = self.corrector.infer(repl_nextToken)
                    if repl_bi != repl_nextToken:
                        repl = repl_bi.replace(sent_splitted[i+1], '').strip()
                    outp.append(repl)
                else:
                    outp.append(self.corrector.infer(sent_splitted[i]))
            else:
                outp.append(sent_splitted[i])
        return ' '.join(outp)

    def _infer_detection(self, sent: Text, non_typo_threshold: float = None):
        sent_splitted = sent.split(' ')
        probs, preds = self.detector.infer(sent=sent, max_word_len=self.word_max_len)
        probs, preds = probs[0], preds[0]

        if non_typo_threshold:
            probs, preds = self._prepro_prediction(preds, probs, non_typo_threshold)

        outp = []
        for i, (pr, pred, token) in enumerate(zip(probs, preds, sent_splitted)):
            if len(token) <= 1:
                continue
            if pred == 1 or pred == 2:
                outp.append({'word': sent_splitted[i], 'position': i})
            elif pred == 0 and non_typo_threshold:
                if pr <= non_typo_threshold:
                    outp.append({'word': sent_splitted[i], 'position': i})
        return outp

    def _infer_correction(self, word: Text):
        return self.corrector.infer(word)

    @staticmethod
    def _prepro_prediction(preds, probs, non_typo_threshold):
        for i, (pr, pred) in enumerate(zip(probs, preds)):
            if pred == 0 and pr <= non_typo_threshold:
                preds[i] = 2
        return probs, preds
