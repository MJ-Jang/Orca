# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from typing import Text

from orca.detection.transformer import TransformerTypoDetector

import os


class OrcaTypoProcessor:
    def __init__(self,
                 detection_model_path: Text = None,
                 word_max_len: int = 10) -> None:
        self.detector = TransformerTypoDetector(word_dim=128,
                                                d_model=128,
                                                n_head=4,
                                                n_layers=1,
                                                dim_ff=128,
                                                dropout=0.5)
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

    def process(self, sent: Text):
        sent_splitted = sent.split(' ')
        probs, preds = self.detector.infer(sent=sent, max_word_len=self.word_max_len)
        probs, preds = probs[0], preds[0]

        outp = []
        for i, (pr, pred, token) in enumerate(zip(probs, preds, sent_splitted)):
            if pred == 1:
                outp.append({"word": token, "prob": round(pr, 3), 'position': i})
        return outp
