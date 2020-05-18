import dill
import os

from orca.tokenizer.pattern import *
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul
from tqdm import tqdm
from collections import Counter


class CharacterTokenizer:
    def __init__(self, model_path: str = None):
        super(CharacterTokenizer, self).__init__()

        self.unk = "<unk>"
        self.pad = '<pad>'
        self.pre_tokens = [self.pad, self.unk]
        self.unk_id, self.pad_id = '', ''

        self.char2idx = {}
        self.idx2char = {}
        self.c_list = []

        self.load(model_path)

    def load(self, model_path: str = None, use_defalt: bool = True):
        if model_path:
            with open(model_path, 'rb') as tokFile:
                model = dill.load(tokFile)
            self.char2idx = model['char2idx']
            self.idx2char = model['idx2char']
            self.c_list = list(self.char2idx.keys())
            self.pad_id = self.char2idx[self.pad]
            self.unk_id = self.char2idx[self.unk]
        else:
            if use_defalt:
                filename = 'char_tokenizer.model'
                here = '/'.join(os.path.dirname(__file__).split('/')[:-1])
                full_filename = os.path.join(here, "resources", filename)
                self.load(full_filename)
            else:
                self.char2idx = {}
                self.idx2char = {}
                self.c_list = []

    def train(self, sents: list, min_count: int, save_path: str, model_prefix: str):
        chars = []
        for s in tqdm(sents, desc='processing corpus'):
            chars += list(s)
        counter = Counter(chars)

        self.char2idx = {}
        self.idx2char = {}

        for v, k in enumerate(self.pre_tokens):
            self.char2idx[k] = v
            self.idx2char[v] = k

        for i, key in enumerate(counter):
            if counter[key] >= min_count:
                self.char2idx[key] = len(self.char2idx)
                self.idx2char[len(self.char2idx)] = key
        self.c_list = list(self.char2idx.keys())
        self.pad_id = self.char2idx[self.pad]
        self.unk_id = self.char2idx[self.unk]

        os.makedirs(save_path, exist_ok=True)

        model_name = os.path.join(save_path, model_prefix + ".model")
        outp = {"char2idx": self.char2idx, 'idx2char': self.idx2char}
        with open(model_name, "wb") as file:
            dill.dump(outp, file)

    def tokenize(self, text, to_id=True):
        if to_id:
            res = list(text)
            res = [self.char2idx[t] if t in self.c_list else self.char2idx[self.unk] for t in res]
            return res
        else:
            return list(text)

    @staticmethod
    def text_to_token(text: str) -> list:
        return list(text)

    @staticmethod
    def token_to_text(token: list) -> str:
        return ''.join(token)

    def text_to_idx(self, text: str) -> list:
        token = self.text_to_token(text)
        token = [self.char2idx[t] if t in self.c_list else self.char2idx[self.unk] for t in token]
        return token

    def idx_to_text(self, idxs: list) -> str:
        token = [self.idx2char[t] for t in idxs]
        text = self.token_to_text(token)
        return ''.join(text)

    def __len__(self):
        return len(self.c_list)


class JasoTokenizer:
    def __init__(self):
        super(JasoTokenizer, self).__init__()

        self.unk = "<unk>"
        self.pad = '<pad>'
        self.c_list = [self.pad, self.unk] + KOR_CHAR_JA_BOTH_PATTERN + KOR_CHAR_JA_UNDER_PATTERN\
                      + KOR_CHAR_MO_PATTERN + ENG_CHAR_UP_PATTERN\
                      + ENG_CHAR_LOW_PATTERN + NUMBER_PATTERN + SYMBOL_PATTERN

        self.tok_to_id_dict = {}
        self.id_to_tok_dict = {}

        self.load()
        self.unk_id = self.tok_to_id_dict[self.unk]
        self.pad_id = self.tok_to_id_dict[self.pad]

    def load(self,):
        for i, c in enumerate(self.c_list):
            self.tok_to_id_dict[c] = i
            self.id_to_tok_dict[i] = c

    def tokenize(self, text, to_id=True):
        if to_id:
            res = list(text)
            res = [self.tok_to_id_dict[t] if t in self.c_list else self.tok_to_id_dict[self.unk] for t in res]
            return res
        else:
            return list(text)

    @staticmethod
    def text_to_token(text: str) -> list:
        return flat_hangeul(text)

    @staticmethod
    def token_to_text(token: list) -> str:
        # To avoid error in joining splitted hangeul
        for i, t in enumerate(token):
            if t == '<unk>':
                token[i] = '윬'
        return merge_flatted_hangeul(token)

    def text_to_idx(self, text: str) -> list:
        token = self.text_to_token(text)
        token = [self.tok_to_id_dict[t] if t in self.c_list else self.tok_to_id_dict[self.unk] for t in token]
        return token

    def idx_to_text(self, idxs: list) -> str:
        token = [self.id_to_tok_dict[t] for t in idxs]
        text = self.token_to_text(token)
        text = ['<unk>' if t == '윬' else t for t in text]
        return ''.join(text)

    def __len__(self):
        return len(self.c_list)
