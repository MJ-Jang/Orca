import dill

from orca.tokenizer.pattern import *
from orca.utils.hangeul import flat_hangeul, merge_flatted_hangeul


class CharacterTokenizer:
    def __init__(self):
        super(CharacterTokenizer, self).__init__()

        self.unk = "<unk>"
        self.pad = '<pad>'
        self.c_list = [self.pad, self.unk] + KOR_CHAR_PATTERN + ENG_CHAR_PATTERN + NUMBER_PATTERN + SYMBOL_PATTERN
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
