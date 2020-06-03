from orca.utils.hangeul import flat_hangeul
corpus_path = 'data/train_data.txt'
with open(corpus_path, 'r', encoding='utf-8') as file:
    corpus = file.readlines()
corpus = [s.strip() for s in corpus]

from tqdm import tqdm
bigrams = []
for t in tqdm(corpus):
    text = t.split(' ')
    _bigrams = zip(*[text[i:] for i in range(2)])
    bigrams += [' '.join(s) for s in list(_bigrams)]

from collections import Counter
b = Counter(bigrams)

from tqdm import tqdm
from collections import Counter


def count_bigrams(corpus: list, min_count: int):
    bigrams = []
    for t in tqdm(corpus):
        if t.__class__ != str:
            continue
        else:
            text = t.split(' ')
            _bigrams = zip(*[text[i:] for i in range(2)])
            bigrams += [' '.join(s) for s in list(_bigrams)]

    count = Counter(bigrams)
    new_dict = {}
    for key, value in count.items():
        if value >= min_count:
            new_dict[key] = value
    return new_dict

a = count_bigrams(corpus, 5)
len(a)