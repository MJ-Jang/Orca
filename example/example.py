# data preprocessing ##
import re

with open('data/nlu.md', 'r', encoding='utf-8') as file:
    sents = file.readlines()

p1 = re.compile('[\[\]]+')
p2 = re.compile('\([a-zA-Z_\d]+\)')

data = []
for s in sents:
    s = s.strip()
    if s and s.startswith('-'):
        d = p1.sub('', s[1:].strip())
        d = p2.sub('', d)
        data.append(d)

with open('data/test_data.txt', 'w', encoding='utf-8') as file:
    for line in data:
        file.write(line + '\n')

import random

with open('../data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
tr_data = [s.strip() for s in data]

random.shuffle(tr_data)

from orca import CBOWTypoCorrector
model = CBOWTypoCorrector(256, 128)

model.load_model('example/test.modeldict')
model.infer('브랑스로 여행간다', window_size=10, threshold=0.9)


# train_config = {
#     'window_size': 5,
#     'batch_size': 128,
#     'num_epochs': 500,
#     'lr': 0.01,
#     'save_path': 'example/',
#     'model_prefix': 'test'
# }
#
# model.train(tr_data, **train_config)
