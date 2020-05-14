import random

with open('data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
tr_data = [s.strip() for s in data]

random.shuffle(tr_data)

from orca import TextCNNTypoCorrector
model = TextCNNTypoCorrector(256, 256)


train_config = {
    'batch_size': 512,
    'num_epochs': 500,
    'lr': 0.001,
    'save_path': 'example/',
    'model_prefix': 'test_cnn',
    'max_len': 150,
    'threshold': 0.3,
    'noise_char_ratio': 0.05
}

model.train(tr_data, **train_config)
