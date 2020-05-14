import random

with open('data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
tr_data = [s.strip() for s in data]

random.shuffle(tr_data)

from orca import TransformerTypoCorrector
model = TransformerTypoCorrector(d_model=256, n_head=8, n_layers=3, dim_ff=128, dropout=0.4)
#
# train_config = {
#     'batch_size': 128,
#     'num_epochs': 50,
#     'lr': 0.001,
#     'save_path': 'example/',
#     'model_prefix': 'test_transformer',
#     'max_len': 150,
#     'threshold': 0.3,
#     'noise_char_ratio': 0.05
# }
#
# model.train(tr_data, **train_config)
model.load_model(model_path='example/test_transformer.modeldict')
model.infer('내일부터 일시정지 부탁합니다', threshold=0.9)
