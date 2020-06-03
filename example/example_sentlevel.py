from orca.detection import TransformerTypoDetector


with open('data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
data = [s.strip() for s in data]

detector = TransformerTypoDetector(word_dim=128, d_model=256, n_head=4, n_layers=2, dim_ff=128, dropout=0.5)
#
# train_config = {
#     'sents': data[:200],
#     'batch_size': 128,
#     'num_epochs': 10,
#     'lr': 1e-4,
#     'save_path': './',
#     'model_prefix': 'test_sentlevel',
#     'typo_num': 2,
#     'max_sent_len': 20,
#     'max_word_len': 10,
#     'ignore_index': 2
# }
# detector.train(**train_config)

detector.load_model('example/tr_sentlevel_all.modeldict')
# detector.infer('ㅛㅏ랑해', threshold=0.6)
detector.infer('데이터 선물하렬면 어덯게 해?', max_word_len=10)

