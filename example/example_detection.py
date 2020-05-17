from orca.detection import TextCNNTypoDetector


with open('../data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
data = [s.strip() for s in data]

detector = TextCNNTypoDetector(256, 256, 2)
train_config = {
    'sents': data,
    'batch_size': 125,
    'num_epochs': 100,
    'lr': 1e-4,
    'save_path': './',
    'model_prefix': 'tcnn_detector',
    'typo_num': 2,
    'max_len': 20}
detector.train(**train_config)