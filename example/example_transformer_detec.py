from orca.detection import TransformerTypoDetector


with open('data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
data = [s.strip() for s in data]

detector = TransformerTypoDetector(d_model=256, n_head=4, n_layers=2, dim_ff=512, dropout=0.5)

train_config = {
    'sents': data[:200],
    'batch_size': 125,
    'num_epochs': 100,
    'lr': 1e-4,
    'save_path': './',
    'model_prefix': 'tr_detector',
    'typo_num': 2,
    'max_len': 10}
detector.train(**train_config)

# detector.load_model('example/tcnn_detector_test.modeldict')
# detector.infer('도움말', threshold=0.6)
#
#
from orca.dataset import TypoDetectionDataset

dataset = TypoDetectionDataset(data, typo_num=2)

from torch.utils.data import DataLoader
from tqdm import tqdm

loader = DataLoader(dataset, batch_size=256)
for i in tqdm(loader):
    batch = i



batch[0]
batch[1]
detector.tokenizer.idx_to_text(batch[0][-3].tolist())
detector.tokenizer.idx2char[3]