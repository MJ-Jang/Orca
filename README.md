# Orca

![Python 3.5+](https://img.shields.io/badge/python-3.5+-green.svg)


![](images/orca.png )

---
**ORyu Correcting Assistance**
- built for Korean spell checking

### 1. Structure
TBD...

### 2. Training detector
##### a. prepare data load data
```python
with open('test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
data = [s.strip() for s in data]
```
##### b. specify model parameters and train config
```python
from orca.detection import TransformerTypoDetector

detector = TransformerTypoDetector(word_dim=128, d_model=256, n_head=4, n_layers=2, dim_ff=128, dropout=0.5)

train_config = {
     'sents': data[:200],
     'batch_size': 128,
     'num_epochs': 10,
     'lr': 1e-4,
     'save_path': './',
     'model_prefix': 'test_sentlevel',
     'typo_num': 2,
     'max_sent_len': 20,
     'max_word_len': 10,
     'ignore_index': 2,
     'num_workers': 4
}
```

##### c. train model
```python
detector.train(**train_config)
```

### 3. Train corrector
```python
from orca.correction import SymDeletingTypoCorrecter

corrector = SymDeletingTypoCorrecter(max_edit_dist=2, prefix_length=10)
corrector.train(corpus_path='test_data.txt', save_path='data/', unigram_dict_prefix='unigram_dict')
```

### 4. Inference
```python
from orca import OrcaTypoProcessor

spellchecker = OrcaTypoProcessor(unigram_dict_path='data/unigram_dict.txt', detection_model_path='test_sentlevel.modeldict')
text = 'ㅠㅡ랑스로 여행갈래애'
spellchecker.process(text)
```

### 5. Manage unigram dictionary manually
##### a. load resource manager
```python
from orca.resource_manager import UnigramDictManager
manager = UnigramDictManager('data/unigram_dict.txt')
```

##### b-1. create new unigram
```python
manager.create(word='대한민국', count=1000)
```
##### b-2. read unigram
```python
manager.read(word='대한민국')
```
##### b-3. delete unigram
```python
manager.delete(word='대한민국')
```
##### b-4. update unigram
```python
manager.update(word='대한민국', repl_word='한국')
```
##### b-5. search is unigram exist in dict
```python
manager.search(word='대한민국')
```
##### c. save unigram dict
```python
manager.save_dict(local_path='test.dict')
```





