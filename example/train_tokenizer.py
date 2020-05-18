from orca.tokenizer import CharacterTokenizer

with open('data/test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
data = [s.strip() for s in data]

tok = CharacterTokenizer()
tok.train(data, 2, save_path='example', model_prefix='char_tokenizer')
tok.load(model_path='example/char_tokenizer.model')
