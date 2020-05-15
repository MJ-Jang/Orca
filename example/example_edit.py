from orca.typo_corrector.non_words import EditDistTypoCorrecter

corrector = EditDistTypoCorrecter()
corrector.load_model(model_path='example/test_vocab.vocab')
corrector.infer(sent='ㅠㅡ랑스로 여행간다', threshold=2)