from orca.correction import SymDeletingTypoCorrecter

aa = SymDeletingTypoCorrecter()
aa.train('../data/train_data.txt', 'data/', 'test_uni', 'test_bi')
aa.load_model('./data/test_uni.txt', './data/test_bi.txt')
print(aa.infer('인청'))

