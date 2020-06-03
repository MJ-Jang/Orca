from orca import OrcaTypoProcessor
from orca.utils.hangeul import normalize_unicode

aa = OrcaTypoProcessor(unigram_dict_path='data/test_uni.txt')
text = '데이터 선물하려면 뭘 헤야 할까요?'

import time
t = time.time()
text = normalize_unicode(text)
aa.process(text)


print(time.time() - t)



aa._infer_detection('데이터 공유한 기록 좀 보요줘')
