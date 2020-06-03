from orca.resource_manager import UnigramDictManager

manager = UnigramDictManager(file_path='data/test_uni.txt')

words = [
    '가능해?',
    '좋아?',
    '있어?',
    '어떻게',
    '있을까?',
    '좋아?',
    '얼마야?',
    "알고싶어",
    "싶어",
    "말해줘봐",
    "모르겠어",
    "싶습니다",
    "잔여량은?",
    "확인할까?",
    "좋나요?",
    "해주나요?",
    "얼마해줘?",
    "지난주",
    "이번",
    "제작년",
    "내년",
    "미래에",
    "끝",
    "매",
    "다음",
    "하루평",
    "추후에",
    "해마",
    "어",
    "제1",
    "제2",
    "뭐니?",
    "뭐야?",
    "궁금해",
    "누구야?",
    "등록해?",
    "등록하냐?",
    "등록하려고",
    "등록할래",
    "방법",
    "보내고",
    "보낼게",
    "보여주십시오",
    "좋겠는데",
    "알려줄래?",
    "SK",
    "SKT",
    "삼성",
    "후져",
    "했어?",
    "돼냐",
    "돼냐?",
    "문의",
    "전전주"
]

for w in words:
    manager.create(w, count=3000)
manager.save_dict(local_path='data/unigram_dict.txt')