import re

import pandas as pd
from pandas import DataFrame
from transformers import BigBirdTokenizer

token_length = 1024

data = pd.read_csv('data/mbti.csv')
tokenizer = BigBirdTokenizer.from_pretrained("BigBird_roBERTa_Base")
_dataset = []
_label = []
processed_dataset = [[] for i in range(16)]
processed_dataset_int = []

x = [[] for _ in range(16)]

pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ',
              'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']


def transform_label(lb):
    lb_int = 0
    lb_int += 8 if 'I' in lb else 0
    lb_int += 4 if 'S' in lb else 0
    lb_int += 2 if 'T' in lb else 0
    lb_int += 1 if 'P' in lb else 0
    return lb_int


print(transform_label('ENTP'))
exit(0)


def data_process(posts):
    posts = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [MASK] ', posts)
    posts = re.sub(r'\d+', '*', posts)
    posts = re.sub(r'([a-z]|_|-)\1{2,}', r'\1', posts)
    # posts = re.sub('\\|\\|\\|', '', posts)
    for __ in pers_types:
        posts = re.sub(__, ' [MASK] ', posts)
        posts = re.sub(__.lower(), ' [MASK] ', posts)
    posts = re.sub('\\*', ' [MASK] ', posts)
    posts = re.sub(r'(\s)\1+', ' ', posts)  # remove multi spaces
    posts = re.sub(r'(\S)\1{2,}[\s|\w]*', '', posts)
    # posts = re.sub(r'\\.\\.\\.', '.', posts)
    # posts = re.sub(r'\\.\\.', '.', posts)
    mask_count = str.count(posts, '[MASK]')
    try:
        if posts[0] == "'":
            posts = posts[1:]
        if posts[-1] == "'":
            posts = posts[:-1]
        if posts[-1] not in '?()_.,':
            posts += '.'
    except Exception as sb:
        pass
    return posts if len(posts.split(' ')) > 3 and mask_count <= 0.5 * len(posts.split(' ')) else ' '


for _ in range(len(data)):
    _dataset.append(data['posts'][_])
    _label.append(data['type'][_])

for i in range(len(_dataset)):
    post_slice = _dataset[i].split('|||')
    for _ in post_slice:
        __ = data_process(_)
        if __ != ' ':
            processed_dataset[transform_label(_label[i])].append(__)

for i in range(16):
    _ = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(processed_dataset[i][j]))
         for j in range(len(processed_dataset[i]))]
    processed_dataset_int.append(_)
for i in range(16):
    temp_seq = []
    fin = False
    for _ in processed_dataset_int[i]:
        if len(_) + len(temp_seq) <= token_length:
            for __ in _:
                temp_seq.append(__)
        else:
            x[i].append(temp_seq)
            temp_seq = []
            for __ in _:
                temp_seq.append(__)
            fin = True
    if not fin:
        x[i].append(temp_seq)

train_X = []
train_Y = []
test_X = []
test_Y = []

for _ in range(16):
    EVALSET_SIZE = len(x[_]) // 10 + 1
    TRAINSET_SIZE = len(x[_]) - EVALSET_SIZE

    train_samples = x[_][:TRAINSET_SIZE]
    train_labels = _
    eval_samples = x[_][:EVALSET_SIZE]
    eval_labels = _

    for i in eval_samples:
        test_X.append(i)
        test_Y.append(_)

    for i in train_samples:
        train_X.append(i)
        train_Y.append(_)

train_X += test_X
train_Y += test_Y

train_X_processed = [str(i)[1:-1] for i in train_X]

pd.DataFrame.to_csv(DataFrame({"label": train_Y, "token": train_X_processed}), 'data/mbti_token.csv', index=False)

'''
Train: 13538
Total: 12175
'''
