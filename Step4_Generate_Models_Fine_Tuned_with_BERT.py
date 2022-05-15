import re
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import BertTokenizer
import torch.nn as nn
from pytorch_transformers import BertModel
from torch.autograd import Variable
import time

model_name = 'bert-base-uncased'  # 指定需下载的预训练模型参数

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

data_set = pd.read_csv("data/mbti.csv")

pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ',
              'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
pers_types = [p.lower() for p in pers_types]

_dataset = []
_label = []


def data_process(posts):
    posts = posts.lower()
    posts = re.sub(r'https?://.*?[\s+]', '* ', posts)
    posts = re.sub(r'http?://.*?[\s+]', '* ', posts)
    posts = re.sub(r'\d+', '*', posts)
    posts = re.sub(r'([a-z]|_|-)\1{2,}', r'\1', posts)
    posts = re.sub('\\.\\.\\.', '', posts)
    posts = re.sub(r'(\s)\1+', ' ', posts)  # remove multi spaces
    if posts[0] == "'":
        posts = posts[1:-1]
    for __ in pers_types:
        posts = re.sub(__, '*', posts)
    return posts


def remove_useless_posts(post):
    count = 1
    post = post.split('|||')
    _sp = ''
    for _ in post:
        _ss = ''
        try:
            single_post = _.split('.')
        except Exception as err:
            print(err)
            single_post = _
        for __ in single_post:
            single_sentence = re.sub(r'[^a-zA-Z\s]', ' ', __)
            try:
                if __[0] == ' ':
                    __ = __[1:]
            except Exception as err:
                pass
            if len(single_sentence.split()) >= 5:
                _ss += f'{__}. '
        if len(_sp.split(' ')) + len(_ss.split(' ')) <= 300:
            _sp += _ss
        else:
            #  print(_sp)
            _dataset.append(_sp)
            count += 1
            _sp = ''
            _sp += _ss
    #  print(_sp)
    _dataset.append(_sp)
    return count


for _ in range(len(data_set)):
    data = data_process(data_set['posts'][_])
    x = remove_useless_posts(data)
    for __ in range(x):
        _label.append(0 if data_set['type'][_][2] == 'F' else 1)

dataset = np.array(_dataset)
labels = np.array(_label)

TOTAL_SIZE = len(dataset)

np.random.seed(10)
mix_index = np.random.choice(TOTAL_SIZE, TOTAL_SIZE)
dataset = dataset[mix_index]
labels = labels[mix_index]

TRAINSET_SIZE = int(0.9 * TOTAL_SIZE)
EVALSET_SIZE = TOTAL_SIZE - TRAINSET_SIZE

train_samples = dataset[:TRAINSET_SIZE]
train_labels = labels[:TRAINSET_SIZE]
eval_samples = dataset[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]
eval_labels = labels[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]


def get_dummies(l, size=2):
    res = list()
    for i in l:
        tmp = [0] * size
        tmp[i] = 1
        res.append(tmp)
    return res


tokenized_text = [tokenizer.tokenize(i) for i in train_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = get_dummies(train_labels)  # 使用 get_dummies 函数转换标签

for j in range(len(input_ids)):
    # 将样本数据填充至长度为 512
    i = input_ids[j]
    if len(i) <= 512:
        input_ids[j].extend([0] * (512 - len(i)))
    else:
        input_ids[j] = input_ids[j][:512]

# 构建数据集和数据迭代器，设定 batch_size 大小为 4
train_set = TensorDataset(torch.LongTensor(input_ids),
                          torch.FloatTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=32,
                          shuffle=True)

tokenized_text = [tokenizer.tokenize(i) for i in eval_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = eval_labels

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) <= 512:
        input_ids[j].extend([0] * (512 - len(i)))
    else:
        input_ids[j] = input_ids[j][:512]

eval_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.FloatTensor(input_labels))
eval_loader = DataLoader(dataset=eval_set,
                         batch_size=1,
                         shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BertModel.from_pretrained('./bert-base-uncased')
        self.model.to(device)
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(768, 2)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[1]  # 取池化后的结果 batch * 768
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x


def predict(logits):
    res = torch.argmax(logits, 1)
    return res


cls = fn_cls()
cls.to(device)
cls.train()

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(cls.parameters(), lr=1e-5)

pre = time.time()

accumulation_steps = 4
epoch = 10

for i in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target.view(-1, 2)).to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = cls(data, attention_mask=mask)
        pred = predict(output)

        loss = criterion(sigmoid(output).view(-1, 2), target)

        # 梯度积累
        loss = loss / accumulation_steps
        loss.backward()

        if ((batch_idx + 1) % accumulation_steps) == 0:
            # 每 8 次更新一下网络中的参数
            optimizer.step()
            optimizer.zero_grad()

        if ((batch_idx + 1) % accumulation_steps) == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                i + 1, batch_idx, len(train_loader), 100. *
                batch_idx / len(train_loader), loss.item()
            ))
        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print('labels:', target)
            print('pred:', pred)

print('训练时间：', time.time() - pre)
torch.save(cls, 'FT.model')

model = torch.load('FT.model')
model.eval()

correct = 0
total = 0

for batch_idx, (data, target) in enumerate(eval_loader):
    data = data.to(device)
    target = target.long().to(device)

    mask = []
    for sample in data:
        mask.append([1 if i != 0 else 0 for i in sample])
    mask = torch.Tensor(mask).to(device)

    output = model(data, attention_mask=mask)
    pred = predict(output)

    correct += (pred == target).sum().item()
    total += len(data)

# 准确率应该达到百分之 90 以上
print('正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%'.format(
    correct, total, 100. * correct / total))
