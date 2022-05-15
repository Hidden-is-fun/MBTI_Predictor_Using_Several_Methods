import re
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BigBirdModel, BigBirdTokenizer
import torch.nn as nn
from torch.autograd import Variable
import time
import os

current_job = 3
start_epoch = 17
learning_rate = 1e-5
token_length = 1536
model_path = 'saved_models/JP_epoch_10.model'
type = ['IE', 'NS', 'FT', 'JP']

model_name = 'BigBird_roBERTa_Base'  # 指定需下载的预训练模型参数

tokenizer = BigBirdTokenizer.from_pretrained(model_name, cache_dir="BigBird_roBERTa_Base")

data_set = pd.read_csv(f"data/mbti_{type[current_job]}.csv")

pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ',
              'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']

_dataset = []
_label = []


def data_process(posts):
    posts = re.sub(r'https?://.*?[\s+]', '[MASK]', posts)
    posts = re.sub(r'http?://.*?[\s+]', '[MASK]', posts)
    # posts = re.sub(r'\d+', '*', posts)
    posts = re.sub(r'([a-z]|_|-)\1{2,}', r'\1', posts)
    posts = re.sub('\\|\\|\\|', '', posts)
    posts = re.sub(r'(\s)\1+', ' ', posts)  # remove multi spaces
    if posts[0] == "'":
        posts = posts[1:-1]
    for __ in pers_types:
        posts = re.sub(__, '[MASK]', posts)
        posts = re.sub(__.lower(), '[MASK]', posts)
    return posts


for _ in range(len(data_set)):
    _dataset.append(data_process(data_set['text'][_]))
    _label.append(data_set['label'][_])

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
    if len(i) <= token_length:
        input_ids[j].extend([0] * (token_length - len(i)))
    else:
        input_ids[j] = input_ids[j][:token_length]

# 构建数据集和数据迭代器，设定 batch_size 大小为 4
train_set = TensorDataset(torch.LongTensor(input_ids),
                          torch.FloatTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=1,
                          shuffle=True)

tokenized_text = [tokenizer.tokenize(i) for i in eval_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = eval_labels

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) <= token_length:
        input_ids[j].extend([0] * (token_length - len(i)))
    else:
        input_ids[j] = input_ids[j][:token_length]

eval_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.FloatTensor(input_labels))
eval_loader = DataLoader(dataset=eval_set,
                         batch_size=1,
                         shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BigBirdModel.from_pretrained(model_name, cache_dir="./BigBird_roBERTa_Base")
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
# cls = torch.load(model_path)
cls.to(device)

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(cls.parameters(), lr=learning_rate)

pre = time.time()

accumulation_steps = 16
epoch = 5

for i in range(epoch):
    cls.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target.view(-1, 2)).to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = cls(data, attention_mask=mask)
        print(output)
        print(sigmoid(output))
        print(target)
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
                i + 1 + start_epoch, batch_idx, len(train_loader), 100. *
                batch_idx / len(train_loader), loss.item()
            ))
        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print(f'Epoch {i + 1 + start_epoch} finished.')
            cls.eval()

            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(eval_loader):
                data = data.to(device)
                target = target.long().to(device)

                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                mask = torch.Tensor(mask).to(device)

                output = cls(data, attention_mask=mask)
                pred = predict(output)

                correct += (pred == target).sum().item()
                total += len(data)

            print('{} / {} Correct，Acc：{:.2f}%'.format(
                correct, total, 100. * correct / total))
            torch.save(cls, f'{type[current_job]}_epoch_{i + 1 + start_epoch}.model')

model = torch.load(model_path)
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

    print(pred, target)
    correct += (pred == target).sum().item()
    total += len(data)

# 准确率应该达到百分之 90 以上
print('{}二分类，正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}% {}'.format(
    type[current_job], correct, total, 100. * correct / total, model_path))
