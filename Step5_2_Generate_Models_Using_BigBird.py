import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BigBirdModel, BigBirdTokenizer
import torch.nn as nn
from torch.autograd import Variable
import time

current_job = 3  # (0, 1, 2, 3 -> I, N, F, J)
start_epoch = 1
learning_rate = 1e-5
token_length = 1024
accumulation_steps = 4
epoch = 5
batch_size = 8
TOTAL_SIZE = 13538  # 13538
TRAINSET_SIZE = 12175
type = ['IE', 'NS', 'FT', 'JP']
model_path = f'saved_models/BigBird_{type[current_job]}_epoch_{start_epoch}.model'


def transform_label(lb_int):
    lb_list = []
    for i in range(4):
        if lb_int % 2 == 0:
            lb_list.append(0)
            lb_int /= 2
        else:
            lb_list.append(1)
            lb_int = lb_int // 2
    return lb_list[3 - current_job]


model_name = 'BigBird_roBERTa_Base'  # 指定需下载的预训练模型参数

tokenizer = BigBirdTokenizer.from_pretrained(model_name, cache_dir="BigBird_roBERTa_Base")

data_set = pd.read_csv(f"data/mbti_token.csv")

pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ',
              'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']

input_ids = [list(map(int, data_set['token'][i].split(','))) for i in range(TOTAL_SIZE)]
input_labels = [transform_label(data_set['label'][i]) for i in range(TOTAL_SIZE)]

EVALSET_SIZE = TOTAL_SIZE - TRAINSET_SIZE

train_samples = input_ids[:TRAINSET_SIZE]
train_labels = input_labels[:TRAINSET_SIZE]
eval_samples = input_ids[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]
eval_labels = input_labels[TRAINSET_SIZE:TRAINSET_SIZE + EVALSET_SIZE]


def get_dummies(l, size=2):
    res = list()
    for i in l:
        tmp = [0] * size
        tmp[i] = 1
        res.append(tmp)
    return res


for j in range(len(input_ids)):
    # 将样本数据填充至长度为 512
    i = input_ids[j]
    if len(i) <= token_length:
        input_ids[j].extend([0] * (token_length - len(i)))
    else:
        input_ids[j] = input_ids[j][:token_length]

input_ids = train_samples
input_labels = get_dummies(train_labels)

train_set = TensorDataset(torch.LongTensor(input_ids),
                          torch.FloatTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

input_ids = eval_samples
input_labels = get_dummies(eval_labels)

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
        x = outputs[1]
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x


def predict(logits):
    res = torch.argmax(logits, 1)
    return res


cls = fn_cls()
cls = torch.load(model_path)
cls.to(device)

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(cls.parameters(), lr=learning_rate)

pre = time.time()

''' The code below is for TRAIN, DO NOT run it if you just want to EVALUATE the model
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

        pred = predict(output)

        loss = criterion(sigmoid(output).view(-1, 2), target)
        print(sigmoid(output).view(-1, 2))
        print(target)

        # 梯度积累
        loss = loss / accumulation_steps
        loss.backward()

        if ((batch_idx + 1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        if ((batch_idx + 1) % accumulation_steps) == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
                print(output)
                pred = predict(output)

                correct += (pred == target).sum().item()
                total += len(data)

            print('{} / {} Correct，Acc: {:.2f}%'.format(
                correct, total, 100. * correct / total))
            torch.save(cls, f'BigBird_{type[current_job]}_epoch_{i + 1 + start_epoch}.model')
'''
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

    correct += (pred == target)[:, 0].item()
    total += len(data)

    print(pred, target)

print('{} / {} Correct，Acc: {:.2f}%'.format(
    correct, total, 100. * correct / total))
