import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.keras.utils.vis_utils import plot_model
from torch import nn
from torchsummary import summary
from transformers import BigBirdModel
import tensorflow as tf


# This is a file that can help you to output the data of models.


''' Transform the data output from tensorflow
with open('output_file/BERT-Tokenizer.out') as f:
    data = f.readlines()
judgement = ['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy']
value = []
for _ in data:
    pass_count = 0
    v = []
    for i in judgement:
        try:
            a = _.split(f'{i}: ')[1]
            a = a.split(' -')[0]
            a = a.split(f'\n')[0]
            pass_count += 1
            v.append(float(a))
        except Exception as sb:
            pass
    if pass_count == 4:
        value.append(v)

pd.DataFrame.to_csv(DataFrame(value), 'value.csv', index=False, header=False)
'''

''' Transform the data output from PyTorch
def calc_f1_score(tag, predict, target):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for _ in range(len(predict)):
        if target[_] == predict[_]:
            if target[_]:
                tp += 1
            else:
                tn += 1
        else:
            if predict[_]:
                fp += 1
            else:
                fn += 1
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    f1_score = 2 * P * R / (P + R)
    print(f'F1-Score of {tag}: {f1_score}')
    return [tp, fp], [fn, tn]


tag = ['IE', 'NS', 'FT', 'JP']
f1_matrix = []
pred = []
targ = []
macro = []
for i in range(4):
    _pred = []
    _targ = []
    with open(f'output_file/{tag[i]}_BigBird.out', encoding='utf-8') as f:
        raw_text = f.readlines()
    for _ in raw_text:
        try:
            _pred.append(int(_.split('tensor([')[1][0]))
            _targ.append(int(_.split('tensor([')[2][0]))
        except Exception as sb:
            pass
    _ = calc_f1_score(tag[i], _pred, _targ)
    f1_matrix.append(_[0])
    f1_matrix.append(_[1])
    pred.append(_pred)
    targ.append(_targ)
    macro.append([f1_score(_pred, _targ, average='macro'), 0])
pred_int = []
targ_int = []
for _ in range(len(pred[0])):
    pred_int.append(
        ((8 * pred[0][_] +
          4 * pred[1][_] +
          2 * pred[2][_] +
          1 * pred[3][_]) + 8) % 16)
for _ in range(len(targ[0])):
    targ_int.append(
        ((8 * targ[0][_] +
          4 * targ[1][_] +
          2 * targ[2][_] +
          1 * targ[3][_]) + 8) % 16)
pred_table = [[0 for i in range(16)] for j in range(16)]
for _ in range(len(pred_int)):
    pred_table[targ_int[_]][pred_int[_]] += 1
macro.append([f1_score(targ_int, pred_int, average='macro'), 0])
for i in macro:
    f1_matrix.append(i)
print(f1_matrix)
with pd.ExcelWriter('csv_output/BigBird.xlsx') as f:
    DataFrame(pred_table).to_excel(f, sheet_name='Prediction', index=False, header=False)
    DataFrame(f1_matrix).to_excel(f, sheet_name='F1-Score Data', index=False, header=False)
'''

''' For test
pred_table = [[0 for i in range(16)] for j in range(16)]
pred = [0]
targ = [1]
for _ in range(len(pred)):
    pred_table[pred[_]][targ[_]] += 1
pd.DataFrame.to_csv(DataFrame(pred_table), 'for_test.csv', index=False, header=False)
'''


''' Output PyTorch model construct
class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BigBirdModel.from_pretrained('', cache_dir="./BigBird_roBERTa_Base")
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


device = torch.device('cuda')
model = torch.load('saved_models/IE_epoch_17.model')
print(model)
summary(model)

# Output TF model construct
model = tf.keras.models.load_model('saved_models/bert_tokenize')
model.summary()
'''

# Calc BigBird Loss
sigmoid = nn.Sigmoid()
loss = nn.BCELoss()
personal_type = ['IE', 'NS', 'FT', 'JP']
t = [[] for _ in range(4)]
p = [[] for _ in range(4)]
for i in range(4):
    with open(f'output_file/{personal_type[i]}_BigBird_Lite.out') as f:
        data = f.readlines()
    epoch = 0
    for _ in range(len(data)):
        if 'finished.' in data[_]:
            epoch += 1
            _loss = 0
            count = 0
            correct = 0
            while True:
                _ += 1
                try:
                    pred = sigmoid(torch.Tensor([float(data[_].split('tensor([[')[1].split(',')[0])]))
                    targ = torch.tensor([float(data[_].split(") tensor([[")[1].split(',')[0])])
                    '''
                    print(loss(pred, targ))
                    print(pred, targ)
                    '''
                    _loss += loss(pred, targ).item()
                    pred = 1 if pred.item() >= 0.5000 else 0
                    correct += 1 if pred == targ.item() else 0
                    count += 1
                    if epoch == 10:
                        t[i].append(int(targ.item()))
                        p[i].append(pred)
                except Exception as sb:
                    # print(_loss / count)
                    # print(correct, count, correct / count)
                    break
    print()

predict = []
target = []
for _ in range(len(t[0])):
    lb_int = 0
    lb_int += 8 if t[0][_] else 0
    lb_int += 0 if t[1][_] else 4
    lb_int += 0 if t[2][_] else 2
    lb_int += 0 if t[3][_] else 1
    target.append(lb_int)
for _ in range(len(p[0])):
    lb_int = 0
    lb_int += 8 if p[0][_] else 0
    lb_int += 0 if p[1][_] else 4
    lb_int += 0 if p[2][_] else 2
    lb_int += 0 if p[3][_] else 1
    predict.append(lb_int)

predict_table = [[0 for i in range(16)] for j in range(16)]
for _ in range(len(predict)):
    predict_table[predict[_]][target[_]] += 1
for _ in predict_table:
    print(_)

pd.DataFrame.to_csv(DataFrame(predict_table), 'csv_output/BigBird_Lite.csv', index=False, header=False)
