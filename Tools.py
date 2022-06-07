import pandas as pd
import torch
from graphviz import Digraph
from pandas import DataFrame
from pytorch_transformers import BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.keras.utils.vis_utils import plot_model
from torch import nn
from torch.autograd import Variable
from torchsummary import summary
from transformers import BigBirdModel
import tensorflow as tf

# This is a file that can help you to output the data of models.


''' Transform the data output from tensorflow
with open('output_file/Bert_Tokenizer.out') as f:
    data = f.readlines()
# judgement = ['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy']
judgement = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
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
            print(sb)
    if pass_count == 4:
        value.append(v)

pd.DataFrame.to_csv(DataFrame(value), 'csv_output/bert_tokenizer_data.csv', index=False, header=False)
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


# Output PyTorch model construct
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BertModel.from_pretrained('bert_base_uncased')
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
model = fn_cls()
model.eval()
# model = torch.load('saved_models/IE9.model')
print(model)
summary(model)
'''

# Output TF model construct
model = tf.keras.models.load_model('saved_models/bert_tokenize')
model.summary()
'''

''' Calc BigBird Loss
sigmoid = nn.Sigmoid()
loss = nn.BCELoss()
personal_type = ['IE', 'NS', 'FT', 'JP']
t = [[] for _ in range(4)]
p = [[] for _ in range(4)]
val_loss = [[] for _ in range(4)]
val_acc = [[] for _ in range(4)]
train_loss = [[] for _ in range(4)]
train_loss_ = [[] for _ in range(4)]
for i in range(4):
    with open(f'output_file/{personal_type[i]}_BigBird_Lite.out') as f:
        data = f.readlines()
    epoch = 0
    loss_count = 0
    t_l = 0
    for _ in range(len(data)):
        if 'Loss: ' in data[_]:
            train_loss_[i].append(float(data[_].split("Loss: ")[1]))
            loss_count += 1
            t_l += float(data[_].split("Loss: ")[1])
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
                    # print(loss(pred, targ))
                    # print(pred, targ)
                    _loss += loss(pred, targ).item()
                    pred = 1 if pred.item() >= 0.5000 else 0
                    correct += 1 if pred == targ.item() else 0
                    count += 1
                    if (epoch == 10 and i in [1, 3]) or (epoch == 9 and i in [0, 2]):
                        t[i].append(int(targ.item()))
                        p[i].append(pred)
                except Exception as sb:
                    # print(_loss / count)
                    val_loss[i].append(_loss / count)
                    # print(correct, count, correct / count)
                    val_acc[i].append(correct / count)
                    train_loss[i].append(t_l / loss_count)
                    loss_count = 0
                    t_l = 0
                    break
    print()

predict = []
target = []
for _ in range(len(t[0])):
    lb_int = 0
    lb_int += 0 if t[0][_] else 8
    lb_int += 0 if t[1][_] else 4
    lb_int += 0 if t[2][_] else 2
    lb_int += 0 if t[3][_] else 1
    target.append(lb_int)
for _ in range(len(p[0])):
    lb_int = 0
    lb_int += 0 if p[0][_] else 8
    lb_int += 0 if p[1][_] else 4
    lb_int += 0 if p[2][_] else 2
    lb_int += 0 if p[3][_] else 1
    predict.append(lb_int)

predict_table = [[0 for i in range(16)] for j in range(16)]
for _ in range(len(predict)):
    predict_table[predict[_]][target[_]] += 1
for _ in predict_table:
    print(_)
# pd.DataFrame.to_csv(DataFrame(val_loss), 'csv_output/BigBird_Lite_val_loss.csv', index=False, header=False)
# pd.DataFrame.to_csv(DataFrame(val_acc), 'csv_output/BigBird_Lite_val_acc.csv', index=False, header=False)
# pd.DataFrame.to_csv(DataFrame(train_loss), 'csv_output/BigBird_Lite_train_loss.csv', index=False, header=False)
# pd.DataFrame.to_csv(DataFrame(train_loss_), 'csv_output/BigBird_Lite_train_loss_.csv', index=False, header=False)
pd.DataFrame.to_csv(DataFrame(predict_table), 'csv_output/BigBird_Lite.csv', index=False, header=False)
'''

''' Output BERT (Fine Tune) data
tag = ['IE', 'NS', 'FT', 'JP']
pred = [[] for _ in range(4)]
targ = [[] for _ in range(4)]
for _ in range(4):
    with open(f'output_file/BERT_{tag[_]}.out', encoding='utf-8') as f:
        data = f.readlines()
    for i in data:
        try:
            x = i.split('tensor([')
            targ[_].append(int(x[1][0]))
            pred[_].append(int(x[2][0]))
        except:
            pass
predict = []
target = []
for _ in range(len(targ[0])):
    p = 0
    t = 0
    p += 0 if pred[0][_] else 8
    p += 4 if pred[1][_] else 0
    p += 2 if pred[2][_] else 0
    p += 1 if pred[3][_] else 0
    t += 0 if targ[0][_] else 8
    t += 4 if targ[1][_] else 0
    t += 2 if targ[2][_] else 0
    t += 1 if targ[3][_] else 0
    predict.append(p)
    target.append(t)
print(predict)
print(target)
predict_table = [[0 for i in range(16)] for j in range(16)]
for _ in range(len(predict)):
    predict_table[predict[_]][target[_]] += 1
for _ in predict_table:
    print(_)
pd.DataFrame.to_csv(DataFrame(predict_table), 'csv_output/BERT.csv', index=False, header=False)
'''
