import os
import re

import pandas as pd
import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from transformers import BigBirdModel, BigBirdTokenizer
import warnings

server_mode = True
model_token_length = 1024
warnings.filterwarnings("ignore")


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BigBirdModel.from_pretrained('', cache_dir="BigBird_roBERTa_Base")
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


model_name = 'BigBird_roBERTa_Base'
tokenizer = BigBirdTokenizer.from_pretrained(model_name, cache_dir="BigBird_roBERTa_Base")
device = torch.device('cuda')

read_path = 'web_core/text.txt'
write_path = 'web_core/res.txt'


class Predict:
    def __init__(self, input_text, max_token_length):
        self.input_tokens = None
        self.input_ids = None
        self.input_text = input_text
        self.max_token_length = max_token_length
        self.device = torch.device('cuda')

    def rebuild_token(self):
        current_length = 0
        new_ids = []
        temp = []
        for i in self.input_ids:
            if len(i) + current_length <= self.max_token_length:
                for ids in i:
                    temp.append(ids)
                current_length += len(i)
            else:
                new_ids.append(temp)
                temp = i
                current_length = len(i)
        if len(temp) >= 10:
            new_ids.append(temp)
        for i in new_ids:
            for _ in range(self.max_token_length - len(i)):
                i.append(0)
        return new_ids

    def prediction(self):
        self.input_text = re.sub('\\|\\|\\|', '', self.input_text)
        self.input_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|%[0-9a-fA-F][0-9a-fA-F])+', ' [MASK] ',
                                 self.input_text)
        self.input_text = re.sub(r'\d+', '*', self.input_text)
        self.input_text = re.sub(r'([a-z]|_|-)\1{2,}', r'\1', self.input_text)
        self.input_text = re.sub('\\*', ' [MASK] ', self.input_text)
        self.input_text = re.sub(r'(\s)\1+', ' ', self.input_text)  # remove multi spaces
        self.input_text = re.sub(r'(\S)\1{2,}[\s|\w]*', '', self.input_text)

        self.input_text = self.input_text.split('.')
        self.input_text = [_ + '.' for _ in self.input_text]

        self.input_tokens = [tokenizer.tokenize(_) for _ in self.input_text]
        self.input_ids = [tokenizer.convert_tokens_to_ids(_) for _ in self.input_tokens]
        print(self.input_tokens)
        print(self.input_ids)

        data_split = self.rebuild_token()

        sigmoid = nn.Sigmoid()
        lb = ['IE', 'NS', 'FT', 'JP']
        pre_type = []
        result = []

        for _ in range(4):
            model = torch.load(f'saved_models/BigBird_{lb[_]}_epoch_10.model')
            model.to(self.device)
            model.eval()

            count = 0
            value = 0
            for __ in data_split:
                count += 1
                data = torch.LongTensor([__])
                data = data.to(self.device)

                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                mask = torch.Tensor(mask).to(self.device)

                output = model(data, attention_mask=mask)
                value += sigmoid(output)[:, 0].item()
                print(sigmoid(output)[:, 0].item())
            print(f'Label: {lb[_]} Value: {"%.4f" % (value / count)}')
            pre_type.append(lb[_][1] if value / count < 0.5 else lb[_][0])
            result.append(value / count * 100)
        # Swap I/E label
        if pre_type[0] == 'I':
            pre_type[0] = 'E'
        else:
            pre_type[0] = 'I'
        result[0] = 100 - result[0]
        print(pre_type)
        print(result)
        type_bool = [1 if pre_type[_] in 'INFJ' else 0 for _ in range(4)]
        print(type_bool)
        return type_bool, result


if __name__ == '__main__':
    if server_mode:
        while True:
            try:
                with open(read_path) as f:
                    text = f.read()
                    tag, res = Predict(text, model_token_length).prediction()
                os.remove(read_path)
                with open(write_path, 'w', encoding='utf-8') as f:
                    f.write(f'{tag[0]}\n{tag[1]}\n{tag[2]}\n{tag[3]}\n{res[0]}\n{res[1]}\n{res[2]}\n{res[3]}')
            except Exception as sb:
                pass
    else:
        while True:
            token = input()
            Predict(token, model_token_length).prediction()
