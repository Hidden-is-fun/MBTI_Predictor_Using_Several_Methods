import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from transformers import BigBirdModel, BigBirdTokenizer

model_name = 'BigBird_roBERTa_Base'
text = 'Process finished with exit code 0'

tokenizer = BigBirdTokenizer.from_pretrained(model_name, cache_dir="BigBird_roBERTa_Base")
input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
print(input_ids)
if len(input_ids) < 1024:
    for i in range(1024 - len(input_ids)):
        input_ids.append(0)
else:
    input_ids = input_ids[:1024]


class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BigBirdModel.from_pretrained('', cache_dir="BigBird_roBERTa_Base")
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


eval_set = TensorDataset(torch.LongTensor([input_ids]),
                         torch.FloatTensor([0]))
eval_loader = DataLoader(dataset=eval_set,
                         batch_size=1,
                         shuffle=False)
for _ in eval_loader:
    print(_)

device = torch.device('cuda')
lb = ['IE', 'NS', 'FT', 'JP']
for _ in range(4):
    model = torch.load(f'saved_models/BigBird_{lb[_]}_epoch_10.model')
    model.to(device)
    model.eval()

    for batch_idx, (data, target) in enumerate(eval_loader):
        data = data.to(device)
        target = target.long().to(device)

        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = model(data, attention_mask=mask)
        pred = torch.argmax(output)
        print(pred)

