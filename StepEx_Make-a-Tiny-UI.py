import random
import re
import sys
import time
from threading import Thread

import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QPlainTextEdit, QPushButton, QLabel, QRadioButton, QDialog, QCheckBox
from qt_material import apply_stylesheet
from torch import nn
from transformers import BigBirdModel, BigBirdTokenizer

tokenizer = BigBirdTokenizer.from_pretrained("BigBird_roBERTa_Base")
device = torch.device('cuda')


class Predict:
    # *** This Class is copied from 'Step5_3_Make_a_Prediction.py' *** #
    def __init__(self, input_text, max_token_length):
        self.input_tokens = None
        self.input_ids = None
        self.input_text = input_text
        self.max_token_length = max_token_length

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
            model.to(device)
            model.eval()

            count = 0
            value = 0
            for __ in data_split:
                count += 1
                data = torch.LongTensor([__])
                data = data.to(device)

                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                mask = torch.Tensor(mask).to(device)

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


def sss(widget, size=20, style=0, font="default"):  # setStyleSheet
    css = f"font-size:{size}px;"
    if style:
        if style == 1:
            css += "font-weight:bold;"
        else:
            css += "font-weight:light;"
    if font == 'default':
        css += "font-family:JetBrains Mono;"
    else:
        css += f"font-family:{font};"
    widget.setStyleSheet(css)


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


class Test(QWidget):
    def __init__(self):
        super(Test, self).__init__()
        self.setWindowTitle('MBTI Personality Predictor')

        self.MainWidget = QWidget(self)
        self.setFixedSize(1210, 850)

        title = QLabel('MBTI Personality Predictor', self.MainWidget)
        title.move(25, 20)
        sss(title, 30, 1)

        self.textEdit = QPlainTextEdit(self.MainWidget)
        self.textEdit.setPlaceholderText("Input something here to analyze")
        self.textEdit.move(25, 90)
        self.textEdit.resize(550, 575)
        sss(self.textEdit)

        btn_line_temp = QPushButton(self.MainWidget)
        btn_line_temp.setGeometry(20, 70, 700, 2)

        _ = []
        for _i in range(3):
            __ = QLabel(self.MainWidget)
            __.setGeometry(650, 262 + 180 * _i, 500, 1)
            __.setStyleSheet('background:#202224;')
            _.append(__)

        self.btn_confirm = QPushButton('Confirm', self.MainWidget)
        self.btn_confirm.resize(150, 50)
        self.btn_confirm.move(25, 690)
        sss(self.btn_confirm)
        self.btn_confirm.clicked.connect(self.confirm)

        self.btn_cancel = QPushButton('Clear', self.MainWidget)
        self.btn_cancel.resize(150, 50)
        self.btn_cancel.move(190, 690)
        sss(self.btn_cancel)
        self.btn_cancel.clicked.connect(self.clear)

        self.result = QLabel(self.MainWidget)
        self.result.resize(550, 50)
        self.result.move(350, 740)
        sss(self.result)

        _keyword = [['Introverted', 'Extraverted'],
                    ['Intuitive', 'Sensitive'],
                    ['Feeling', 'Thinking'],
                    ['Judging', 'Prospecting']]
        self.value = []
        self.keyword = []
        for _i in range(4):
            self.value.append([])
            self.keyword.append([])
            for _j in range(2):
                keyword = QLabel(self.MainWidget)
                keyword.setText(_keyword[_i][_j])
                keyword.move(1030 if _j else 620, 210 + 185 * _i)
                keyword.resize(150, 40)
                sss(keyword)
                value = QLabel(self.MainWidget)
                value.setText('N/A')
                value.move(1100 if _j else 600, 165 + 185 * _i)
                value.resize(100, 40)
                sss(value)
                value.setAlignment(Qt.AlignCenter)
                keyword.setAlignment(Qt.AlignRight if _j else Qt.AlignLeft)
                self.value[_i].append(value)
                self.keyword[_i].append(keyword)

        _label = ['Mind', 'Energy', 'Nature', 'Tactics']
        self.label = []
        for _i in range(4):
            label = QLabel(self.MainWidget)
            label.setText(_label[_i])
            label.resize(500, 100)
            label.setAlignment(Qt.AlignCenter)
            sss(label, 28, 1)
            label.move(660, 60 + 185 * _i)
            self.label.append(label)

        _desc = ['This trait determines how we interact with our environment.',
                 'This trait shows where we direct our mental energy.',
                 'This trait determines how we make decisions and cope with emotions.',
                 'This trait reflects our approach to work, planning and decision-making.']
        self.desc = []
        for _i in range(4):
            desc = QLabel(self.MainWidget)
            desc.setText(_desc[_i])
            desc.resize(600, 100)
            desc.setAlignment(Qt.AlignCenter)
            desc.setWordWrap(True)
            sss(desc, 14)
            desc.move(600, 95 + 185 * _i)
            self.desc.append(desc)

        self.bar = []
        for _i in range(4):
            bar = QLabel(self.MainWidget)
            bar.resize(400, 10)
            bar.setStyleSheet("background:#1e2124")
            bar.move(700, 180 + 185 * _i)

        color = ['#4499bb', '#ddbb33', '#33aa77', '#886699']
        self.res_bar = []
        for _i in range(4):
            bar = QLabel(self.MainWidget)
            bar.resize(0, 0)
            bar.setStyleSheet(f"background:{color[_i]}")
            bar.move(700, 180 + 185 * _i)
            self.res_bar.append(bar)

    def confirm(self):
        color = ['#4499bb', '#ddbb33', '#33aa77', '#886699']
        type_chart, result = Predict(self.textEdit.toPlainText(), 1024).prediction()
        print(type_chart, result)
        text = 'The result is: '
        _ = ['IE', 'NS', 'FT', 'JP']
        for _i in range(4):
            text += _[_i][0] if type_chart[_i] else _[_i][1]
        self.result.setText(text)
        for _i in range(4):
            value = result[_i]
            self.value[_i][0].setText(f'{str(value)[:4]}%')
            self.value[_i][1].setText(f'{str(100.1 - value)[:4]}%')
        for _i in range(4):
            if result[_i] > 50:
                self.res_bar[_i].resize(int(4 * result[_i]), 10)
                self.res_bar[_i].move(700, 180 + 185 * _i)
                self.value[_i][0].setStyleSheet(f'color:{color[_i]};'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:bold;'
                                                f'font-size:20px;')
                self.value[_i][1].setStyleSheet(f'color:#ffffff;'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:normal;'
                                                f'font-size:20px;')
                self.keyword[_i][0].setStyleSheet(f'color:{color[_i]};'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:bold;'
                                                  f'font-size:20px;')
                self.keyword[_i][1].setStyleSheet(f'color:#ffffff;'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:normal;'
                                                  f'font-size:20px;')
            else:
                self.res_bar[_i].resize(int(4 * (100 - result[_i])), 10)
                self.res_bar[_i].move(int(1100 - 4 * (100 - result[_i])), 180 + 185 * _i)
                self.value[_i][1].setStyleSheet(f'color:{color[_i]};'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:bold;'
                                                f'font-size:20px;')
                self.value[_i][0].setStyleSheet(f'color:#ffffff;'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:normal;'
                                                f'font-size:20px;')
                self.keyword[_i][1].setStyleSheet(f'color:{color[_i]};'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:bold;'
                                                  f'font-size:20px;')
                self.keyword[_i][0].setStyleSheet(f'color:#ffffff;'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:normal;'
                                                  f'font-size:20px;')

    def clear(self):
        self.textEdit.setPlainText('')
        self.result.setText('')
        for _i in range(4):
            self.res_bar[_i].setGeometry(0, 0, 0, 0)
            for _j in range(2):
                self.value[_i][_j].setText('N/A')
                self.value[_i][_j].setStyleSheet(f'color:#ffffff;'
                                                 f'font-family:JetBrains Mono;'
                                                 f'font-weight:normal;'
                                                 f'font-size:20px;')
                self.keyword[_i][_j].setStyleSheet(f'color:#ffffff;'
                                                   f'font-family:JetBrains Mono;'
                                                   f'font-weight:normal;'
                                                   f'font-size:20px;')


if __name__ == '__main__':
    model = []
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    main_ui = Test()
    main_ui.show()
    sys.exit(app.exec_())
