import os
import random

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    world = "Happy World!"
    return render(request, 'Home.html', {"w": world})


def mbti(request):
    if request.method == 'POST':
        try:
            if request.POST.get('text'):
                with open("web_core/text.txt", "w") as f:
                    f.write(request.POST.get('text'))
            text = request.POST.get('text')
        except:
            pass
        while True:
            try:
                data = []
                with open('web_core/res.txt') as f:
                    data = f.readlines()
                    result = f'{"I" if int(data[0]) else "E"}' \
                             f'{"N" if int(data[1]) else "S"}' \
                             f'{"F" if int(data[2]) else "T"}' \
                             f'{"J" if int(data[3]) else "P"}'
                    IE = f'IE二分类信度: {"%.2f"%float(data[4]) if float(data[4]) > 50 else "%.2f"%float(100 - float(data[4]))} %'
                    NS = f'NS二分类信度: {"%.2f"%float(data[5]) if float(data[5]) > 50 else "%.2f"%float(100 - float(data[5]))} %'
                    FT = f'FT二分类信度: {"%.2f"%float(data[6]) if float(data[6]) > 50 else "%.2f"%float(100 - float(data[6]))} %'
                    JP = f'JP二分类信度: {"%.2f"%float(data[7]) if float(data[7]) > 50 else "%.2f"%float(100 - float(data[7]))} %'
                os.remove('web_core/res.txt')
                break
            except Exception as sb:
                with open("web_core/log.txt", "w") as f:
                    f.write(str(sb))
        return render(request, 'mbti.html', locals())
    return render(request, 'mbti.html', locals())
