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
                    # f.write('Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity. ')
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
                pass
        return render(request, 'mbti.html', locals())
    return render(request, 'mbti.html', locals())
