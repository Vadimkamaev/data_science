import pandas as pd
from servise_ds import okno
import cat_encoder
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import re   #import regular expression module for individual amino acid extrcation

def amino_division():
    train = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
    test = pd.read_csv("C:\\kaggle\\белки\\test1.csv")
    search_amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for amino in search_amino:
        train[amino] = train['protein_sequence'].str.count(amino, re.I)
        test[amino] = test['protein_sequence'].str.count(amino, re.I)
    train.to_csv("C:\\kaggle\\белки\\train2.csv", index=False)
    test.to_csv("C:\\kaggle\\белки\\test2.csv", index=False)

# функция похожести определяет вхождения всех подстрок строки 1 в строку 2
def similar12(s1,s2):
    sum = 0 # сумма длин похожих строк
    i_s1=0
    l_s1=len(s1)
    while i_s1 < l_s1: # цикл по строке s1
        i_s2 = 0
        l_s2 = len(s2)
        l=1 # длина совпадающей подстроки
        max_l = 0 # наибольшая из совпадающих подстрок
        while i_s2 + l <= l_s2:  # цикл по количеству совпадающих символов
            srez= s1[i_s1:i_s1+l]
            np = s2.find(srez, i_s2)
            i_s2 = np
            if np > -0.5:
                max_l = l
                l += 1
            else:
                break
        sum += max_l
        i_s1 += 1
    sim = sum / l_s1
    return sim

#функция похожести 2-х строк
def similar(s1,s2):
    sim1 = similar12(s1, s2)
    sim2 = similar12(s2, s1)
    return (sim1+sim2)/2

# функция похожести 2-х строк
# def similar(s1,s2):
#     return abs(len(s2)-len(s1))

def pohoshi_na_1(train, s, granica):
    train['similar'] = -1
    delt = len(s)*granica/100000
    while True:
        maska = (train['lenstr'] > len(s)-delt)&(train['lenstr'] < len(s)+delt)
        if maska.sum() > granica:
            break
        else:
            delt = delt *2
    ltrain = train[maska]
    for index in ltrain.index:
        train.loc[index, 'similar'] = similar(s,train.loc[index, 'protein_sequence'])#/train.loc[index, 'lenstr']
    return(train)

def ocenka(train, cel_znach):
    train = train[train['similar']>0]
    train=train.sort_values(by='similar',ascending=False)
    summa = 0
    ves = 0
    for i, row in train.iloc[0:30].iterrows():
        summa += row['tm']*row['similar']**4
        ves += row['similar']**4
    sr1 = summa/ves
    er1 = abs(cel_znach - sr1)
    er1pr = er1 / sr1 * 100
    sim1 = train['similar'].iloc[0:30].mean()

    tect = train['tm'].iloc[0:10]
    sr10 = train['tm'].iloc[0:10].mean()
    er10 = abs(cel_znach - sr10)
    er10pr = er10 / sr10 * 100
    sim10 = train['similar'].iloc[0:10].mean()

    sr20 = train['tm'].iloc[0:20].mean()
    er20 = abs(cel_znach - sr20)
    er20pr = er20 / sr20 * 100
    sim20 = train['similar'].iloc[0:20].mean()

    maska = train['similar'] > 4.5
    if maska.sum() < 5:
        maska = train['similar'] > 4
    if maska.sum() < 5:
        maska = train['similar'] > 3.5
    if maska.sum() >= 5:
        sr30 = train[maska]['tm'].mean()
        er30 = abs(cel_znach - sr30)
    if maska.sum() < 5:
        sr30 = sr10
        er30 = er10

    print('Среднее от 1', sr1, 'Ошибка', er1)
    print('Среднее от 10', sr10, 'Ошибка', er10)
    print('Среднее от 20', sr20, 'Ошибка', er20)
    print('Среднее similar > 4.5', sr30, 'Ошибка', er30)
    rez = [er1, er1pr, sim1, er10, er10pr, sim10, er20, er20pr, sim20, er30]
    return rez

#kodirovka()
#similar('AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLVG','AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR')
#pohoshi_na_1()

def rezultat(rez, start_time, granica, kol):
    print('Кол. строк в тесте', kol, 'Граница =', granica )
    print('Ошибка 1', rez['er1'].mean(), 'в %', rez['er1%'].mean())
    print('Ошибка 10', rez['er10'].mean(), 'в %', rez['er10%'].mean())
    print('Ошибка 20', rez['er20'].mean(), 'в %', rez['er20%'].mean())
    print('Ошибка 30', rez['er30'].mean())
    print('Время ', datetime.now() - start_time)
    with open("C:\\kaggle\\белки\\otсhets.txt", 'a') as file:
        file.write(f'Кол. строк в тесте. {kol} Граница = {granica} \n')
        file.write(f"Ошибка 1 {rez['er1'].mean()} в % {rez['er1%'].mean()} \n")
        file.write(f"Ошибка 10 {rez['er10'].mean()} 'в %' {rez['er10%'].mean()} \n")
        file.write(f"Ошибка 20 {rez['er20'].mean()}, 'в %', {rez['er20%'].mean()} \n")
        file.write(f"Ошибка 30 {rez['er30'].mean()} \n")
        file.write(f"Время  {datetime.now() - start_time}\n\n")

    rez.to_csv("C:\\kaggle\\белки\\rez.csv", index=False)

def work_model(q, gtrain, test, granica):
    grez = pd.DataFrame(columns=['er1', 'er1%', 'sim1', 'er10', 'er10%', 'sim10', 'er20', 'er20%', 'sim20', 'er30'])
    for index, row in test.iterrows():
        train = gtrain.copy()
        s = row['protein_sequence']
        cel_znach = row['tm']
        train = pohoshi_na_1(train, s, granica)
        rez = ocenka(train, cel_znach)
        grez.loc[len(grez.index)] = rez
    q.put(grez) # передаем в очередь

from multiprocessing import  Process, Queue

def parallel_model(train, test, granica, start_time):
    test1, test2 = train_test_split(test, test_size=0.5)
    if __name__ == '__main__':
        q1 = Queue() # создание очереди для 1 процесса, очередь нужна для возвратас процессом результата работы
        q2 = Queue()  # создание очереди для 2 процесса, очередь нужна для возвратас процессом результата работы
        p1 = Process(target=work_model, args=(q1, train, test1, granica)) # создание 1-го процесса
        p2 = Process(target=work_model, args=(q2, train, test2, granica)) # создание 2-го процесса
        p1.start() # запуск 1-го процесса
        p2.start()  # запуск 2-го процесса
        rez1 = q1.get() # получение результата работы 1-го процесса из очереди
        rez2 = q2.get()  # получение результата работы 2-го процесса из очереди
        p1.join()
        p2.join()
        rez = pd.concat([rez1,rez2])
        rezultat(rez, start_time, granica, len(test))


def pohoshi():
    pd.options.display.width = 0
    train0 = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
    train1, test =  train_test_split(train0, test_size=500)
    granica = 500
    start_time = datetime.now()
    parallel_model(train1, test, granica, start_time)

    #okno.vewdf(rez)

pohoshi()

