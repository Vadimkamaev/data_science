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
    i_s1=0 # номер проверяемого символа в s1
    l_s1=len(s1)
    while i_s1 < l_s1: # цикл по строке s1
        i_s2 = 0
        l_s2 = len(s2)
        l=1 # длина совпадающей подстроки
        max_l = 0 # наибольшая из совпадающих подстрок
        while i_s2 + l <= l_s2:  # цикл по количеству совпадающих символов
            if i_s1+l > l_s1:
                break
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
    return sim1+sim2

def pohoshi_na_1(train, s):
    train['similar'] = -1
    for index in train.index:
        train.loc[index, 'similar'] = similar(s,train.loc[index, 'protein_sequence'])
    return(train)

def ocenka(train):
    train = train[train['similar']>0]
    train=train.sort_values(by='similar',ascending=False)
    summa = 0
    ves = 0
    for i, row in train.iloc[0:30].iterrows():
        summa += row['tm']*row['similar']**4
        ves += row['similar']**4
    sr1 = summa/ves
    summa = 0
    ves = 0
    for i, row in train.iloc[0:100].iterrows():
        summa += row['tm']*row['similar']
        ves += row['similar']
    sr2 = summa/ves
    return sr1, sr2

def model(test):
    rezult1 = pd.DataFrame(columns=['seq_id', 'tm'])  # результат оптимизации
    rezult2 = pd.DataFrame(columns=['seq_id', 'tm'])  # результат оптимизации
    for ind, row in test.iterrows():
        train = test.copy()
        train.drop(index=ind, inplace=True)
        s = row['protein_sequence']
        train = pohoshi_na_1(train, s)
        sr1, sr2 = ocenka(train)
        rezult1.loc[len(rezult1.index)] = [row['seq_id'], sr1]
        print(ind)
        if ind+1 % 100 == 0:
            rezult1.to_csv("C:\\kaggle\\белки\\reztest1.csv")
        rezult2.loc[len(rezult2.index)] = [row['seq_id'], sr2]
        if ind+1 % 100 == 0:
            rezult2.to_csv("C:\\kaggle\\белки\\reztest2.csv")
    rezult1.to_csv("C:\\kaggle\\белки\\reztest1.csv")
    rezult2.to_csv("C:\\kaggle\\белки\\reztest2.csv")

def finish(rez1, rez2, test, granica, start_time):
    rez = pd.concat([rez1, rez2])

from multiprocessing import Process, Queue

# Функция в которой один из процессов, каждый из процессов. Содержательная часть в функции model
def one_process(q, train, test, granica):
    grez = model(train, test, granica)
    q.put(grez) # передаем в очередь

# Cоздание параллельных процессов. Датафрейм test делится количество частей равное количеству параллельных
# процессов и каждая его часть обрабатывается отдельным процессом выполняемым в функции one_process
def parallel_process(test): # train, test - датафреймы, другие параметры передаются в процессы
    test1, test2 = train_test_split(test, test_size=0.5)
    if __name__ == '__main__':
        q1 = Queue()  # создание очереди для 1 процесса, очередь нужна для возвратас процессом результата работы
        q2 = Queue()  # создание очереди для 2 процесса, очередь нужна для возвратас процессом результата работы
        p1 = Process(target=one_process, args=(q1, test, test1))  # создание 1-го процесса
        p2 = Process(target=one_process, args=(q2, test, test2))  # создание 2-го процесса
        p1.start()  # запуск 1-го процесса
        p2.start()  # запуск 2-го процесса
        rez1 = q1.get()  # получение результата работы 1-го процесса из очереди
        rez2 = q2.get()  # получение результата работы 2-го процесса из очереди
        p1.join()
        p2.join()
        return (rez1, rez2)
        #finish(rez1, rez2, train, test, granica, start_time)

def start_parallel(test):
    pd.options.display.width = 0
    granica = 500
    start_time = datetime.now()
    rez = parallel_process(test)
    if rez != None:
        rez1, rez2 = rez
        finish(rez1, rez2, test, granica, start_time)

def start_one_process(test):
    pd.options.display.width = 0
    model(test)

test = pd.read_csv("C:\\kaggle\\белки\\test1.csv")
test.drop(columns=['lenstr', 'n_data_source', 'pH'], inplace=True)
sub = pd.read_csv("C:\\kaggle\\белки\\submiss1.csv")
test['tm']=sub['tm']

# with open("C:\\kaggle\\белки\\otсhets.txt", 'a') as file:
#     file.write('Не параллельный процесс')
start_one_process(test)

# with open("C:\\kaggle\\белки\\otсhets.txt", 'a') as file:
#     file.write('Параллельный процесс')
# start_parallel(test)