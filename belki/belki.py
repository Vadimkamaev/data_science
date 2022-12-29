import pandas as pd
from servise_ds import okno
import cat_encoder
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ispravit_error():
    train_s_error = pd.read_csv("C:\\kaggle\\белки\\train_s_error.csv")
    train_updates = pd.read_csv("C:\\kaggle\\белки\\train_updates_20220929.csv") #2434 строки
    train_updates_isnull = train_updates[train_updates['pH'].isnull()] # должно быть 2409 строк
    train_updates_notnull = train_updates[train_updates['pH'].notnull()] # должно быть 25 строк
    maska = train_s_error['seq_id'].isin(train_updates['seq_id'])
    train_s_error = train_s_error[~maska]
    train = pd.concat([train_s_error,train_updates_notnull], sort=False)
    train.sort_values(by='seq_id', inplace=True)
    train.to_csv("C:\\kaggle\\белки\\train0.csv", index = False)
    pass

def kodirovka():
    train = pd.read_csv("C:\\kaggle\\белки\\train0.csv")
    test = pd.read_csv("C:\\kaggle\\белки\\test.csv")
    train, test = cat_encoder.coder(train, test, 'data_source', 'tm')
    # train.set_index('seq_id', drop=True, inplace=True)
    # test.set_index('seq_id', drop=True, inplace=True)
    train = train.convert_dtypes()
    #test = test.convert_dtypes()
    train['lenstr'] = train['protein_sequence'].str.len()
    test['lenstr'] = test['protein_sequence'].str.len()
    train.to_csv("C:\\kaggle\\белки\\train1.csv", index=False)
    test.to_csv("C:\\kaggle\\белки\\test1.csv", index=False)

#kodirovka()

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

# Аминокислоты
search_amino=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# -100
# Ошибка 1 7.8241025641025645 в % 7.8241025641025645
# Ошибка 10 6.96613712374582 в % 6.96613712374582
# Ошибка 20 6.917341137123746 в % 6.917341137123746
# Ошибка 30 6.60311483273102
# Время  0:07:27.799152


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
    sim1 = similar12(s1, s2)#/len(s2)
    sim2 = similar12(s2, s1)#/len(s1)
    return (sim1+sim2)/((len(s1)+len(s2))**0.25)

# функция похожести 2-х строк
# def similar(s1,s2):
#     return abs(len(s2)-len(s1))

def pohoshi_na_1(train, s, granica):
    train['similar'] = -1
    # delt = len(s)*granica/100000
    # while True:
    #     maska = (train['lenstr'] > len(s)-delt)&(train['lenstr'] < len(s)+delt)
    #     if maska.sum() > granica:
    #         break
    #     else:
    #         delt = delt *2
    maska = (train['lenstr'] > 60) & (train['lenstr'] < 300)
    ltrain = train[maska]
    for index in ltrain.index:
        train.loc[index, 'similar'] = similar(s,train.loc[index, 'protein_sequence'])#/train.loc[index, 'lenstr']
    return(train)

def ocenka(train):   #, rez, cel_znach):
    train = train[train['similar']>0]
    train=train.sort_values(by='similar',ascending=False)
    summa = 0
    ves = 0
    for i, row in train.iloc[0:100].iterrows():
        summa += row['tm']*row['similar']
        ves += row['similar']
    sr1 = summa/ves
    #sim1 = train['similar'].iloc[0:30].mean()
    return sr1

#kodirovka()
#similar('AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLVG','AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR')
#pohoshi_na_1()

def work_model(train1, test, granica):
    #start_time = datetime.now()
    rezult = pd.DataFrame(columns=['seq_id', 'tm']) # результат оптимизации
    for index, row in test.iterrows():
        train = train1.copy()
        s = row['protein_sequence']
        #cel_znach = row['tm']
        train = pohoshi_na_1(train, s, granica)
        sr = ocenka(train)
        rezult.loc[len(rezult.index)]=[row['seq_id'], sr]
        print(index)
        if index % 20 == 0:
            rezult.to_csv("C:\\kaggle\\белки\\reztest.csv")
    rezult.to_csv("C:\\kaggle\\белки\\reztest.csv")


    # rezultat(rez, start_time, granica, len(test))
    # return rez

def pohoshi():
    pd.options.display.width = 0
    train = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
    test = pd.read_csv("C:\\kaggle\\белки\\test1.csv")
    #train1, test =  train_test_split(train0, test_size=1000)
    work_model(train, test, 20000)

pohoshi()

# rez = pd.read_csv("C:\\kaggle\\белки\\supertext.csv")
# test = pd.read_csv("C:\\kaggle\\белки\\test.csv")
# test = test.join(rez)
# test = test.drop(columns = ['protein_sequence', 'pH', 'data_source', 'Unnamed: 0'])
# test.set_index('seq_id', inplace=True)
# test.fillna(50, inplace=True)
# test.to_csv("C:\\kaggle\\белки\\rez.csv")










#amino_division()
# rez =pd.read_csv("C:\\kaggle\\белки\\rez.csv")
# okno.vewdf(rez[rez['sim1']<20])



# 100 проходов с /train.loc[index, 'lenstr']
# Ошибка 1 9.45
# Ошибка 10 6.854222222222222
# Ошибка 20 7.210052631578947
# Ошибка 30 7.499586206896551

# 100 проходов без /train.loc[index, 'lenstr']
# Ошибка 1 9.604 в % 9.604
# Ошибка 10 6.427333333333334 в % 6.427333333333334
# Ошибка 20 6.783578947368422 в % 6.783578947368422
# Ошибка 30 6.833896551724138

# Ошибка 1 9.73298969072165 в % 9.73298969072165
# Ошибка 10 6.509736540664376 в % 6.509736540664376
# Ошибка 20 6.755670103092783 в % 6.755670103092783
# Ошибка 30 6.871283789447899
# Время  0:11:00.814960

#Граница 1000
# Ошибка 1 9.19090909090909 в % 9.19090909090909
# Ошибка 10 7.106868686868686 в % 7.106868686868686
# Ошибка 20 7.238585858585857 в % 7.238585858585857
# Ошибка 30 7.007009501243359
# Время  0:04:40.389757

#Граница 500
# Ошибка 1 9.122222222222222 в % 9.122222222222222
# Ошибка 10 7.917508417508415 в % 7.917508417508415
# Ошибка 20 7.600956937799043 в % 7.600956937799043
# Ошибка 30 7.1497811012434775
# Время  0:02:21.138051

# Граница 100 плохо

# Граница 700
# Ошибка 1 9.77979797979798 в % 9.77979797979798
# Ошибка 10 7.263299663299662 в % 7.263299663299662
# Ошибка 20 7.255821371610845 в % 7.255821371610845
# Время 0:03:11.697110

# Бессмысленная функция похожести
# Ошибка 1 10.91818181818182 в % 10.91818181818182
# Ошибка 10 8.510000000000002 в % 8.510000000000002
# Ошибка 20 8.017727272727274 в % 8.017727272727274
# Ошибка 30 7.887600343127837

#Близость длин строк
# Ошибка 1 8.223529841498095 в % 8.223529841498095
# Ошибка 10 8.964285714285714 в % 8.964285714285714
# Ошибка 20 8.276071428571427 в % 8.276071428571427
# Ошибка 30 8.252251975449068

# Граница 500, 300 проходов
# Ошибка 1 8.472597547380156 в % 8.472597547380156
# Ошибка 10 6.759866220735785 в % 6.759866220735785
# Ошибка 20 6.80448160535117 в % 6.80448160535117
# Ошибка 30 6.789895085175138
# Время 0:07:22.025349

# Граница 500, 300 проходов
# Ошибка 1 8.556209587513935 в % 8.556209587513935
# Ошибка 10 6.828494983277592 в % 6.828494983277592
# Ошибка 20 6.797826086956523 в % 6.797826086956523
# Ошибка 30 6.751722927418364
# Время  0:07:23.494807