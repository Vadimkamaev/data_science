import pandas as pd
from servise_ds import okno
import cat_encoder
from datetime import datetime
import matplotlib.pyplot as plt
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

#функция похожести 2-х строк
def similar(s1,s2):
    s = s1-s2
    s = s.abs()
    sum = s.sum()
    return 200-sum

# функция похожести 2-х строк
# def similar(s1,s2):
#     return abs(len(s2)-len(s1))

def pohoshi_na_1(train, s):
    train['similar'] = -1
    granica = 500
    delt = len(s)*granica/100000
    while True:
        maska = (train['lenstr'] > len(s)-delt)&(train['lenstr'] < len(s)+delt)
        if maska.sum() > granica:
            break
        else:
            delt = delt *2
    ltrain = train[maska]
    for index in ltrain.index:
        s2 = train.loc[index][5:]
        train.loc[index, 'similar'] = similar(s,s2)
    #train = train[train['similar']>0]
    train=train.sort_values(by='similar',ascending=False)
    return(train)

def ocenka(train, rez, cel_znach):
    max = train['similar'].max()
    maska = train['similar']==max
    sr1 = train[maska]['tm'].mean()
    er1 = abs(cel_znach - sr1)
    er1pr = er1 / sr1 * 100
    sim1 = max
    print('Среднее от 1', sr1, 'Ошибка', er1)
    sr10 = train['tm'].iloc[0:10].mean()
    er10 = abs(cel_znach - sr10)
    er10pr = er10 / sr10 * 100
    sim10 = train['similar'].iloc[0:10].mean()
    print('Среднее от 10', sr10, 'Ошибка', er10)

    sr20 = train['tm'].iloc[0:20].mean()
    er20 = abs(cel_znach - sr20)
    er20pr = er20 / sr20 * 100
    sim20 = train['similar'].iloc[0:20].mean()
    print('Среднее от 20', sr20, 'Ошибка', er20)

    maska = train['similar'] > 4.5
    if maska.sum() < 5:
        maska = train['similar'] > 4
    if maska.sum() < 5:
        maska = train['similar'] > 3.5
    if maska.sum() >= 5:
        sr30 = train[maska]['tm'].mean()
        er30 = abs(cel_znach - sr30)
    if maska.sum() < 5:
        sr30 = sr1
        er30 = er1
    if er30 < 0.1:
        print('Целевое значение', cel_znach, 'Кол. элем. по которым считали', (train['similar'] > 4.5).sum() )
    print('Среднее similar > 4.5', sr30, 'Ошибка', er30)
    rez.loc[len(rez)]=[er1, er1pr, sim1, er10, er10pr, sim10, er20, er20pr, sim20, er30]
    # train[1:].plot.scatter(x='similar', y='tm')
    # plt.show()
    # okno.vewdf(train[1:])
    return rez

#kodirovka()
#similar('AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLVG','AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR')
#pohoshi_na_1()

def pohoshi():
    search_amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pd.options.display.width = 0
    rez =pd.DataFrame(columns=['er1', 'er1%', 'sim1', 'er10', 'er10%', 'sim10', 'er20', 'er20%', 'sim20', 'er30'])
    i=1
    start_time = datetime.now()
    train0 = pd.read_csv("C:\\kaggle\\белки\\train2.csv")
    while i < 50:
        j=i*20
        train = train0.copy()
        s = train.loc[j]
        s=s[5:]
        cel_znach = train.loc[j, 'tm'] # истинная величина искомой величины 'tm'
        train.drop(index=j, inplace=True)
        train = pohoshi_na_1(train, s)
        rez = ocenka(train, rez, cel_znach)
        i += 1
    print('Ошибка 1', rez['er1'].mean(), 'в %', rez['er1'].mean())
    print('Ошибка 10', rez['er10'].mean(), 'в %', rez['er10'].mean())
    print('Ошибка 20', rez['er20'].mean(), 'в %', rez['er20'].mean())
    print('Ошибка 30', rez['er30'].mean())
    print('Время ', datetime.now() - start_time)
    rez.to_csv("C:\\kaggle\\белки\\rez.csv", index=False)
    okno.vewdf(rez)

pohoshi()