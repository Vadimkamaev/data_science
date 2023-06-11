import numpy as np
import pandas as pd

def dobavka_str():
    global i1, i2, i3, df, feature_q
    feature_q.loc[i1,'tip'] = 1
    row = feature_q.loc[i1]
    li_st = list(row)
    df.loc[len(df.index )] = li_st

    feature_q.loc[i1 + 1,'tip'] = 1
    row = feature_q.loc[i1 + 1]
    li_st = list(row)
    df.loc[len(df.index)] = li_st

    feature_q.loc[i1 + 2,'tip'] = 1
    row = feature_q.loc[i1 + 2]
    li_st = list(row)
    df.loc[len(df.index)] = li_st
    i1 += 3

    feature_q.loc[i2,'tip'] = 2
    row = feature_q.loc[i2]
    li_st = list(row)
    df.loc[len(df.index)] = li_st

    feature_q.loc[i2 + 1,'tip'] = 2
    row = feature_q.loc[i2 + 1]
    li_st = list(row)
    df.loc[len(df.index)] = li_st
    i2 += 2

    feature_q.loc[i3, 'tip'] = 3
    row = feature_q.loc[i3]
    li_st = list(row)
    df.loc[len(df.index)] = li_st
    i3 += 1

def one_quest():
    global i1, i2, i3, df, feature_q, quest, kol_strok
    feature_q = feature_sort[feature_sort['quest'] == quest].copy()
    feature_q.sort_values('kach', inplace=True, ascending=False)
    feature_q.reset_index(drop=True, inplace=True)
    feature_q = feature_q.iloc[0:kol_strok+1]
    print('feature_q.shape =', feature_q.shape)
    i1 = 0
    i2 = 0
    i3 = 0
    while i1 + 2 <= kol_strok:
        dobavka_str()

feature_sort = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature_sort.csv")
feature_sort.drop(['nabor', 'val3', 'col3', 'kach1', 'kach2', 'kach3', 'kach4', 'kach5', 'kach6', 'kach7', 'kach8', 'kach9', 'kach10', 'kach11', 'kach12', 'kach13' ], axis = 1, inplace =True)

# df = pd.DataFrame(columns=feature_sort.columns)
#
# kol_strok = 150
# for quest in range(1,4):
#     if quest == 3:
#         pass
#     print(quest)
#     one_quest()
#
# kol_strok = 252
# for quest in range(4,19):
#     print(quest)
#     one_quest()
#
# df.to_csv("C:\\kaggle\\ОбучИгра\\new_feature_sort.csv", index=False)

new_feature_sort = pd.read_csv("C:\\kaggle\\ОбучИгра\\new_feature_sort.csv")

for quest in range(1,19):
    feature_q = feature_sort[feature_sort['quest'] == quest]
    new_feature_q = new_feature_sort[new_feature_sort['quest'] == quest]
    mi_n = feature_q['kach'].min()
    su_m = (feature_q['kach'] > mi_n + 0.00001).sum()
    pass



