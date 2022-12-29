import pandas as pd
import time

# кодировщик, df -
def coder(train, test, name_col, name_y,  vers = 1):
    if vers == 0:
        df_kod = train[[name_col, name_y]].groupby([name_col]).mean() # средние значения y сгруппированные по col
        df_kod.rename(columns={name_y: 'n_'+name_col}, inplace=True) # переименовываем колонку
        train = train.join(df_kod, on =name_col) # добавляем к датафрейму train новую колонку
        train.drop(name_col, axis=1, inplace=True) # удаляем колонку со старыми значениями из train
        test = test.join(df_kod, on=name_col)  # добавляем к датафрейму test новую колонку
        test.drop(name_col, axis=1, inplace=True)  # удаляем колонку со старыми значениями из test
    elif vers == 1:
        df_kod = train[[name_col, name_y]].groupby([name_col]).mean() # средние значения y сгруппированные по col
        df_kod.sort_values(by=name_y, inplace=True) # сортируем колонку
        l = list(range(len(df_kod))) # создаем список целых чисел начиная с 0
        df_kod['n_'+name_col] = l # добавляем колонку с числами упорядоченными по возрастанию среднего знач. y
        df_kod.drop(name_y, axis=1, inplace=True) # удаляем колонку со средними знач. y
        train = train.join(df_kod, on=name_col) # добавляем в датафрейм train новую колонку
        train.drop(name_col, axis=1, inplace=True) # удаляем старую колонку из train
        test = test.join(df_kod, on=name_col)  # добавляем к датафрейму test новую колонку
        test.drop(name_col, axis=1, inplace=True)  # удаляем колонку со старыми значениями из test
    elif vers == 2:
        pd.get_dummies(train, columns=[name_col], drop_first=True) # создаем несколько колонок в train
        pd.get_dummies(test, columns=[name_col], drop_first=True)  # создаем несколько колонок в test
    return train, test
