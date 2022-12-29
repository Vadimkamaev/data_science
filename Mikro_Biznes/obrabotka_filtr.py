import pandas as pd
from servise_ds import okno

# фильтр по фиксированным границам от min до max
def fix_filtr(train):
    min = 0.1
    max = 10
    maska = (train['microbusiness_density']> max)|(train['microbusiness_density']< min)
    musor = train[maska]['cfips'].unique()
    maska = train['cfips'].isin(musor)
    train = train[~maska]
    return train

# Еcть ли разрыв в loc_train больше чем k
def outliers(loc_train, k):  # относительная модель выбросов
    l = len(loc_train)
    for i in range(1, l-1):
        i_loc_train = loc_train['microbusiness_density'].iloc[i]
        i_1 = loc_train['microbusiness_density'].iloc[i-1]
        if abs(i_loc_train - i_1)  > k:
            return True
    return False

# Фильтр по размеру изменения за 1 месяц
def outliers_model_otn(train):
    k = 0.15
    unic_cfips = train['cfips'].unique()
    for cfips in unic_cfips:
         if outliers(train[train['cfips'] == cfips], k):
            train = train[train['cfips'] != cfips]
    return train

# Количество следующих элементов равных предыдущим
def ekvivalent(loc_train):
    l = len(loc_train)
    kol = 0
    for i in range(1, l-1):
        i_loc_train = loc_train['microbusiness_density'].iloc[i]
        i_1 = loc_train['microbusiness_density'].iloc[i-1]
        if abs(i_loc_train - i_1)  < 0.000000001:
            kol += 1
    return kol

# Подсчет количества точных повторов
def count_ekvivalent(train):
    unic_cfips = train['cfips'].unique()
    df_unic_cfips = pd.DataFrame({'repit':0, 'count':0}, index=unic_cfips)
    for cfips in unic_cfips:
        df_unic_cfips.loc[cfips,'repit'] = ekvivalent(train[train['cfips'] == cfips])
    g= df_unic_cfips[['repit', 'count']].groupby(['repit']).count()
    #g=df[['Koltik', 'Survived']].groupby(['Koltik']).count()
    print(g)

def predobrabotka(train):
    # находим средние 'microbusiness_density' по США для каждого месяца
    meanusa = train[['first_day_of_month', 'microbusiness_density']].groupby(['first_day_of_month']).mean()
    meanusa.columns = ['mean_usa']
    train = train.join(meanusa, on='first_day_of_month')
    # делаем 'microbusiness_density' разностью между значением у каунтри и средней по США
    train['microbusiness_density'] = train['microbusiness_density'] - train['mean_usa']
    return train

# загрузка данных
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

def super_filtr(train):
    train = fix_filtr(train)
    train = predobrabotka(train)
    train = outliers_model_otn(train)
    return train

if __name__ == "__main__":
    train, test = start()
    #count_ekvivalent(train)

