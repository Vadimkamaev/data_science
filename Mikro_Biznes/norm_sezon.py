import numpy as np, pandas as pd
import obrabotka_filtr

def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

def delta_model(loc_train):
    l = len(loc_train) # размер всего временного ряда 39
    delt = -0.05
    ideal_tochnost = 100
    ideal_preds = loc_train['microbusiness_density'].iloc[l - 1]
    ideal_delt = 0
    while delt < 0.05:
        sum = 0
        for i in range(l-1):
            pred_skaz = loc_train['microbusiness_density'].iloc[i] + delt
            sum = sum+abs(pred_skaz-loc_train['microbusiness_density'].iloc[i+1])
        tochnost = sum * 100 / (l - 1)
        preds = loc_train['microbusiness_density'].iloc[l - 1]
        preds = preds + delt
        if tochnost < ideal_tochnost:
            ideal_tochnost = tochnost
            ideal_preds = preds
            ideal_delt = delt
        delt += 0.0001
    #print(f'Идеальная дельта ={ideal_delt}')
    return ideal_preds, ideal_tochnost

def sezon_model_2year(loc_train):
    ideal_tochnost = 1000000
    ideal_preds = loc_train['microbusiness_density'].iloc[-1]
    l = len(loc_train)  # размер всего временного ряда 39
    i=25 # предсказываем i-й месяц
    sum = 0
    while i < l:
        te = loc_train.iloc[i - 12]
        d1 = loc_train['microbusiness_density'].iloc[i-12]-loc_train['microbusiness_density'].iloc[i-13]
        be = loc_train.iloc[i - 24]
        d2 = loc_train['microbusiness_density'].iloc[i - 24] - loc_train['microbusiness_density'].iloc[i - 25]
        delt = (d1+d2)/2
        pred_skaz = loc_train['microbusiness_density'].iloc[i-1] + delt
        sum = sum + abs(pred_skaz - loc_train['microbusiness_density'].iloc[i])
        i += 1
    tochnost = sum * 100 / (l - 25)
    if tochnost < ideal_tochnost:
        ideal_tochnost = tochnost
        d1 = loc_train['microbusiness_density'].iloc[l-12]-loc_train['microbusiness_density'].iloc[i-13]
        be = loc_train.iloc[i - 24]
        d2 = loc_train['microbusiness_density'].iloc[i - 24] - loc_train['microbusiness_density'].iloc[i - 25]
        delt = (d1+d2)/2
        ideal_preds = loc_train['microbusiness_density'].iloc[-1] + delt
    return ideal_preds, ideal_tochnost

def sezon_model_3year(loc_train):
# результат работы модели
# model
# = 2346
# = != 7
#sz 782
# Положительная ошибка это плохо 18566.104501333335
    ideal_preds = loc_train['microbusiness_density'].iloc[-1]
    l = len(loc_train)  # размер всего временного ряда 39
    i=37 # предсказываем i-й месяц
    sum = 0
    while i < l:
        te = loc_train.iloc[i - 12]
        d1 = loc_train['microbusiness_density'].iloc[i-12]-loc_train['microbusiness_density'].iloc[i-13]
        be = loc_train.iloc[i - 24]
        d2 = loc_train['microbusiness_density'].iloc[i - 24] - loc_train['microbusiness_density'].iloc[i - 25]
        d3 = loc_train['microbusiness_density'].iloc[i - 36] - loc_train['microbusiness_density'].iloc[i - 37]
        delt = (d1+d2+d3)/3
        pred_skaz = loc_train['microbusiness_density'].iloc[i-1] + delt
        sum = sum + abs(pred_skaz - loc_train['microbusiness_density'].iloc[i])
        i += 1
    tochnost = sum * 100 / (l - 37)
    d1 = loc_train['microbusiness_density'].iloc[l-12]-loc_train['microbusiness_density'].iloc[l-13]
    d2 = loc_train['microbusiness_density'].iloc[l - 24] - loc_train['microbusiness_density'].iloc[l - 25]
    d3 = loc_train['microbusiness_density'].iloc[l - 36] - loc_train['microbusiness_density'].iloc[l - 37]
    delt = (d1 + d2 + d3) / 3
    preds = loc_train['microbusiness_density'].iloc[-1] + delt
    return preds, tochnost

# модель предсказывает равенство следующего предыдущему
def equality_model(loc_train):
    sum=0
    l = len(loc_train)  # размер всего временного ряда 39
    for i in range(l-1):
        sum = sum+abs(loc_train['microbusiness_density'].iloc[1]-loc_train['microbusiness_density'].iloc[i+1])
    tochnost = sum *100/(l-1)
    preds = loc_train['microbusiness_density'].iloc[l-1]
    return preds, tochnost

# модель предсказывает равенство следующего предыдущему
def equality_model37(loc_train):
    l = len(loc_train)  # размер всего временного ряда 39
    i=37 # предсказываем i-й месяц
    sum = 0
    while i < l:
        sum = sum+abs(loc_train['microbusiness_density'].iloc[i-1]-loc_train['microbusiness_density'].iloc[i])
        i += 1
    tochnost = sum *100/(l-37)
    preds = loc_train['microbusiness_density'].iloc[l-1]
    return preds, tochnost

def oll_sezon_equality_model(train):
    rezult = pd.DataFrame(columns=['cfips', 'microbusiness_density', 'error', 'l_regr', 'model', 'err'])
    unic_cfips = train['cfips'].unique() # массив уникальных cfips
    for cfips in unic_cfips:
        loc_train = train[train['cfips'] == cfips]
        ideal_preds, ideal_error = equality_model37(loc_train) # модель предсказывает равенство следующего предыдущему
        ideal_l_x_train = 1
        model = '='
        # preds, tochnost = delta_model(loc_train)
        preds, error = sezon_model_3year(loc_train)
        err = error - ideal_error
        if err < 0:
            ideal_l_x_train = 10
            model = 'sz'
        #print(f'Точность ={ideal_error}, Длина интервала регрессии={ideal_l_x_train}')
        if abs(err) < 0.000000001:
            model = '=!='
        rezult.loc[len(rezult.index)] = [cfips, ideal_preds, ideal_error, ideal_l_x_train, model, err]
    print(rezult.groupby(['model']).count())
    rezult.sort_values(by='err', inplace=True)
    print('Положительная ошибка это плохо', rezult['err'].sum())
    return rezult

train, test = start()
# train = obrabotka_filtr.fix_filtr(train)
# train = obrabotka_filtr.predobrabotka(train)
train = obrabotka_filtr.super_filtr(train)
rezult=oll_sezon_equality_model(train)
# rezult = rezult[["cfips", "microbusiness_density"]]
# rezult.set_index("cfips", inplace=True)
# test = test.join(rezult, on="cfips")
# test[["row_id", "microbusiness_density"]].to_csv("C:\\kaggle\\МикроБизнес\\rezult1331.csv", index=False)