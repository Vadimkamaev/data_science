import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from servise_ds import okno
import obrabotka_filtr

# ВАЖНО УЧЕСТЬ ИНФУ О ПРАЗДНИКАХ
#import holidays

def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

def predobrabotka(train):
    # находим средние 'microbusiness_density' по США для каждого месяца
    meanusa = train[['first_day_of_month', 'microbusiness_density']].groupby(['first_day_of_month']).mean()
    meanusa.columns = ['mean_usa']
    train = train.join(meanusa, on='first_day_of_month')
    train['microbusiness_density'] = train['microbusiness_density'] - train['mean_usa']
    return train

    # # определяем изменение 'mean_difference' за месяц
    # meanusa['mean_difference']=0
    # meanusa.reset_index(inplace=True)
    # for i in range(1,len(meanusa)): # считаем изменение за месяц 'microbusiness_density' в сша
    #     difference = meanusa.loc[i,'microbusiness_density']-meanusa.loc[i-1,'microbusiness_density']
    #     meanusa.loc[i,'mean_difference']= difference
    # # / нашли средние 'microbusiness_density' по США для каждого месяца
    #
    # # находим изменения за каждый месяц и разницу этих изменений и средних изменений
    # df = pd.DataFrame(columns=['row_id', 'difference', 'dif_usa'])
    # unic_cfips = train['cfips'].unique()
    # for cfips in unic_cfips:
    #     loc_train = train[train['cfips'] == cfips]
    #     for i in range(1,len(loc_train)): # считаем изменение за месяц 'microbusiness_density' для каждой строки train
    #         difference = loc_train.iloc[i]['microbusiness_density']-loc_train.iloc[i-1]['microbusiness_density']
    #         fdm = loc_train.iloc[i]['first_day_of_month'] # первый день i-го месяца
    #         maska = meanusa['first_day_of_month'] == fdm
    #         meanusa_fdm = meanusa[maska]['mean_difference'].iloc[0]
    #         dif_usa = difference - meanusa_fdm
    #         df.loc[len(df.index)] = (loc_train.iloc[i]['row_id'], difference, dif_usa)
    #     print(cfips)
    #     #print(df.head(39))
    #     okno.vewdf(df)
    # return meanusa

# RandomForestRegressor случайный лес
# предсказываем n-й элемент последовательности по l_x_train предыдущим
def forest_regress(loc_train, l_x_train):
    l = len(loc_train)
    razmer = l - l_x_train # количество полученных предсказаний
    i = 0
    x=np.zeros((razmer, l_x_train))
    y=np.zeros(razmer)
    while i < razmer:
        x[i] = loc_train['microbusiness_density'].iloc[i:i+l_x_train]
        y[i] = loc_train['microbusiness_density'].iloc[i+l_x_train]
        i += 1
    # model = RandomForestRegressor(n_estimators=l_x_train*5)
    model = GradientBoostingRegressor(n_estimators=l_x_train*5)
    model.fit(x, y)
    test=np.zeros((1,l_x_train))
    test[0] = loc_train['microbusiness_density'].iloc[-l_x_train:]
    preds = model.predict(test)
    cross_val_scores = cross_val_score(model, x, y, cv=5, scoring=make_scorer(mean_absolute_error))
    error = np.mean(cross_val_scores)*100
    return preds[0], error

# линейная регрессия
# предсказываем n-й элемент последовательности по l_x_train предыдущим
def line_regress(loc_train, l_x_train):
    l = len(loc_train)
    razmer = l - l_x_train # количество полученных предсказаний
    i = 0
    x=np.zeros((razmer, l_x_train))
    y=np.zeros(razmer)
    while i < razmer:
        x[i] = loc_train['microbusiness_density'].iloc[i:i+l_x_train]
        y[i] = loc_train['microbusiness_density'].iloc[i+l_x_train]
        i += 1
    model = LinearRegression()
    model.fit(x, y)
    test=np.zeros((1,l_x_train))
    test[0] = loc_train['microbusiness_density'].iloc[-l_x_train:]
    preds = model.predict(test)
    cross_val_scores = cross_val_score(model, x, y, cv=5, scoring=make_scorer(mean_absolute_error))
    error = np.mean(cross_val_scores)*100
    return preds[0], error

# модель предсказывает равенство следующего предыдущему
def equality_model(loc_train):
    sum=0
    l = len(loc_train)
    for i in range(2,l-1):
        sum = sum+abs(loc_train['microbusiness_density'].iloc[i]-loc_train['microbusiness_density'].iloc[i+1])
    error = sum *100/(l-3)
    preds = loc_train['microbusiness_density'].iloc[l-1]
    return preds, error

# модель анализирующая выбросы. Анализировал как абсолютные, так и относительные выбросы
# Вывод. После выброса следующее значение похоже на 1-е значение выброса.
# Не плохим алгоритмом предсказания значения после выброса будет равенство предыдущему
def outliers_model_otn(loc_train, k):  # относительная модель выбросов
    # В зависимости от значения k:
    # 0.05 - в 2115 строках есть разрыв на 10%. Без разрывов в 5% - 1020
    # 0.1 - в 1162 строках есть разрыв на 10%
    # 0.2 - в 450 строках есть разрыв на 20%
    # 0.3 - в 219 строках есть разрыв на 30%
    sum=0
    error = 0
    l = len(loc_train)
    for i in range(2, l-1):
        i_loc_train = loc_train['microbusiness_density'].iloc[i]
        i_1 = loc_train['microbusiness_density'].iloc[i-1]
        i_2 = loc_train['microbusiness_density'].iloc[i-1]
        i_sr = (i_1 + i_2)/2
        if ((abs(i_loc_train - i_1)) / (abs(i_loc_train)+0.000001) > k) & \
                ((abs(i_loc_train - i_2)) / (abs(i_loc_train)+0.000001) > k):
            preds = i_sr
        else:
            preds = i_loc_train
        sum = sum+abs(preds-loc_train['microbusiness_density'].iloc[i+1])
    error = sum *100/(l-3)
    if loc_train['microbusiness_density'].iloc[l-1] < k:
        preds = loc_train['microbusiness_density'].iloc[l-1]
    else:
        # preds = loc_train['microbusiness_density'].iloc[l-2]
        preds = loc_train['microbusiness_density'].iloc[l - 1]
    return preds, error


def oll_line_regress(train) :
    unic_cfips = train['cfips'].unique()
    l_x_train = 5
    rezult = pd.DataFrame(columns=['cfips', 'microbusiness_density', 'error', 'l_regr', 'model', 'err'])
    for cfips in unic_cfips[1:100]:
        loc_train = train[train['cfips'] == cfips].copy()
        ideal_preds, ideal_error = equality_model(loc_train)
        ideal_l_x_train = 1
        model = '='
        for l_x_train in range(2,10):
            preds, error = line_regress(loc_train, l_x_train)
            err = error - ideal_error
            if err < 0:
                ideal_error = error
                ideal_preds = preds
                ideal_l_x_train = l_x_train
                model = 'lr'
            if abs(err) < 0.000000001:
                model = '=!='
        rezult.loc[len(rezult.index)] = [cfips, ideal_preds, ideal_error, ideal_l_x_train, model, err]
    rezult.to_csv("C:\\kaggle\\МикроБизнес\\rezult.csv", index=False)
    print(rezult.groupby(['model']).count())
    print(rezult[rezult['model']=='lr'].groupby(['l_regr']).count())
    print('Положительная ошибка это плохо', rezult['err'].sum())
    return rezult


train, test = start()
#train = predobrabotka(train)
train = obrabotka_filtr.super_filtr(train)
rezult = oll_line_regress(train)
rezult = rezult[["cfips", "microbusiness_density"]]
rezult.set_index("cfips", inplace=True)
test = test.join(rezult, on="cfips")
test[["row_id", "microbusiness_density"]].to_csv("C:\\kaggle\\МикроБизнес\\rezult1331.csv", index=False)


