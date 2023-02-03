import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm # прогресс бар

# определение среднего размера ошибки
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)

# получение вектора ошибоk
def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * smap

# загрузка файлов
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    census = pd.read_csv("C:\\kaggle\\МикроБизнес\\census_starter.csv")
    return train, test, census

# объединенный массив трейна и теста, создание объединенного raw
def maceraw(train, test):
    train['istest'] = 0
    test['istest'] = 1
    # объединенный массив трейна и теста
    raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year
    raw["month"] = raw["first_day_of_month"].dt.month
    # перенумерация внутри группы - номер месяца
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
    # кодирование целыми числами
    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    return raw

# обработка файла census
def censusdef(census, raw):
    columns = ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']
    raw[columns] = 0
    for index, row_census in census.iterrows():
        for year in range(2017, 2022):
            for col in columns:
                ncol = col+'_'+str(year)
                maska = (raw['year']==year+2)&(raw['cfips']==row_census['cfips'])
                raw.loc[maska, col] = row_census[ncol]
    return raw

#SMAPE — это относительная метрика, поэтому цель raw['target'] преобразована и равна отношению
# 'microbusiness_density' за следующий месяц к текущему значению 'microbusiness_density' - 1
def chang_target(raw):
    # 'target' ='microbusiness_density' следующего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    raw['target'] = raw['target'] / raw['microbusiness_density'] - 1

    raw['target'].fillna(0, inplace=True) # с и без, одинаково  Ошибка SMAPE: 1.0803235911612419, хуже чем моделью = 1371

    raw.loc[(raw['target'] < - 0.0054), 'target'] = - 0.0054
    raw.loc[(raw['target'] > 0.0054), 'target'] = 0.0054

    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# создаем новые столбцы 'lastactive'
def mace_fold(raw):
    # 'lastactive' содержит значение 'active' одинаковое для всех строк с одним 'cfips'
    # и равное 'active' за последний месяц (сейчас 2022-10-01)
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    return raw

# создаем 4 лага 'mbd_lag_{lag}' - (отношение 'microbusiness_density' следующего
# месяца к текущему - 1); 4 лага 'act_lag_{lag} - (разность 'active' следующей
# через лаг строки и текущей строки)' и столбцы 'mbd_rollmea{window}_{lag}' -
# суммы значений в окнах
def build_features(raw, target='target', target_act='active', lags=4):
    # 'target' ='microbusiness_density' следующего месяца деленное на
    # 'microbusiness_density' текущего месяца - 1 при том же 'cfips'
    lags = 4 # lags > 4 стабильно хуже.
    feats = []  # список имен лагов и сумм окон
    # создаем лаги target='microbusiness_density' от 1 до lags1
    for lag in range(1, lags):   #XGB. Ошибка SMAPE: 1.07924633283354
        # shift - сдвиг на определеное кол. позиций
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)

        # без этого Ошибка SMAPE: 1.0806806350184754, хуже чем моделью = 1362
        raw[f'mbd_lag_{lag}'].fillna(0, inplace = True) # с этим Ошибка SMAPE: 1.0803235911612419, хуже чем моделью = 1371

        # из значения 'active' следующей через лаг строки вычитается значение
        # текущей строки. diff - разность элемента df с элементом через lag
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
    lag = 1
    #создаем значения сумм окон для lag = 1
    for window in [2, 4, 6]: # пока оптимально
        # сгруппированно по 'cfips' 1-й лаг трансформируем - считаем сумму в окнах
        # размером [2, 4, 6, 8, 10]
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(
            lambda s: s.rolling(window, min_periods=1).sum())
        # raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')

    features = ['state_i']
    # Проверенно бесполезные колонки "month",
    features += feats  # список имен лагов и сумм окон
    features += ['proc_covill', 'Population', 'pct_college', 'median_hh_inc'] #XGB. Ошибка SMAPE: 1.0802560258625178, уже чем моделью = 1192
    #features += ['Population', 'pct_college', 'median_hh_inc']
    #features += ['sp500'] #

    print(features)
    return raw, features
# бесполезны - 'pct_bb', 'pct_foreign_born', 'pct_it_workers'
# XGB. Ошибка SMAPE: 1.0813450261766775/ количство cfips предсказанных хуже чем моделью = 1232

def build_features_1(raw, target='target', target_act='active', lags=4): # 1.436702 -0.047121  154.071433
    train_col = []  # список полей используемых в трайне
    # создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag_1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag_1'].fillna(method='bfill', inplace=True)
    train_col.append('mbd_lag_1')
    # вместе с target_lag error 1.421159 -0.110187

    # создаем лаги 'microbusiness_density' с учетом сглаживания от 1 до lags
    # for lag in range(1, 12):#12):
    #     # с учетом сглаживания # shift - сдвиг на определеное кол. позиций
    #     #raw[f'mbd_lag{lag+1}'] = raw[f'mbd_lag{lag}'] - raw.groupby('cfips')['mbd_gladkaya_dif'].shift(lag)
    #
    #     # без учета сглаживания
    #     raw[f'mbd_lag{lag + 1}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag+1)
    #
    #     raw[f'mbd_lag{lag+1}'].fillna(method='bfill', inplace=True)

    # создаем лаги 'target' с учетом сглаживания от 1 до lags
    # for lag in range(1, 12): #
    #     raw[f'target_lag{lag}'] = raw.groupby('cfips')[f'mbd_lag{lag}'].shift()
    #     # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    #     raw[f'target_lag{lag}'] = raw[f'mbd_lag{lag}'] / raw[f'target_lag{lag}'] - 1
    #     raw.loc[raw[f'mbd_lag{lag}'] == 0, f'target_lag{lag}'] = 0
    #     train_col.append(f'target_lag{lag}')  # список полей используемых в трайне
    # error 1.428251 -0.03867  151.777053

    #создаем лаги 'target' без учета сглаживания
    for lag in range(1, 12): # 1.379084  0.010497  146.812119
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')

    #ЭФФЕКТИВНО
    li = [3, 4, 14, 17] # error - 1.379084  0.010497  146.812119
    # создаем скользящие средние.
    for i in li:
        nam = f'EMA_{i}'
        EMA = pd.Series(raw['target_lag1'].ewm(span=i, adjust=False, min_periods=1).mean(), name=nam)
        raw[nam] = EMA
        train_col += [nam]

    #создаем значения сумм окон для lag = 1
    for i in [2]: #li = [3, 4, 14, 17] и param=2 - 1.428251 -0.038670  151.777053
        nam = f'roll_{i}'
        # сгруппированно по 'cfips' 1-й лаг трансформируем - считаем сумму в окнах
        ROLL = raw.groupby('cfips')['target_lag1'].transform(lambda s: s.rolling(i, min_periods=1).sum())
        raw[nam] = ROLL
        train_col += [nam]

    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['active_lag1']
    train_col += ['state_i', 'proc_covill', 'pct_college', 'median_hh_inc', 'sp500', 'month']
    return raw, train_col

# создание модели градиентного бустинга
def model_XGBRegressor():
    # Создаем модель
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate= 0.0071, #param, #0.0108, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=2,
    )
    return model

# обучение модели
def learn_model(model, raw, train_indices, valid_indices):
    dfXtrain = raw.loc[train_indices, features]
    dfytrain = raw.loc[train_indices, 'target'].clip(-0.0043, 0.0045)

    dfXtest = raw.loc[valid_indices, features]
    dfy_test = raw.loc[valid_indices, 'target']
    # обучение модели
    model.fit(dfXtrain, dfytrain) #Ошибка SMAPE: 1.3849105539299413
    return model

# предсказание на основе модели
def predsk_model(raw, valid_indices, model):
    # предсказание модели на основе валидационной выборки и списка имен столбцов
    ypred = model.predict(raw.loc[valid_indices, features])
    raw.loc[valid_indices, 'ypred_target'] = ypred
    # получаем из значения предсказанного ypred, который есть 'target'
    # предсказанное значение 'microbusiness_density' и помещаем его в 'k'
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    return raw

# Не валидация. Заменяем часть предсказаний модели на предсказание модели '='
def valid_rez(raw, TS):
    # Валидация
    # создаем словари 'microbusiness_density' и 'k'
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()[
        'microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    # создаем новый датафрейм с данными TS+1 месяца
    df = raw.loc[raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state',
                                          'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    # создаем колонку предсказание
    df['pred'] = df['cfips'].map(dt)
    # создаем колонку реальность 'microbusiness_density' TS-го месяца
    df['lastval'] = df['cfips'].map(lastval)

    # создаем колонку предсказаний
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    # создаем колонку предыдущих значений
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values
    print(f'Месяц валидации - TS: {TS}')
    # print('Равенство последнему значению. Ошибка SMAPE:',
    #       smape(df['microbusiness_density'], df['lastval']))
    # print('Предсказано XGB. Ошибка SMAPE:',
    #       smape(df['microbusiness_density'], df['pred']))
    print()
    return raw

def model(raw):
    raw['ypred_last'] = np.nan
    raw['ypred'] = np.nan
    raw['k'] = 1.
    BEST_ROUNDS = []  # лучшие раунды
    # TS - номер месяца валидацииции. Месяцы то TS это трайн.
    for TS in range(29, 38):
        #print(TS)
        # создание модели
        model = model_XGBRegressor()
        # маска тренировочной выборки
        train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1)
        # маска валидационной выборки. Валидация по 1 месяцу
        valid_indices = (raw.istest == 0) & (raw.dcount == TS)
        # обучение модели
        model = learn_model(model, raw, train_indices, valid_indices)
        BEST_ROUNDS.append(model.best_iteration)  # добавляем в список лучших раундов
        # предсказание на основе модели
        raw = predsk_model(raw, valid_indices, model)
        raw = valid_rez(raw, TS)
    # маска при которой вероятно все хорошо
    ind = (raw.dcount >= 30) & (raw.dcount <= 38)
    print('Предсказано XGB. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred']))
    print('Равенство последнему значению. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred_last']))

def kol_error(raw):
    # Выводы
    print('Выводы 1')
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['ypred_last'])
    # создаем датафрейм со столбцами error и error_last
    dt = raw.loc[(raw.dcount >= 30) & (raw.dcount <= 38)].groupby(['cfips', 'dcount'])[['error', 'error_last']].last()
    # преобразуем dt в серию булевых значений 'miss'
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))

def rezultat(raw):
    best_rounds = 794
    TS = 38
    print(TS)
    # создание моделей
    model0 = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=best_rounds,
        learning_rate=0.0075,
        max_leaves=31,
        subsample=0.60,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
    )
    model1 = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=best_rounds,
        learning_rate=0.0075,
        max_leaves=31,
        subsample=0.60,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
    )
    # тренировочная выборка
    train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1)
    # валидационная выборка
    valid_indices = (raw.dcount == TS)
    # обучение моделей
    model0.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046),
    )
    model1.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046),
    )
    # среднее предсказание моделей
    ypred = (model0.predict(raw.loc[valid_indices, features]) + model1.predict(raw.loc[valid_indices, features])) / 2
    # преобразуем 'k' к 'microbusiness_density'
    raw.loc[valid_indices, 'k'] = ypred + 1.
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    # Валидация
    # Приводим 'microbusiness_density' и 'k' к виду словарей
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()[
        'microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    # создаем новый датафрейм с данными TS+1 месяца
    df = raw.loc[
        raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(
        drop=True)
    # создаем колонку - предсказание
    df['pred'] = df['cfips'].map(dt)
    # создаем колонку реальность 'microbusiness_density' TS-го месяца
    df['lastval'] = df['cfips'].map(lastval)

    # если 'microbusiness_density' ниже порога, то предсказание равно реальности
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values

    raw.loc[raw['cfips'] == 28055, 'microbusiness_density'] = 0
    raw.loc[raw['cfips'] == 48269, 'microbusiness_density'] = 1.762115

    dt = raw.loc[raw.dcount == 39, ['cfips', 'ypred']].set_index('cfips').to_dict()['ypred']
    test = raw.loc[raw.istest == 1, ['row_id', 'cfips', 'microbusiness_density']].copy()
    test['microbusiness_density'] = test['cfips'].map(dt)

    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\dlia_sverki.csv', index=False)

# train, test, census = start() # загрузка файлов
# raw = maceraw(train, test) # объединенный массив трейна и теста, создание объединенного raw
# raw = censusdef(census, raw) # объудинение census и raw
# raw.to_csv("C:\\kaggle\\МикроБизнес\\raw0.csv", index = False)
raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])

#raw = del_outliers(raw) # УДАЛЕНИЕ ВЫБРОСОВ. Предсказано XGB. Ошибка SMAPE: 1.0806806350184754, хуже чем моделью = 1362

raw = chang_target(raw) # изменение цели
raw = mace_fold(raw) # создаем новые столбцы 'lastactive'
raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])

raw, features = build_features(raw, 'target', 'active', lags = 4) # создаем лаги и суммы в окнах
model(raw)
kol_error(raw) # анализ ошибок собранных по всем моделям
#rezultat(raw)

# XGB. Ошибка SMAPE: 1.0805449694484606
