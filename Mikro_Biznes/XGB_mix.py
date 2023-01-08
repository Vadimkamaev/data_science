import numpy as np, pandas as pd

import gc # сборщик мусора
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm # прогресс бар

from servise_ds import okno
import obrabotka_filtr

not_blec = 1 # 1 - без блек листа
if not_blec == 1:
    blacklistcfips =pd.DataFrame(columns=[['cfips','miss']])
    #без блек листов XGB. 1.0813450261766775/ количство cfips предсказанных хуже чем моделью = 1232
else:
    # БЛЕК ЛИСТ СОХРАНЯЮЩИЙСЯ В ФАЙЛЕ 1.0692455614504854
    # список 'cfips' у которых ошибка модели больше чем ошибка модели '='
    blacklistcfips = pd.read_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv")

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

# УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
def del_outliers(raw):
    outliers = []  # список выбросов
    cnt = 0  # счетчик выбросов
    vsego = 0
    # цикл по уникальным cfips
    for o in tqdm(raw.cfips.unique()):  # tqdm - прогресс бар
        indices = (raw['cfips'] == o)  # маска отфильтрованная по cfips
        tmp = raw.loc[indices].copy().reset_index(drop=True)  # df фильтрован по cfips
        # массив значений microbusiness_density
        var = tmp.microbusiness_density.values.copy()
        # vmax = np.max(var[:38]) - np.min(var[:38])

        # цикл назад от предпоследнего до 2-го элемента
        for i in range(37, 2, -1):
            # среднее значение microbusiness_density с 0-го по i-й элемент * 0.2
            #thr = 0.15 * np.mean(var[:i]) # XGB. Ошибка SMAPE: 1.1366349123015205
            thr = 0.10 * np.mean(var[:i]) # 1.0863 - 22 место
            #thr = 0.05 * np.mean(var[:i]) # 1.0865
            # difa = abs(var[i] - var[i - 1])
            # if (difa >= thr):  # если microbusiness_density изменился больше чем на 20%
            #     #var[:i] *= (var[i] / var[i - 1])  # меняем все значения до i-го
            #     var[:i] += (var[i] / var[i - 1])  # меняем все значения до i-го
            #     outliers.append(o)  # добавляем cfips в список выбросов
            #     cnt += 1  # счетчик выбросов
            # else:
            # разность i-го и i-1-го значения microbusiness_density
            difa = var[i] - var[i - 1] #XGB. Ошибка SMAPE: 1.161117561342498
            if (difa >= thr) or (difa <= -thr):  # если microbusiness_density изменился больше чем на 20%
                if difa > 0:
                    var[:i] += difa - 0.0045 # 0.0045 лучше чем 0.003 и чем 0.006
                else:
                    var[:i] += difa + 0.0043 # 0.0043 лучше чем 0.003 и чем 0.006
                # Предсказано XGB.Ошибка SMAPE: 0.9178165883836801
                # Равенство последнему значению. Ошибка SMAPE: 1.0078547907542301
                outliers.append(o)  # добавляем cfips в список выбросов
                cnt += 1  # счетчик выбросов
            vsego += 1
        var[0] = var[1] * 0.99
        raw.loc[indices, 'microbusiness_density'] = var
    outliers = np.unique(outliers)
    print(len(outliers), cnt, cnt/vsego*100)
    return raw

#SMAPE — это относительная метрика, поэтому цель raw['target'] преобразована и равна отношению
# 'microbusiness_density' за следующий месяц к текущему значению 'microbusiness_density' - 1
def chang_target(raw):
    # 'target' ='microbusiness_density' следующего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    raw['target'] = raw['target'] / raw['microbusiness_density'] - 1
    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# создаем новые столбцы 'lastactive' и 'lasttarget'
def mace_fold(raw):
    # 'lastactive' содержит значение 'active' одинаковое для всех строк с одним 'cfips'
    # и равное 'active' за последний месяц (сейчас 2022-10-01)
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    # серия из 'cfips' и 'microbusiness_density' 28-го месяца из трайн
    dt = raw.loc[raw.dcount == 28].groupby('cfips')['microbusiness_density'].agg('last')
    # 'lasttarget' содержит значение 'microbusiness_density' 28-го месяца из трайн
    # одинаковое для всех строк с одним 'cfips'
    raw['lasttarget'] = raw['cfips'].map(dt)
    # в пределах от 0 до 8000
    raw['lastactive'].clip(0, 8000).hist(bins=30)
    return raw

# скользящая средняя
def weighted_moving_average(df, column, n):
    nam = 'EMA_' + str(n)
    EMA = pd.Series(df[column].ewm(span=n, adjust=False, min_periods=n).mean(), name = nam)
    df = df.join(EMA)
    return df, nam

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
        # из значения 'active' следующей через лаг строки вычитается значение
        # текущей строки. diff - разность элемента df с элементом через lag
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
    lag = 1
    #создаем значения сумм окон для lag = 1
    for window in [2, 4, 6]: # пока оптимально
        # сгруппированно по 'cfips' 1-й лаг трансформируем - считаем сумму в окнах
        # размером [2, 4, 6]
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(
            lambda s: s.rolling(window, min_periods=1).sum())
        # raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')

    # создаем скользящие средние для lag = 1 без них Ошибка SMAPE: 1.0813450261766775
    # for ss in [4,6,8]:
    #     raw, name = weighted_moving_average(raw, 'mbd_lag_1', ss)
    #     feats.append(name)

    features = ['state_i']  # номера штатов
    # Проверенно бесполезные колонки "month",
    features += feats  # список имен лагов и сумм окон
    features += ['pct_college', 'median_hh_inc'] #XGB. Ошибка SMAPE: 1.0813450261766775
    print(features)
    return raw, features
# бесполезны - 'pct_bb', 'pct_foreign_born', 'pct_it_workers'
# ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']
# XGB. Ошибка SMAPE: 1.0813450261766775/ количство cfips предсказанных хуже чем моделью = 1232


# создание модели градиентного бустинга
def model_XGBRegressor():
    model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=4999,
        learning_rate=0.0075,
        max_leaves=17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
        early_stopping_rounds=70,
    )
    return model

# обучение модели
def learn_model(model, raw, train_indices, valid_indices):
    # обучение модели
    model.fit(
        # Трайн. features - список имен столбцов
        raw.loc[train_indices, features],
        # y(игрик) модели
        raw.loc[train_indices, 'target'].clip(-0.0043, 0.0045),
        # валидационная выборка из 1 месяца
        eval_set=[(raw.loc[valid_indices, features], raw.loc[valid_indices, 'target'])],
        verbose=500,
    )
    return model

# предсказание на основе модели
def predsk_model(raw, valid_indices, model):
    # предсказание модели на основе валидационной выборки и списка имен столбцов
    ypred = model.predict(raw.loc[valid_indices, features])
    # получаем из значения предсказанного ypred, который есть 'target'
    # предсказанное значение 'microbusiness_density' и помещаем его в 'k'
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    return raw

def blacklist(df):
    global blacklistcfips, not_blec
    if not_blec != 1: # 1 - без блек листа
        pass
        for index, row in blacklistcfips.iterrows():
            cfips = row['cfips']
            # XGB. Ошибка SMAPE: 1.0692455614504854 / 1.0871
            if (row['miss'] > 0.5) | (row['hit'] < 0):
                df.loc[df['cfips'] == cfips, 'pred'] = df.loc[df['cfips'] == cfips, 'lastval']
            # XGB.Ошибка SMAPE: 1.072776015639455 / 1.088
            # if (row['miss'] > 0.5):
            #       df.loc[df['cfips'] == cfips, 'pred'] = df.loc[df['cfips'] == cfips, 'lastval']
            # XGB. Ошибка SMAPE: 1.0459393718374939 / 1.0891
            # if (row['hit'] < 0):
            #     df.loc[df['cfips'] == cfips, 'pred'] = df.loc[df['cfips'] == cfips, 'lastval']

            # XGB. Ошибка SMAPE: 1.1003233174468263 / 1.0875
            # if (row['miss'] > 0.4):
            #       df.loc[df['cfips'] == cfips, 'pred'] = df.loc[df['cfips'] == cfips, 'lastval']

    return df

# Валидация
def valid_rez(raw, TS, ACT_THR, ABS_THR):
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
    # если active маленький то предсказание равно реальности TS-го месяца
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    # если 'microbusiness_density' ниже порога, то предсказание равно реальности
    df.loc[df['lastval'] <= ABS_THR, 'pred'] = df.loc[df['lastval'] <= ABS_THR, 'lastval']
    # если 'cfips' в черном списке то предсказание равно реальности TS-го месяца
    df = blacklist(df)
    #df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
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
    ACT_THR = 1.8  # условие попадания в обучающую выборку raw.lastactive>ACT_THR
    ABS_THR = 1.00  # условие попадания в обучающую выборку raw.lasttarget>ABS_THR
    raw['ypred_last'] = np.nan
    raw['ypred'] = np.nan
    raw['k'] = 1.
    BEST_ROUNDS = []  # лучшие раунды
    # TS - номер месяца валидацииции. Месяцы то TS это трайн.
    for TS in range(29, 38):
    #for TS in range(36, 38):
        #print(TS)
        # создание модели
        model = model_XGBRegressor()
        # маска тренировочной выборки
        train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR) & (
                raw.lasttarget > ABS_THR)
        # маска валидационной выборки. Валидация по 1 месяцу
        valid_indices = (raw.istest == 0) & (raw.dcount == TS)
        # обучение модели
        model = learn_model(model, raw, train_indices, valid_indices)
        BEST_ROUNDS.append(model.best_iteration)  # добавляем в список лучших раундов
        # предсказание на основе модели
        raw = predsk_model(raw, valid_indices, model)
        raw = valid_rez(raw, TS, ACT_THR, ABS_THR)
    # маска при которой вероятно все хорошо
    ind = (raw.dcount >= 30) & (raw.dcount <= 38)
    print('Предсказано XGB. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred']))
    print('Равенство последнему значению. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred_last']))

def kol_error(raw):
    global blacklistcfips
    # Выводы
    print('Выводы 1')
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['ypred_last'])
    # создаем датафрейм со столбцами error и error_last
    dt = raw.loc[(raw.dcount >= 30) & (raw.dcount <= 38)].groupby(['cfips', 'dcount'])[['error', 'error_last']].last()
    # преобразуем dt в серию булевых значений 'miss'
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    ser_err = dt.groupby('cfips')['error'].mean()
    ser_error_last = dt.groupby('cfips')['error_last'].mean()
    df_dt = pd.DataFrame({'cfips': seria_dt.index, 'miss': seria_dt, 'hit':(ser_error_last-ser_err)})
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))
    if not_blec == 1: # 1 - без блек листа
        df_dt.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)
    # else:
    #     blacklistcfips = blacklistcfips.set_index('cfips')
    #     blacklistcfips['miss2'] = df_dt['miss']
    #     #df_dt.rename(columns={'miss': 'miss2'})
    #     blacklistcfips.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)
    #     pass

def rezultat(raw):
    ACT_THR = 1.8  # условие попадания в обучающую выборку raw.lastactive>ACT_THR
    ABS_THR = 1.00  # условие попадания в обучающую выборку raw.lasttarget>ABS_THR
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
    train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR) & (
                raw.lasttarget > ABS_THR)
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
    # если active маленький то предсказание равно реальности TS-го месяца
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    # если 'microbusiness_density' ниже порога, то предсказание равно реальности
    df.loc[df['lastval'] <= ABS_THR, 'pred'] = df.loc[df['lastval'] <= ABS_THR, 'lastval']
    # если 'cfips' в черном списке то предсказание равно реальности TS-го месяца
    df = blacklist(df)
    #df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
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
# raw = del_outliers(raw) # УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
# raw = chang_target(raw) # изменение цели
# raw = mace_fold(raw) # создаем новые столбцы 'lastactive' и 'lasttarget'
# raw.to_csv("C:\\kaggle\\МикроБизнес\\raw1.csv", index = False)
raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw1.csv")
raw, features = build_features(raw, 'target', 'active', lags = 4) # создаем лаги и суммы в окнах
model(raw)
kol_error(raw)
rezultat(raw)


# maincfips = blacklistcfips[blacklistcfips['main']]['cfips']
# raw_good = raw[raw['cfips'].isin(maincfips)].copy()
# model(raw_good)
# kol_error(raw_good)
#
# mikscfips = blacklistcfips[blacklistcfips['miks']]['cfips']
# raw_miks = raw[raw['cfips'].isin(mikscfips)].copy()
# model(raw_miks)
# kol_error(raw_miks)

# lastcfips = blacklistcfips[blacklistcfips['last']]['cfips']
# raw_last = raw[raw['cfips'].isin(lastcfips)].copy()
# model(raw_last)
# kol_error(raw_last)

# new_raw = pd.concat([maincfips, mikscfips])
# kol_error(new_raw)


