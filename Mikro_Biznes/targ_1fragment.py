import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno
import obrabotka_filtr

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

# создание нового таргета
def new_target1(raw, param, param2):
    # 'target' ='microbusiness_density' предыдущего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    # 'target' ='microbusiness_density' текущего месяца делить на предыдущего месяца - 1
    raw['target'] = raw['microbusiness_density'] / raw['target'] - 1
    raw.loc[(raw['microbusiness_density'] == 0)|(raw['target'] > 10),'target'] = 0
    raw['target'].fillna(0, inplace=True)
    # raw.loc[(raw['target'] < - 0.0069), 'target'] = - 0.0069 # error 1.100728  0.028686     0.003218
    # raw.loc[(raw['target'] > 0.0081), 'target'] = 0.0081  # error 1.100728  0.028686     0.003218
    raw.loc[(raw['target'] < - param), 'target'] = - param # error 1.100728  0.028686     0.003218
    raw.loc[(raw['target'] > param + 0.0034), 'target'] = param + 0.0034  # error 1.100728  0.028686     0.003218
    # raw.loc[(raw['target'] > param), 'target'] = param
    # raw.loc[(raw['target'] < -param2), 'target'] = -param2
    return raw

# создание нового таргета
def new_target2(raw, param):
    k = 0.0017
    # 'target' ='microbusiness_density' предыдущего месяца при том же 'cfips'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    #y_lin_pred = raw['mbd_lag1'] * 1.0016 + 0.0018
    raw['target'] = (raw['microbusiness_density'] - raw['mbd_lag1'])/(raw['mbd_lag1']+1)
    raw['target'].fillna(0, inplace=True)
    # raw.loc[(raw['target'] < k - param), 'target'] = k - param # error 1.100728  0.028686     0.003218
    # raw.loc[(raw['target'] > param + k), 'target'] = param + k


    # raw.loc[(raw['target'] < - param2), 'target'] = - param2 #
    # raw.loc[(raw['target'] > param), 'target'] = param #
    # 0.007 на 0.026
    raw.loc[(raw['target'] > 0.1), 'target'] = 0.1
    raw.loc[(raw['target'] < -0.1), 'target'] = -0.1
    return raw


# создание лагов  error 1.379034  0.010546  146.796656
def build_lag(raw, param): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 17): #17 -  1.409514  0.009615   144.354724
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    #создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1', 'median_hh_inc', 'month']
    k = 0.0017
    raw.loc[(raw['target'] < k - param), 'target'] = k - param # error 1.100728  0.028686     0.003218
    raw.loc[(raw['target'] > param + k), 'target'] = param + k
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки 1.408914 -0.097942
    maska_train = (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test


# считаем сколько предсказаний лучше
def baz_otchet(raw, mes_1, mes_val):
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[(raw.dcount >= mes_1) & (raw.dcount <= mes_val)].groupby(['cfips', 'dcount'])[
        ['error', 'error_last', 'lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last']  # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    seria_dt = seria_dt.loc[seria_dt >= 0.50]  # оставляем только те, где ['miss'].mean() >= 0.5

    hor = (raw['error_last'] < raw['error']).sum()
    ploh = (raw['error'] < raw['error_last']).sum()

    return len(seria_dt), ploh, hor

# считаем сколько предсказаний лучше после 1-й модели
def otchet(raw, mes_1, mes_val):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l, ploh, hor = baz_otchet(raw, mes_1, mes_val)
    vse = len(raw)
    print('количство cfips предсказанных хуже чем моделью =', l, 'Отношение плохих предсказаний к хорошим', ploh/hor), 'Кол. хороших', hor, 'Плохих', ploh,
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', vse-hor-ploh)

# вывод на печать информацию по качеству работы модели в каждом штате
def print_state(raw, maska):
    df_state = pd.DataFrame({'state_i':0, 'state': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for state in raw.state.unique():
        maska2 = (raw['state'] == state) & maska
        state_i = raw.loc[maska2, 'state_i']
        state_i = state_i.iloc[0]
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (state_i, state, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# вывод на печать информацию по качеству работы модели в каждом месяце
def print_month(raw, maska):
    df_state = pd.DataFrame({'month': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for dcount in raw[maska].dcount.unique():
        maska2 = (raw['dcount'] == dcount) & maska
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (dcount, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    #df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
            (raw.cfips <= max_cfips)  & (~raw['cfips'].isin(blac_test_cfips))
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)
    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)
    dif_err = err_last - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_mod, dif_err, 0]
    #print_month(raw, maska)
    return rezult

def vsia_model1(raw, mes_1, mes_val, train_col, blac_cfips, param=0):
    # получение трайна и 'y' (игрик) для модели
    df = raw[~raw['cfips'].isin(blac_cfips)]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    # Создаем модель
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate=0.0071, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=3,
    )
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)

    # mask = raw.dcount == mes_val
    # sppol = []
    # for lag in range(1, 17):
    #     sppol.append(f'target_lag{lag}')
    # delt = 0.8
    # quan = raw.loc[mask, sppol].abs().quantile(delt, axis=1)
    # quan = quan.values
    # y_pred[y_pred > quan] = quan[y_pred > quan]
    # y_pred[y_pred < -quan] = -quan[y_pred < -quan]

    #1.204584  0.007842    -0.000338

    # y_pred = y_pred + 1
    # y_pred = raw.loc[raw.dcount == mes_val, 'mbd_lag1'] * y_pred

    # сохраняем результат обработки одного цикла
    # raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred

    return y_pred

# формирование колонки 'ypred' по результатам 1-го процесса оптимизации
def post_model1(raw, y_pred, mes_val):
    maska = raw.dcount == mes_val
    # преобразовываем target в 'microbusiness_density'
    y_pred = y_pred + 1
    y_pred = raw.loc[maska, 'mbd_lag1'] * y_pred
    # сохраняем результат обработки одного цикла
    #maska =(~raw['cfips'].isin(blac_test_cfips))&(raw.dcount == mes_val)
    raw.loc[maska, 'ypred'] = y_pred
    return raw

# формирование колонки 'ypred' по результатам 1-го процесса оптимизации
def post_model2(raw, y_pred, mes_val):
    maska = raw.dcount == mes_val
    # преобразовываем target в 'microbusiness_density'
    # y_pred = y_pred + 1
    y_pred = raw.loc[maska, 'mbd_lag1'] * (1+y_pred)+y_pred
    # сохраняем результат обработки одного цикла
    #maska =(~raw['cfips'].isin(blac_test_cfips))&(raw.dcount == mes_val)
    raw.loc[maska, 'ypred'] = y_pred
    return raw

# МУЛЬТ
# Моульт для следующего месяца
def mult1(raw, mes_val, param, param2):
    kol_mes = 12
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [param2, param]}
    maska = (raw.dcount >= mes_val - kol_mes) & (raw.dcount < mes_val)
    train_data = raw[maska].copy()
    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag1'] + mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred'])
    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    #
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1'] + raw.loc[maska, 'best_mult']
    raw.loc[maska, 'multi'] = raw.loc[maska, 'best_mult']
    raw.drop('best_mult', axis=1, inplace = True)


    raw['ypred'].fillna(0, inplace = True)
    return raw

# Моульт для месяца через 1
def mult2(raw, mes_val):
    kol_mes = 4
    # raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag2'] = raw.groupby('cfips')['microbusiness_density'].shift(2)
    maska = (raw.dcount >= mes_val-1 - kol_mes) & (raw.dcount < mes_val -1)
    train_data = raw[maska].copy()
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [0, 0.013]}
    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag2'] + mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred'])
    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska,'mbd_lag2'] + raw.loc[maska,'best_mult']
    raw.drop('best_mult', axis=1, inplace = True)

    # raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2'] * 1.0058 #(1 + 58 / 10000) # 58/10000
    raw['ypred'].fillna(0, inplace = True)
    return raw

# Моульт для месяца через 2
def mult3(raw, mes_val):
    kol_mes = 6
    raw['mbd_lag3'] = raw.groupby('cfips')['microbusiness_density'].shift(3)
    maska = (raw.dcount >= mes_val-2 - kol_mes) & (raw.dcount < mes_val -2)
    train_data = raw[maska].copy()
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [0, 0.023]}
    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag3'] + mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred'])
    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska,'mbd_lag3'] + raw.loc[maska,'best_mult']
    raw.drop('best_mult', axis=1, inplace = True)
    raw['ypred'].fillna(0, inplace = True)
    return raw

# ЛАЙНЕ
# line для следующего месяца
def line1(raw, mes_val):
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1'] * 1.0016 + 0.0018
    maska = maska & (raw.lastactive < 220)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# line для месяца через 1
def line2(raw, mes_val):
    raw['mbd_lag2'] = raw.groupby('cfips')['microbusiness_density'].shift(2)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2'] * 1.0039 + 0.0033 # error 1.692963 -0.563549
    maska = maska & (raw.lastactive < 150)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# line для месяца через 2
def line3(raw, mes_val):
    raw['mbd_lag3'] = raw.groupby('cfips')['microbusiness_density'].shift(3)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag3'] * 1.006 + 0.0068 # error 2.089475 -0.960060
    maska = maska & (raw.lastactive < 100)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag3']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# line для месяца через 3
def line4(raw, mes_val):
    raw['mbd_lag4'] = raw.groupby('cfips')['microbusiness_density'].shift(4)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag4'] * 1.0084 + 0.0099 # error 2.468962 -1.339547
    maska = maska & (raw.lastactive < 50)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag4']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# С ПАРАМЕТРАМИ ДЛЯ НАСТРОЙКИ

def line1_param(raw, mes_val, param, param2):
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1'] * (1 + param) + param2

    # maska150 = maska & (raw['lastactive'] >= 150) & (raw['lastactive'] < 1000)
    # raw.loc[maska150, 'ypred'] = raw.loc[maska150, 'mbd_lag1'] + 0.0044

    # maska1000 = maska & (raw['lastactive'] >= 1000) & (raw['lastactive'] < 2000)
    # raw.loc[maska1000, 'ypred'] = raw.loc[maska1000, 'mbd_lag1'] *(1.0005) + 0.004

    # maska2000 = maska & (raw['lastactive'] >= 2000) & (raw['lastactive'] < 5000)
    # raw.loc[maska2000, 'ypred'] = raw.loc[maska2000, 'mbd_lag1'] *(1.0005) + 0.006

    # maska5000 = maska & (raw['lastactive'] >= 5000)
    # raw.loc[maska5000, 'ypred'] = raw.loc[maska5000, 'mbd_lag1'] *(1.0015) + 0.01

    #raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1'] * 1.0016 + 0.0018 # средняя
    maska = maska & (raw.lastactive < 135)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# line для месяца через 1
def line2_param(raw, mes_val, param, param2):
    raw['mbd_lag2'] = raw.groupby('cfips')['microbusiness_density'].shift(2)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2'] * (1+param) + param2

    # maska150 = maska & (raw['lastactive'] >= 150) & (raw['lastactive'] < 1000)
    # raw.loc[maska150, 'ypred'] = raw.loc[maska150, 'mbd_lag2'] + 0.0095

    # maska1000 = maska & (raw['lastactive'] >= 1000) & (raw['lastactive'] < 2000)
    # raw.loc[maska1000, 'ypred'] = raw.loc[maska1000, 'mbd_lag2'] * 1.001) + 0.0105

    # maska2000 = maska & (raw['lastactive'] >= 2000) & (raw['lastactive'] < 5000)
    # raw.loc[maska2000, 'ypred'] = raw.loc[maska2000, 'mbd_lag2'] * 1.001) + 0.0145

    # maska5000 = maska & (raw['lastactive'] >= 5000)
    # raw.loc[maska5000, 'ypred'] = raw.loc[maska5000, 'mbd_lag2'] *(1.0015) + 0.026

    # raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2'] * 1.0039 + 0.0033 # средняя
    maska = maska & (raw.lastactive < 150)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag2']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# line для месяца через 2
def line3_param(raw, mes_val, param, param2):
    raw['mbd_lag3'] = raw.groupby('cfips')['microbusiness_density'].shift(3)
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag3'] * (1+param) + param2

    # maska150 = maska & (raw['lastactive'] >= 150) & (raw['lastactive'] < 1000)
    # raw.loc[maska150, 'ypred'] = raw.loc[maska150, 'mbd_lag3'] + 0.0170

    # maska1000 = maska & (raw['lastactive'] >= 1000) & (raw['lastactive'] < 2000)
    # raw.loc[maska1000, 'ypred'] = raw.loc[maska1000, 'mbd_lag3'] * 1.0015) + 0.0190


    #raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag3'] * 1.006 + 0.0068 # средняя error 2.089475 -0.960060
    maska = maska & (raw.lastactive < 100)
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag3']
    raw['ypred'].fillna(0, inplace=True)
    return raw

def modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips, param, param2):
    start_time = datetime.now()  # время начала работы модели
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно

    for mes_val in range(start_val, stop_val + 1):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        min_err = 10000
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 2 #
        # y_pred  = vsia_model1(raw, mes_1, mes_val, train_col, blac_cfips, param=0)
        # post_model2(raw, y_pred, mes_val)
        line3_param(raw, mes_val, param, param2)


    otchet(raw, start_val, stop_val)
    print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
    # print('Значимость колонок трайна в модели')
    return raw

# создание новых фиктивных cfips
def new_cfips(raw, lastactive, max_cfips, kol = 2):
    #kol - оптимально 2
    if max_cfips != raw['cfips'].max():
        print('Ошибка max_cfips')
    df = raw[raw['lastactive'] < lastactive].copy()
    df.sort_values('lastactive', ascending=False, inplace=True)
    if len(df.index) == 0:
        return raw
    list_cfips = df['cfips'].unique() #  массив уникальных cfips, которые группируются
    lec = len(list_cfips) # количество уникальных cfips, которые группируются
    le = lec + kol - lec%kol # длина матрицы - количество новых генерируемых cfips
    matrica = np.zeros((kol,le)) # матрица каждая строка которой - список cfips которые суммируются
    fin = le // kol  # номер элемента до которого все cfips использованы в 0-м ряду и больше их не применять
    for l in range(fin): # цикл по заполнению 0-го ряда
        for k in range(kol):
            x = list_cfips[l]
            matrica[0,l*kol+k] = x
    sled = fin # номер следующего cfips который вставляется в матрицу
    for k in range(1, kol): # цикл по заполнению рядов с 1 по последний
        for l in range(le-1,-1,-1): # цикл в котором заполняется один ряд матрицы
            if list_cfips[sled] == matrica[k-1, l]:
                sled+=1
            matrica[k, l] = list_cfips[sled]
            if sled == lec-1:
                sled = fin
            else:
                sled+=1
    # матрица заполнена, далее суммирование cfips
    # считываем стандартные колонки одинаковые у всех cfips
    cfips = list_cfips[0]
    dfcop = raw[raw['cfips']==cfips].copy()
    dfcop.reset_index(drop=True, inplace=True)
    dfcop[['row_id', 'cfips', 'county','state', 'first_day_of_month',
           'microbusiness_density', 'active',
           'county_i', 'pct_bb', 'pct_college', 'pct_foreign_born',
           'pct_it_workers', 'median_hh_inc', 'covill', 'covdeat', 'Population',
           'proc_covill', 'proc_covdeat', 'ypred',
           'error_otdelno', 'mbd_gladkaya', 'mbd_gladkaya_dif', 'mbd_lag1',
           'mbd_lag2', 'mbd_lag3', 'EMA_3', 'active_lag1', 'ypred_otdelno',
           'lastactive']] = 0
    dfcop['state_i']=100
    print('Последний цикл')
    for l in tqdm(range(le)):
        nas = 0
        gen_cfips = max_cfips+l+1 # номер нового сгенерированного cfips
        dfnew = dfcop.copy()
        dfnew['cfips'] = gen_cfips
        df =kol*[0]
        for i in range(kol):
            df[i] = raw[raw['cfips'] == matrica[i, l]].copy()
            df[i].reset_index(drop=True, inplace=True)
            dfnew['active'] = dfnew['active'] + df[i]['active']
            dfnew['lastactive'] = dfnew['lastactive'] + df[i]['lastactive']
            nas = nas + df[i]['Population'][0]
            # dfnew['proc_covill'] = dfnew['proc_covill'] + df[i]['proc_covill']*df[i]['Population'][0]
            # dfnew['pct_college'] = dfnew['pct_college'] + df[i]['pct_college'] * df[i]['Population'][0]
            # dfnew['median_hh_inc'] = dfnew['median_hh_inc'] + df[i]['median_hh_inc'] * df[i]['Population'][0]
        dfnew['microbusiness_density'] = dfnew['active']/nas
        # dfnew['proc_covill'] = dfnew['proc_covill']/nas
        # dfnew['pct_college'] = dfnew['pct_college']/nas
        # dfnew['median_hh_inc'] = dfnew['median_hh_inc'] /nas
        raw = pd.concat([raw,dfnew], ignore_index=True)
    return raw


def posle_sglashivfnia(raw, rezult):
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    # создаем блек-лист cfips которых не используем в тесте
    min_int = 1000
    max_int = 2000
    blac_test_cfips = raw.loc[(raw['lastactive'] < min_int) | (raw['lastactive'] >= max_int), 'cfips']
    blac_test_cfips = blac_test_cfips.unique()
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    train_col =[]
    for param1 in range(0, 20, 5): # при 30 - 1.404981  0.014148     0.000272
        param = param1 / 10000
        for param2 in range(50, 300, 5):
            param2 = param2/10000

            lastactive = -1

            blac_cfips = []
            # создаем гибриды из cfips для трайна
            # raw = new_cfips(raw, lastactive, max_cfips)
            raw = new_target2(raw,param)
            raw, train_col = build_lag(raw, param)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
            start_val = 28  # первый месяц валидации с которого проверяем модель
            stop_val = 38  # последний месяц валидации до которого проверяем модель
            #модель без блек листа
            # raw = modeli_po_mesiacam(raw, start_val, stop_val, train_col, [])
            # raw['y_no_blac'] = raw['ypred']
            # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_no_blac.csv", index=False)

            # валидация по результату обработки всех месяцев валидации в цикле
            raw= modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips, param, param2)
            rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, lastactive, param, param2)

            print('Сортировка по dif_err')
            rezult.sort_values(by='dif_err', inplace=True, ascending=False)
            print(rezult.head(50))



# МОДЕЛЬ.
def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    max_cfips = raw['cfips'].max() # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    lastactive = 40  # оптимально 205
    # for lastactive in range(46, 51, 1):
    #     for param in range(41, lastactive+1, 1):
    raw = raw[raw['cfips'] <= max_cfips]
    #raw = new_cfips(raw, param, max_cfips)
    posle_sglashivfnia(raw, rezult)

if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    #raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    # glavnaia(raw, rezult)

    #raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_otdelno.csv")

    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_cens.csv")

    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0
    #glavnaia(raw, rezult) # 1.379034  0.010546  146.796656
    posle_sglashivfnia(raw, rezult)
