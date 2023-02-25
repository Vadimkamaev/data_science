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
def new_target(raw, param):
    # 'target' ='microbusiness_density' предыдущего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    # 'target' ='microbusiness_density' текущего месяца делить на предыдущего месяца - 1
    raw['target'] = raw['microbusiness_density'] / raw['target'] - 1
    raw.loc[(raw['microbusiness_density'] == 0)|(raw['target'] > 10),'target'] = 0
    raw['target'].fillna(0, inplace=True)

    porog = 0.0054
    raw.loc[(raw['target'] < - porog), 'target'] = - porog
    raw.loc[(raw['target'] > porog), 'target'] = porog # 1.379034  0.010546  146.796656

    # raw.loc[(raw['target'] > param), 'target'] = param
    # raw.loc[(raw['target'] < -param), 'target'] = -param
    return raw

# создание лагов  error 1.379034  0.010546  146.796656
def build_lag(raw, param): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 12): # 1.379084  0.010497  146.812119
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    # # создаем скользящие средние.
    for i in [3, 4, 14, 17]:
        nam = f'EMA_{i}'
        EMA = pd.Series(raw['target_lag1'].ewm(span=i, adjust=False, min_periods=1).mean(), name=nam)
        raw[nam] = EMA
        train_col += [nam]
    # #создаем значения сумм окон для lag = 1
    for i in [param]: #без 1.736938 -0.005141  178.270746
        nam = f'roll_{i}'
        # сгруппированно по 'cfips' 1-й лаг трансформируем - считаем сумму в окнах
        ROLL = raw.groupby('cfips')['target_lag1'].transform(lambda s: s.rolling(i, min_periods=1).sum())
        raw[nam] = ROLL
        train_col += [nam]
    # создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1']
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки 1.408914 -0.097942
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test

# создание блек листа 'cfips'
def mace_blac_list(raw, mes_1, mes_val, blac_cfips):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # создаем датафрейм со столбцами error и error_last
    dt = raw.loc[(raw.dcount >= mes_1) & (raw.dcount <= mes_val)].groupby(['cfips', 'dcount'])[['error', 'error_last']].last()
    # преобразуем dt в серию булевых значений 'miss'
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    ser_err = dt.groupby('cfips')['error'].mean()
    ser_error_last = dt.groupby('cfips')['error_last'].mean()
    df_dt = pd.DataFrame({'cfips': seria_dt.index, 'miss': seria_dt, 'hit':(ser_error_last-ser_err)})
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))
    # if not_blec == 1: # 1 - без блек листа
    #     df_dt.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)
    return blac_cfips

# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_cfips, max_cfips, lastactive, param=1):
    # маска при которой была валидация и по которой сверяем ошибки
    # maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
    #         (raw.cfips <= max_cfips) & (raw.lastactive > lastactive)  & (~raw['cfips'].isin(blac_cfips))
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val)
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)

    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)

    ypred_target = raw.loc[maska, 'ypred_target']
    tar_get = raw.loc[maska, 'target']
    err_target = smape(tar_get, ypred_target)
    print('Ошибка target:', err_target)

    raw.loc[maska, 'error']=vsmape(target, ypred)
    raw.loc[maska, 'error_last']=smape(target, mbd_lag1)
    # dfnew = raw[(raw["dcount"] == 38) | (raw["dcount"] == 37)]
    # dfnew = dfnew[['row_id', 'cfips', 'microbusiness_density', 'target', 'ypred_target', 'ypred', 'error', 'error_last']]
    dif_err = err_last - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [lastactive, param, err_mod, dif_err, err_target]
    return rezult

def vsia_model1(raw, mes_1, mes_val, train_col, znachenie, param=1):
    # получение трайна и 'y' (игрик) для модели
    X_train, y_train = train_and_y(raw, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    # Создаем модель
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate= 0.0071, # важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    # Предсказываем
    y_pred = model.predict(X_test)

    #raw.loc[(raw.dcount == mes_val)&(raw.istest == 0) , 'ypred_target'] = y_pred
    y_pred = y_pred + 1
    y_pred = raw.loc[raw.dcount == mes_val, 'mbd_lag1'] * y_pred



    #Предсказано иссдедуемой моделью. Ошибка SMAPE: 1.5717414657304203
    # сохраняем результат обработки одного цикла
    raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred
    # прибавляем значимость столбцов новой модели к значениям предыдущих
    znachenie['importance'] = znachenie['importance'] + model.feature_importances_.ravel()
    return raw, znachenie

def new_cfips(raw, lastactive, max_cfips):
    kol = 3 # количество элементов которое суммируется
    df = raw[raw['lastactive'] < lastactive].copy()
    df.sort_values('lastactive', ascending=False, inplace=True)
    if len(df.index) == 0:
        return raw
    list_cfips = df['cfips'].unique() #  массив уникальных cfips, которые группируются
    lec = len(list_cfips) # количество уникальных cfips, которые группируются
    le = lec + kol - lec%kol
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

    for l in range(le):
        gen_cfips = max_cfips+l+1 # номер нового сгенерированного cfips
        dfnew = dfcop.copy()
        dfnew['cfips'] = gen_cfips
        # maska0 = raw['cfips'] == matrica[0, l]
        # maska1 = raw['cfips'] == matrica[1, l]
        # maska2 = raw['cfips'] == matrica[2, l]
        df0 = raw[raw['cfips'] == matrica[0, l]].copy()
        df1 = raw[raw['cfips'] == matrica[1, l]].copy()
        df2 = raw[raw['cfips'] == matrica[2, l]].copy()
        df0.reset_index(drop=True, inplace=True)
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        dfnew['active'] = df0['active'] + df1['active'] + df2['active']
        dfnew['lastactive'] = df0['lastactive'] + df1['lastactive'] + df2['lastactive']
        n0 = df0['Population'][0]
        n1 = df1['Population'][0]
        n2 = df2['Population'][0]
        nas = n0 + n1 + n2
        dfnew['microbusiness_density'] = dfnew['active']/nas
        dfnew['proc_covill'] = (df0['proc_covill']*n0 + df1['proc_covill']*n1 + df2['proc_covill']*n2)/nas
        dfnew['pct_college'] = (df0['pct_college']*n0 + df1['pct_college']*n1 + df2['pct_college']*n2)/nas
        dfnew['median_hh_inc'] = (df0['median_hh_inc']*n0 + df1['median_hh_inc']*n1 + df2['median_hh_inc']*n2)/nas

        if (n0 == 0)|(n1 == 0):
            print(n0,n1)
            df = raw[raw['Population'] == 0]
            okno.vewdf(dfnew)
        #dfnew.fillna(method="ffill", inplace=True)

        raw = pd.concat([raw,dfnew], ignore_index=True)
    return raw

def posle_sglashivfnia(raw, rezult, lastactive = 0):
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    blac_cfips = []
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    #raw = new_cfips(raw, 10, max_cfips)

    raw['ypred_target']=0

    # возможные поля 'Population', 'proc_covdeat', 'pct_bb', 'pct_foreign_born', 'pct_it_workers', 'unemploy'
    start_val = 6  # первый месяц валидации с которого проверяем модель
    stop_val = 38  # последний месяц валидации до которого проверяем модель

    # цикл по оптимизируемому параметру модели
    for param in range(2,12,1):
        # создаем новый таргет
        raw = new_target(raw, param/10000)
        raw, train_col = build_lag(raw, param)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
        # здесь должен начинаться цикл перебирающий все комбинации из списка полей
        znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
        # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
        start_time = datetime.now()  # время начала работы модели
        for mes_val in tqdm(range(start_val, stop_val + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
            # здесь должен начинаться цикл по перебору номера первого месяца для трайна
            mes_1 = 2#4
            raw, znachenie = vsia_model1(raw, mes_1, mes_val, train_col, znachenie, param)

        df = raw.sort_values(by = 'ypred_target')

        blac_cfips = mace_blac_list(raw, start_val, mes_val, blac_cfips)
        print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, start_val, stop_val, rezult, blac_cfips, max_cfips, lastactive, param)
    # здесь заканчиваются циклы оптимизации
    rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim1.csv", index=False)
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_vmeste_forest.csv", index=False)
    print('Значимость колонок трайна в модели')
    znachenie.sort_values(by='importance', ascending=False, inplace=True)
    print(znachenie)
    rezult.sort_values(by='dif_err', ascending=False, inplace=True)
    print(rezult.head(40))


# МОДЕЛЬ.
def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    max_cfips = raw['cfips'].max() # максимальная реальна 'cfips', больше неё фиктивные 'cfips'

    # raw = raw[raw['cfips'] <= max_cfips]
    #raw = new_cfips(raw, 10, max_cfips)

    posle_sglashivfnia(raw, rezult)

if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0.csv")
    rezult = pd.DataFrame(columns=['model','param', 'error', 'dif_err', 'err_target'])
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    glavnaia(raw, rezult)
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_otdelno.csv")
    # raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    # raw['ypred'] = 0
    # posle_sglashivfnia(raw, rezult)
