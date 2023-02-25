import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno

# загрузка данных
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    census = pd.read_csv("C:\\kaggle\\МикроБизнес\\census_starter.csv")
    new_data = pd.read_csv("C:\\kaggle\\МикроБизнес\\revealed_test.csv")
    return train, test, census, new_data

# загрузка данных переписи
def load_perepis():
    # загузка из ноутбука
    # df2020 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv', usecols=COLS)
    # df2021 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv',usecols=COLS)
    df2020 = pd.read_csv("C:\\kaggle\\МикроБизнес\\ACSST5Y2020.S0101-Data.csv", dtype = 'object')
    df2021 = pd.read_csv("C:\\kaggle\\МикроБизнес\\ACSST5Y2021.S0101-Data.csv", dtype = 'object')

    df2020 = df2020[['GEO_ID', 'S0101_C01_026E']]
    df2020 = df2020.iloc[1:]
    df2020 = df2020.astype({'S0101_C01_026E':'int'})

    df2021 = df2021[['GEO_ID', 'S0101_C01_026E']]
    df2021 = df2021.iloc[1:]
    df2021 = df2021.astype({'S0101_C01_026E':'int'})


    df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
    df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]))

    df2020['mnoshitel'] = df2020['S0101_C01_026E'] / df2021['S0101_C01_026E']
    df2020 = df2020[['cfips','mnoshitel']]
    df2020.set_index('cfips', inplace=True)
    return df2020

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
    #raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    return raw

# добавление в raw данных за ноябрь и декабрь
def new_data_in_raw(raw, new_data):
    for index, row_new in new_data.iterrows():
        maska = row_new['row_id'] == raw['row_id']
        raw.loc[maska, 'microbusiness_density'] = row_new['microbusiness_density']
        raw.loc[maska, 'active'] = row_new['active']
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

def load_data():
    train, test, census, new_data = start()  # загрузка начальных файлов
    raw = maceraw(train, test)  # объединенный массив трейна и теста, создание объединенного raw
    raw = new_data_in_raw(raw, new_data)  # присоединение новых данных к raw
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw.csv", index=False)
    raw = censusdef(census, raw)  # присоединение файла census к raw
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_cens.csv", index=False)
    return raw

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

# считаем сколько предсказаний лучше
def baz_otchet(raw, maska):
    global start_val, stop_val, minimum
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw.lastactive > minimum)
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[maska].groupby(['cfips', 'dcount'])[['error', 'error_last','lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() >= 0.5

    hor = (raw.loc[maska, 'error_last'] > raw.loc[maska, 'error']).sum()
    ploh = (raw.loc[maska, 'error'] > raw.loc[maska, 'error_last']).sum()
    notdif = len(raw[maska]) - hor - ploh

    return len(seria_dt), ploh, hor, notdif

# считаем сколько предсказаний лучше после 1-й модели
def otchet(raw):
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw.lastactive > minimum)
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    dif_err = err_last - err_mod  # положительная - хорошо
    print('Предсказано моделью. Ошибка SMAPE:', err_mod, 'Разность c "=" (чем >, тем лучше):', dif_err)
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l, ploh, hor, notdif = baz_otchet(raw, maska)
    print('количство cfips предсказанных хуже чем моделью =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)

# считаем сколько предсказаний лучше после 2-й модели
def model2_otchet(raw):
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw.lastactive > minimum)
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l, ploh, hor, notdif = baz_otchet(raw, maska)
    print('')
    print('------ Отчет после выполнения оптимизации 2-й модели ------')
    print('количство cfips предсказанных хуже чем моделью № 1 =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)
    raw['error'] = vsmape(raw['microbusiness_density'], raw['y_inegr'])
    l, ploh, hor, notdif = baz_otchet(raw, maska)
    print('количство cfips предсказанных хуже чем моделью № 2 =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)

# Модифицированный мульт. Оптимизированн конкретно под январь - подгонка
def mult_ianvar41(raw):
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    maska = (raw.dcount >= 38) & (raw.dcount <= 40)
    train_data = raw[maska].copy()
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [1.00, 1.002, 1.004]}
    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag1'] * mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred'])
    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    raw['ymult'] = raw['mbd_lag1'] * raw['best_mult']
    raw['ypred'] = raw['ymult']
    # raw['ypred'].fillna(0, inplace = True)
    return raw

# Универсальный мульт. Хороший для последних месяцев в среднем
def mult(raw, mes_val):
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    maska = (raw.dcount >= mes_val - 12) & (raw.dcount < mes_val)
    train_data = raw[maska].copy()
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [1.00, 1.001, 1.002, 1.003]}
    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag1'] * mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred'])
    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ymult'] = raw.loc[maska,'mbd_lag1'] * raw.loc[maska,'best_mult']
    raw['ypred'] = raw['ymult']
    # raw['ypred'].fillna(0, inplace = True)
    raw.drop('best_mult', axis=1, inplace=True)
    return raw

def mult_po_mesiacam(raw):
    global start_val, stop_model
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(12, stop_model + 1)):
        if mes_val == 41:
            raw = mult_ianvar41(raw)
        else:
            raw = mult(raw, mes_val)
    print('Модель mult')
    otchet(raw)
    return raw

# создание нового таргета
def new_target(raw, param):
    # 'target' ='microbusiness_density' текущего месяца делить на 'ymult' - 1
    raw['target'] = raw['microbusiness_density'] / raw['ymult'] - 1
    raw.loc[(raw['microbusiness_density'] == 0)|(raw['target'] > 10),'target'] = 0
    raw['target'].fillna(0, inplace=True)
    raw.loc[(raw['target'] > param), 'target'] = param
    raw.loc[(raw['target'] < -param), 'target'] = -param
    return raw

# создание лагов
def build_lag(raw): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 17):
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    #создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
 #   raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1', 'median_hh_inc', 'month']
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки
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

# модель применяемая в первом процессе оптимизации
def vsia_model(raw, mes_1, mes_val, train_col, param=0):
    df = raw[raw['lastactive'] > minimum]
    # получение трайна и 'y' (игрик) для модели
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    model = xgb.XGBRegressor(
        tree_method="hist",
        n_estimators=850,
        learning_rate=0.0071, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=3,
    )
    # обучение модели
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)
    return y_pred

# формирование колонки 'ypred' по результатам процесса оптимизации
def post_model(raw, y_pred, mes_val):
    maska = raw.dcount == mes_val
    # преобразовываем target в 'microbusiness_density'
    y_pred = y_pred + 1
    y_pred = raw.loc[maska, 'ymult'] * y_pred
    raw.loc[maska, 'ypred'] = y_pred  # сохраняем результат обработки
    return raw

# помесячная оптимизация в 1-м процессе
def modeli_po_mesiacam(raw, train_col):
    global start_val, stop_model
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_model включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):
        mes_1 = 12 # первый месяц для трайна
        y_pred = vsia_model(raw, mes_1, mes_val, train_col, param=0)
        raw = post_model(raw, y_pred, mes_val)
    otchet(raw)
    return raw

def oll_1_model(raw, param):
    raw = new_target(raw, param)
    # здесь должен начинаться цикл по количеству лагов
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
    # запускаем модель по всем месяцам
    raw = modeli_po_mesiacam(raw, train_col)
    return raw



# обработка января 2023 в связи с изменением численности населения по переписи
def january(raw, mnoshitel, colum): # colum имя колонки которую меняем
    raw = raw.join(mnoshitel, on='cfips')
    maska = (raw["year"]==2023)&(raw["month"]==1)
    raw.loc[maska, colum] = raw.loc[maska, colum] * raw.loc[maska, 'mnoshitel']
    raw.drop(columns = 'mnoshitel' , inplace = True)
    return raw

def otvet(raw):
    raw = january(raw, mnoshitel, 'ypred')
    test = raw[raw.istest == 1].copy()
    maska = test['microbusiness_density'].isnull()
    test.loc[maska,'microbusiness_density'] = test.loc[maska,'ypred']
    # test['microbusiness_density'] = test['ypred']
    test = test[['row_id', 'cfips', 'microbusiness_density']]
    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\main0.csv', index=False)

# инициализация после загрузки
def init_after_load(raw):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0
    raw['ymult'] = 0
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    return raw

if __name__ == "__main__":
    minimum = 60 # ниже этого значения 'lastactive' считаем что 'microbusiness_density' не меняется
    # глобальные переменные

    stop_model = 41 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 40  # последний месяц валидации до которого проверяем модель
    main_start_val = 30 # первый месяц окончательной валидации с которого проверяем модель

    start_val = 28  # первый месяц предварительной валидации с которого проверяем модель
    pd.options.display.width = 0 # для печати

    # НАЧАЛЬНАЯ ЗАГРУЗКА
    # raw = load_data()
    mnoshitel = load_perepis() # загрузка данных переписи

    # ЗАГРУЗКА ДАННЫХ
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_cens.csv")

    raw = init_after_load(raw) # инициализация после загрузки

    raw = mult_po_mesiacam(raw)
    otvet(raw)
    oll_1_model(raw, param=0.005)
    otvet(raw)
